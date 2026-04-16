"""
Belief updaters — Layer 2 of the two-layer WOM pipeline.

Layer 1 (``WOMEngine.compute_receptions`` in ``environment/wom.py``) is a
cheap code gate that decides *which* neighbours receive a message.  Layer
2 — this module — decides *how the target's beliefs update* once a message
has landed.

Two implementations ship by default:

* ``LinearBeliefUpdater`` — closed-form, asymmetric formula.  Fast, no
  LLM calls, captures negativity-bias at the parameter level (negative
  WOM moves sentiment harder than equally-intense positive WOM).
* ``LLMBeliefUpdater`` — semantic.  Sends the target's current belief
  state + the incoming message to the LLM and asks for nuanced deltas.
  Expensive — one call per reception — but catches effects that a linear
  formula cannot (e.g. "I trust Alice's opinion on tech but not on
  apparel," or "this message contradicts a belief I'm already confident
  about, so it barely moves me").

Swap via ``SimulationConfig.belief_updater``.  Default is
``LinearBeliefUpdater`` (no LLM cost).

See ``docs/30_two_layer_wom.md`` for the full design rationale.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.state import Belief, Memory, MemoryType
from llm.dispatcher import dispatcher
from llm.prompts import render_prompt
from llm.schemas import BeliefUpdateOutput

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# ABSTRACT INTERFACE
# ─────────────────────────────────────────────────────────────

class BeliefUpdater(ABC):
    """Abstract base for Layer-2 WOM belief updates.

    Implementations mutate the *target* agent's ``BeliefSystem`` and
    ``MemoryStore`` based on the incoming message and the source's
    trust.  The update method is async so that LLM-based impls can run
    in parallel across a step's receptions.
    """

    @abstractmethod
    async def update(
        self,
        target: "ConsumerAgent",
        source_id: str,
        sentiment: str,
        message: str,
        trust: float,
        step: int,
    ) -> None:
        """Apply a belief update to *target* in response to a received WOM."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for reports."""


# ─────────────────────────────────────────────────────────────
# LINEAR (CLOSED-FORM) UPDATER
# ─────────────────────────────────────────────────────────────

class LinearBeliefUpdater(BeliefUpdater):
    """Closed-form belief update with asymmetric positive/negative deltas.

    Mirrors the Phase 1 ``receive_wom`` logic but makes the positive /
    negative asymmetry explicit.  Consumer-research literature
    consistently shows that negative word-of-mouth moves attitudes more
    than equally-intense positive WOM (negativity bias + loss aversion).

    Parameters
    ----------
    positive_sentiment_delta : float
        Raw sentiment shift applied when a positive WOM arrives.
    negative_sentiment_delta : float
        Raw sentiment shift applied when a negative WOM arrives.  By
        default ~2× larger in magnitude than the positive delta.
    intent_multiplier : float
        Scales sentiment-delta into purchase-intent delta.  Keeps intent
        moves smaller than sentiment moves — people don't flip from
        "curious" to "buying" on a single review.
    memory_importance_scale : float
        Multiplied by trust to produce the memory's importance score.
    """

    def __init__(
        self,
        positive_sentiment_delta: float = 0.10,
        negative_sentiment_delta: float = -0.22,
        intent_multiplier: float = 0.15,
        memory_importance_scale: float = 0.6,
    ) -> None:
        self.positive_sentiment_delta = positive_sentiment_delta
        self.negative_sentiment_delta = negative_sentiment_delta
        self.intent_multiplier = intent_multiplier
        self.memory_importance_scale = memory_importance_scale

    async def update(
        self,
        target: "ConsumerAgent",
        source_id: str,
        sentiment: str,
        message: str,
        trust: float,
        step: int,
    ) -> None:
        raw_delta = self._raw_sentiment_delta(sentiment)
        sentiment_delta = raw_delta * trust
        intent_delta = sentiment_delta * self.intent_multiplier

        _apply_sentiment(target, sentiment_delta)
        _apply_intent(target, intent_delta)
        _append_memory(
            target,
            message=message,
            sentiment=sentiment,
            trust=trust,
            step=step,
            importance_scale=self.memory_importance_scale,
        )
        _append_belief(
            target,
            sentiment=sentiment,
            message=message,
            trust=trust,
            step=step,
            confidence=round(trust, 2),
        )

    def _raw_sentiment_delta(self, sentiment: str) -> float:
        if sentiment == "positive":
            return self.positive_sentiment_delta
        if sentiment == "negative":
            return self.negative_sentiment_delta
        return 0.0

    def describe(self) -> str:
        return (
            f"LinearBeliefUpdater(pos={self.positive_sentiment_delta:+.2f}, "
            f"neg={self.negative_sentiment_delta:+.2f}, "
            f"intent*{self.intent_multiplier})"
        )


# ─────────────────────────────────────────────────────────────
# LLM (SEMANTIC) UPDATER
# ─────────────────────────────────────────────────────────────

class LLMBeliefUpdater(BeliefUpdater):
    """LLM-based belief update.

    One LLM call per reception.  Used when fidelity matters more than
    cost (small-scale diagnostic runs, fine-tuning seed cases).
    """

    def __init__(self, max_tokens: int = 1024, temperature: float = 0.5) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def update(
        self,
        target: "ConsumerAgent",
        source_id: str,
        sentiment: str,
        message: str,
        trust: float,
        step: int,
    ) -> None:
        user_prompt = self._build_prompt(target, sentiment, message, trust)

        try:
            result: BeliefUpdateOutput = await dispatcher.acall(
                system_prompt=render_prompt("belief_update.system"),
                user_prompt=user_prompt,
                schema=BeliefUpdateOutput,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            # LLM failure → fall back to no-op for this reception.  We
            # intentionally do NOT silently run the linear updater here:
            # a crash in the LLM path should surface in logs, not masquerade
            # as a successful semantic update.
            print(f"    [warn] LLMBeliefUpdater failed ({exc}); skipping update.")
            return

        _apply_sentiment(target, result.sentiment_delta)
        _apply_intent(target, result.intent_delta)
        _append_memory(
            target,
            message=message,
            sentiment=sentiment,
            trust=trust,
            step=step,
            importance_override=result.memory_importance,
            valence_override=result.memory_valence,
        )
        target.state.beliefs.add_or_update(Belief(
            belief_id=str(uuid.uuid4())[:8],
            subject="target_product",
            predicate=result.new_belief_predicate,
            confidence=result.belief_confidence,
            source="word_of_mouth",
            timestamp=step,
            evidence=[message[:200]],
        ))

    def _build_prompt(
        self,
        target: "ConsumerAgent",
        sentiment: str,
        message: str,
        trust: float,
    ) -> str:
        p = target.state.profile
        beliefs = target.state.beliefs.beliefs[-4:]
        if beliefs:
            beliefs_block = "\n".join(
                f"  - [{b.confidence:.0%}] {b.subject} {b.predicate} (src: {b.source})"
                for b in beliefs
            )
        else:
            beliefs_block = "  (no prior product beliefs)"

        return _BELIEF_UPDATE_USER_PROMPT.format(
            name=p.name,
            adopter_tier=p.adopter_tier,
            current_sentiment=target.state.beliefs.overall_sentiment,
            current_intent=target.state.beliefs.purchase_intent,
            beliefs_block=beliefs_block,
            trust=trust,
            sentiment=sentiment,
            message=message.replace('"', "'"),
        )

    def describe(self) -> str:
        return f"LLMBeliefUpdater(temp={self.temperature})"


# ─────────────────────────────────────────────────────────────
# SHARED MUTATION HELPERS
# ─────────────────────────────────────────────────────────────

def _apply_sentiment(target: "ConsumerAgent", delta: float) -> None:
    target.state.beliefs.overall_sentiment = max(
        -1.0, min(1.0, target.state.beliefs.overall_sentiment + delta)
    )


def _apply_intent(target: "ConsumerAgent", delta: float) -> None:
    target.state.beliefs.purchase_intent = max(
        0.0, min(1.0, target.state.beliefs.purchase_intent + delta)
    )


def _append_memory(
    target: "ConsumerAgent",
    message: str,
    sentiment: str,
    trust: float,
    step: int,
    importance_scale: float = 0.6,
    importance_override: float | None = None,
    valence_override: float | None = None,
) -> None:
    if importance_override is not None:
        importance = round(importance_override, 2)
    else:
        importance = round(trust * importance_scale, 2)

    if valence_override is not None:
        valence = valence_override
    else:
        base = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}.get(sentiment, 0.0)
        valence = base * trust

    target.state.memories.add(Memory(
        memory_id=str(uuid.uuid4())[:8],
        content=f"Heard from contact: {message[:120]}",
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        timestamp=step,
        emotional_valence=valence,
        context=f"Word-of-mouth via social network (trust: {trust:.2f})",
    ))


def _append_belief(
    target: "ConsumerAgent",
    sentiment: str,
    message: str,
    trust: float,
    step: int,
    confidence: float,
) -> None:
    target.state.beliefs.add_or_update(Belief(
        belief_id=str(uuid.uuid4())[:8],
        subject="target_product",
        predicate=f"{sentiment} review from trusted contact",
        confidence=confidence,
        source="word_of_mouth",
        timestamp=step,
        evidence=[message[:200]],
    ))
