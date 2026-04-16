"""
ConsumerAgent — the primary simulation actor.

Each agent holds a persistent AgentState and uses the LLM to make
purchase decisions based on their profile, beliefs, memories, and
social context.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, TYPE_CHECKING

from core.events import Event, EventType
from core.state import (
    AgentState, AgentProfile, BeliefSystem, Memory, MemoryType,
    ResourcePool, VulnerabilitySet,
)
from environment.product import Product
from llm.dispatcher import dispatcher
from llm.prompts import render_prompt
from llm.schemas import PurchaseDecision, PurchaseDecisionOutput, WOMOutput

if TYPE_CHECKING:
    from agents.experience import ExperienceProfile


# ─────────────────────────────────────────────────────────────
# CONSUMER AGENT
# ─────────────────────────────────────────────────────────────

# Adopter tier baseline parameters — risk tolerance, purchase intent
TIER_DEFAULTS = {
    "innovator":      {"risk_tolerance": 0.9, "purchase_intent": 0.7, "social_capital": 0.8},
    "early_adopter":  {"risk_tolerance": 0.7, "purchase_intent": 0.5, "social_capital": 0.7},
    "early_majority": {"risk_tolerance": 0.5, "purchase_intent": 0.3, "social_capital": 0.5},
    "late_majority":  {"risk_tolerance": 0.3, "purchase_intent": 0.2, "social_capital": 0.4},
    "laggard":        {"risk_tolerance": 0.1, "purchase_intent": 0.1, "social_capital": 0.3},
}


class ConsumerAgent:

    def __init__(self, profile: AgentProfile):
        defaults = TIER_DEFAULTS.get(profile.adopter_tier, TIER_DEFAULTS["early_majority"])

        self.state = AgentState(
            profile=profile,
            resources=ResourcePool(
                budget=profile.income_amount,
                social_capital=defaults["social_capital"],
            ),
            beliefs=BeliefSystem(
                purchase_intent=defaults["purchase_intent"],
            ),
            vulnerabilities=VulnerabilitySet(
                risk_tolerance=defaults["risk_tolerance"],
            ),
        )

    # ── Core decision ─────────────────────────────────────────

    def _build_purchase_prompt(
        self,
        product: Product,
        adopted_ids: set,
        current_step: int,
        *,
        step_duration_days: int = 7,
        market_context: Optional[str] = None,
    ) -> str:
        days_since_launch = (current_step - 1) * step_duration_days
        return render_prompt(
            "purchase_decision.user",
            name=self.state.profile.name,
            age=self.state.profile.age,
            occupation=self.state.profile.occupation,
            income_level=self.state.profile.income_level,
            income_amount=self.state.profile.income_amount,
            adopter_tier=self.state.profile.adopter_tier,
            lifestyle_tags=", ".join(self.state.profile.lifestyle_tags),
            cognitive_style=self.state.profile.cognitive_style,
            goals=self._format_goals(),
            beliefs=self._format_beliefs(),
            memories=self._format_memories(current_step=current_step),
            adopted_friends=len(self.state.relationships.get_adopted_neighbors(adopted_ids)),
            network_adoption_rate=self._network_adoption_rate(adopted_ids),
            product_name=product.name,
            brand=product.brand,
            price=product.price,
            features=", ".join(product.features),
            quality=product.quality,
            brand_reputation=product.brand_reputation,
            channels=", ".join(product.distribution_channels),
            risk_tolerance=self.state.vulnerabilities.risk_tolerance,
            loss_aversion=self.state.vulnerabilities.loss_aversion,
            budget=self.state.resources.budget,
            market_context=market_context or "",
            current_step=current_step,
            step_duration_days=step_duration_days,
            days_since_launch=days_since_launch,
            min_defer=current_step + 1,
            max_defer=current_step + 5,
        )

    def _build_purchase_event(
        self, result: PurchaseDecisionOutput, product: Product, current_step: int
    ) -> Event:
        return Event(
            event_type=EventType.PURCHASE_DECISION,
            timestamp=current_step,
            source=self.state.profile.agent_id,
            content={
                "product_id": product.product_id,
                "decision": result.decision.value,
                "confidence": result.confidence,
                "price_acceptable": result.price_acceptable,
                "reasoning": result.reasoning,
            },
        )

    async def decide_purchase(
        self,
        product: Product,
        adopted_ids: set,
        current_step: int,
        *,
        step_duration_days: int = 7,
        temperature: float = 0.7,
        market_context: Optional[str] = None,
    ) -> tuple[PurchaseDecisionOutput, Event]:
        """
        Ask the LLM whether this agent buys, defers, or rejects the product.
        Returns the parsed decision and the corresponding simulation Event.
        """
        user_prompt = self._build_purchase_prompt(
            product, adopted_ids, current_step,
            step_duration_days=step_duration_days,
            market_context=market_context,
        )

        result: PurchaseDecisionOutput = await dispatcher.acall(
            system_prompt=render_prompt("purchase_decision.system"),
            user_prompt=user_prompt,
            schema=PurchaseDecisionOutput,
            max_tokens=4096,
            temperature=temperature,
        )

        self._apply_decision(result, product, current_step)
        return result, self._build_purchase_event(result, product, current_step)

    def _build_wom_prompt(
        self,
        product: Product,
        experience: "ExperienceProfile",
        price_paid: float,
    ) -> str:
        expected_price = product.price
        if price_paid <= expected_price * 0.9:
            price_vs_expectation = "better than expected (paid less)"
        elif price_paid >= expected_price * 1.1:
            price_vs_expectation = "worse than expected (paid more)"
        else:
            price_vs_expectation = "as expected"

        praises_block = (
            "\n".join(f"  - \"{p}\"" for p in experience.surfaced_praises)
            or "  (none surfaced)"
        )
        defects_block = (
            "\n".join(f"  - \"{d}\"" for d in experience.surfaced_defects)
            or "  (none surfaced)"
        )

        influencer_note = (
            "- You are a recognised opinion leader in your network — your reviews carry weight."
            if self.state.profile.is_influencer else ""
        )

        return render_prompt(
            "wom_generation.user",
            name=self.state.profile.name,
            adopter_tier=self.state.profile.adopter_tier,
            social_capital=self.state.resources.social_capital,
            influencer_note=influencer_note,
            product_name=product.name,
            quality_rating=experience.overall_score,
            price_vs_expectation=price_vs_expectation,
            overall_experience=experience.overall_experience,
            praises_block=praises_block,
            defects_block=defects_block,
        )

    def _build_wom_event(
        self, result: WOMOutput, product: Product, current_step: int
    ) -> Optional[Event]:
        if result.target_audience.value == "nobody" or result.share_probability <= 0.3:
            return None
        return Event(
            event_type=EventType.WORD_OF_MOUTH,
            timestamp=current_step + 1,
            source=self.state.profile.agent_id,
            content={
                "product_id": product.product_id,
                "sentiment": result.sentiment.value,
                "message": result.message_content,
                "emotional_intensity": result.emotional_intensity,
                "target_audience": result.target_audience.value,
            },
        )

    async def generate_wom(
        self,
        product: Product,
        experience: "ExperienceProfile",
        price_paid: float,
        current_step: int,
    ) -> tuple[WOMOutput, Optional[Event]]:
        """
        After purchase, decide whether and how to share the experience.
        Returns WOM output and an event (None if agent decides not to share).

        The ExperienceProfile carries the buyer's grounded, per-dimension
        view of the product (including surfaced defects and praises).
        """
        user_prompt = self._build_wom_prompt(product, experience, price_paid)

        result: WOMOutput = await dispatcher.acall(
            system_prompt=render_prompt("wom_generation.system"),
            user_prompt=user_prompt,
            schema=WOMOutput,
            max_tokens=4096,
        )

        return result, self._build_wom_event(result, product, current_step)

    # ── WOM reception ────────────────────────────────────────

    async def receive_wom(
        self,
        source_id: str,
        sentiment: str,
        message: str,
        trust: float,
        step: int,
    ) -> None:
        """Update beliefs and memories when word-of-mouth arrives.

        Thin wrapper around the default ``LinearBeliefUpdater`` — kept
        for ad-hoc callers and tests.  The simulation runner drives
        updates through its configured ``BeliefUpdater`` directly and
        does NOT go through this method.
        """
        from agents.belief import LinearBeliefUpdater
        await LinearBeliefUpdater().update(
            target=self,
            source_id=source_id,
            sentiment=sentiment,
            message=message,
            trust=trust,
            step=step,
        )

    # ── State updates ─────────────────────────────────────────

    def _apply_decision(
        self, result: PurchaseDecisionOutput, product: Product, step: int
    ) -> None:
        if result.decision == PurchaseDecision.BUY:
            self.state.has_purchased = True
            self.state.purchase_step = step
            self.state.resources.spent_budget += product.price
            self.state.resources.budget -= product.price

            self.state.memories.add(Memory(
                memory_id=str(uuid.uuid4())[:8],
                content=f"Purchased {product.name} for ¥{product.price:,.0f}",
                memory_type=MemoryType.EPISODIC,
                importance=0.8,
                timestamp=step,
                emotional_valence=0.6,
                context=f"Decision confidence: {result.confidence:.0%}",
                lessons=result.reasoning[:200],
            ))

        elif result.decision == PurchaseDecision.REJECT:
            self.state.memories.add(Memory(
                memory_id=str(uuid.uuid4())[:8],
                content=f"Decided against buying {product.name}. Concerns: {', '.join(result.key_concerns)}",
                memory_type=MemoryType.EPISODIC,
                importance=0.5,
                timestamp=step,
                emotional_valence=-0.1,
                lessons=result.reasoning[:200],
            ))

        # Update purchase intent based on confidence and direction
        if result.decision == PurchaseDecision.BUY:
            self.state.beliefs.purchase_intent = min(1.0, self.state.beliefs.purchase_intent + 0.2)
        elif result.decision == PurchaseDecision.REJECT:
            self.state.beliefs.purchase_intent = max(0.0, self.state.beliefs.purchase_intent - 0.2)

        self.state.current_step = step

    # ── Prompt helpers ────────────────────────────────────────

    def _format_goals(self) -> str:
        if not self.state.goals:
            return "No explicit goals set."
        return "\n".join(
            f"- [{g.priority:.0%} priority] {g.description} ({g.status.value})"
            for g in self.state.goals
        )

    def _format_beliefs(self) -> str:
        if not self.state.beliefs.beliefs:
            return f"No prior beliefs. Overall sentiment: {self.state.beliefs.overall_sentiment:+.1f}"
        lines = [f"Overall sentiment: {self.state.beliefs.overall_sentiment:+.1f}"]
        for b in self.state.beliefs.beliefs[-5:]:  # last 5 beliefs
            lines.append(f"- [{b.confidence:.0%} confident] {b.subject} {b.predicate} (source: {b.source})")
        return "\n".join(lines)

    def _format_memories(
        self,
        current_step: int,
        decay_factor: float = 0.85,
        min_effective_importance: float = 0.05,
        keep_top: int = 5,
    ) -> str:
        """Render memories with time-decayed importance.

        Older memories are down-weighted by ``decay_factor ** age``
        (Phase 1.5 upgrade #9, "awareness decay") so reviews from many
        steps ago do not influence current decisions as strongly as
        fresh ones.  Memories whose effective importance falls below
        ``min_effective_importance`` are dropped from the prompt entirely.
        """
        memories = self.state.memories.retrieve_recent(
            n=keep_top, current_step=current_step, decay_factor=decay_factor,
        )
        if not memories:
            return "No relevant memories."

        lines = []
        for m in memories:
            age = max(0, current_step - m.timestamp)
            weight = decay_factor ** age
            effective = m.importance * weight
            if effective < min_effective_importance:
                continue
            if age > 0:
                lines.append(
                    f"- [step {m.timestamp}, age {age}] {m.content} "
                    f"(importance: {effective:.0%}, decayed from {m.importance:.0%})"
                )
            else:
                lines.append(
                    f"- [step {m.timestamp}] {m.content} (importance: {m.importance:.0%})"
                )
        return "\n".join(lines) if lines else "No fresh memories (all faded)."

    def _network_adoption_rate(self, adopted_ids: set) -> float:
        neighbors = self.state.relationships.get_neighbors()
        if not neighbors:
            return 0.0
        adopted = sum(1 for n in neighbors if n in adopted_ids)
        return adopted / len(neighbors)

    @property
    def agent_id(self) -> str:
        return self.state.profile.agent_id
