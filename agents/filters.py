"""
Decision filters — the "System 1" (fast, code-only) pre-filter.

Motivation
----------
Every LLM call costs time and money.  Many decisions can be resolved by
cheap code checks: an agent with no budget will not buy; a late_majority
agent with zero social proof will defer, no matter what the LLM reasons.
Running the LLM on these cases is waste.

This module defines a pluggable ``DecisionFilter`` ABC.  The default impl
``System1Filter`` applies two gates in order:

  1. **Budget gate** — reject if the product is prohibitively expensive
     for the agent given a configurable headroom factor.
  2. **Social proof gate** — defer if the agent's tier requires more
     adopted neighbors than currently exists.

If both gates pass, the filter returns ``None``, meaning "let the LLM
decide" (System 2).

Swapping the filter
-------------------
Inject a different ``DecisionFilter`` via ``SimulationConfig.decision_filter``.
Pass ``NullFilter`` to disable System 1 entirely (every decision hits the LLM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from llm.schemas import (
    PerceivedAttributes,
    PurchaseDecision,
    PurchaseDecisionOutput,
)

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent
    from environment.product import Product


# ─────────────────────────────────────────────────────────────
# FILTER RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """Return value from a DecisionFilter.

    Either *decision* is None (fall through to System 2 / LLM), or it
    contains a fully-formed ``PurchaseDecisionOutput`` that can be used
    as-is without any LLM call.
    """
    decision: Optional[PurchaseDecisionOutput]
    reason: str  # human-readable; "budget_gate", "social_proof_gate", "pass_to_llm"


# ─────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────

class DecisionFilter(ABC):
    """Abstract base class for pre-LLM decision filters."""

    @abstractmethod
    def check(
        self,
        agent: "ConsumerAgent",
        product: "Product",
        adopted_ids: set,
        current_step: int,
    ) -> FilterResult:
        """Inspect the agent's state and decide whether to short-circuit.

        Returns ``FilterResult(decision=None, ...)`` to defer to the LLM,
        or a ``FilterResult`` with a concrete ``PurchaseDecisionOutput``.
        """

    @abstractmethod
    def describe(self) -> str:
        """One-line description for reports/logs."""


# ─────────────────────────────────────────────────────────────
# NULL FILTER (disables System 1)
# ─────────────────────────────────────────────────────────────

class NullFilter(DecisionFilter):
    """Never short-circuits — every decision goes to the LLM.

    Use this as a baseline to measure System 1's impact, or to debug
    LLM behaviour without interference.
    """

    def check(self, agent, product, adopted_ids, current_step):  # noqa: D401
        return FilterResult(decision=None, reason="pass_to_llm")

    def describe(self) -> str:
        return "NullFilter (System 1 disabled)"


# ─────────────────────────────────────────────────────────────
# SYSTEM 1 FILTER
# ─────────────────────────────────────────────────────────────

# Minimum fraction of adopted neighbors required before each tier will
# even consider evaluating.  Innovators/early_adopters have no threshold
# — they lead, they don't follow.
DEFAULT_TIER_SOCIAL_THRESHOLDS = {
    "innovator":      0.0,
    "early_adopter":  0.0,
    "early_majority": 0.15,
    "late_majority":  0.40,
    "laggard":        0.65,
}


class System1Filter(DecisionFilter):
    """Default fast pre-filter — budget + social proof.

    Parameters
    ----------
    budget_headroom_factor : float
        A product is rejected on budget grounds when its price exceeds
        ``agent.budget * budget_headroom_factor``.  Default 0.5 (50%)
        reflects that nobody spends half their disposable income on a
        single durable good on impulse.
    tier_social_thresholds : Dict[str, float]
        Per-tier minimum adopted-neighbor ratio.  Below this, the filter
        defers the decision; above it, the LLM evaluates normally.
    skip_social_check_at_launch : bool
        If True (default), the social proof gate is skipped at step 1
        (product launch) — nobody has adopted anything yet, so enforcing
        a threshold would defer everyone permanently.
    """

    def __init__(
        self,
        budget_headroom_factor: float = 0.5,
        tier_social_thresholds: Optional[dict] = None,
        skip_social_check_at_launch: bool = True,
    ) -> None:
        self.budget_headroom_factor = budget_headroom_factor
        self.tier_social_thresholds = (
            tier_social_thresholds or dict(DEFAULT_TIER_SOCIAL_THRESHOLDS)
        )
        self.skip_social_check_at_launch = skip_social_check_at_launch

    def check(
        self,
        agent: "ConsumerAgent",
        product: "Product",
        adopted_ids: set,
        current_step: int,
    ) -> FilterResult:
        # ── Gate 1: budget ──────────────────────────────────
        budget_cap = agent.state.resources.budget * self.budget_headroom_factor
        if product.price > budget_cap:
            return FilterResult(
                decision=self._build_reject(
                    reasoning=(
                        f"System 1 budget gate: price ¥{product.price:,.0f} exceeds "
                        f"{self.budget_headroom_factor:.0%} of available budget "
                        f"¥{agent.state.resources.budget:,.0f} "
                        f"(cap ¥{budget_cap:,.0f})."
                    ),
                    price_acceptable=False,
                    key_concerns=["price exceeds budget headroom"],
                ),
                reason="budget_gate",
            )

        # ── Gate 2: social proof ────────────────────────────
        if current_step > 1 or not self.skip_social_check_at_launch:
            tier = agent.state.profile.adopter_tier
            threshold = self.tier_social_thresholds.get(tier, 0.0)

            if threshold > 0.0:
                neighbors = agent.state.relationships.get_neighbors()
                if neighbors:
                    adopted_ratio = (
                        sum(1 for n in neighbors if n in adopted_ids) / len(neighbors)
                    )
                else:
                    adopted_ratio = 0.0

                if adopted_ratio < threshold:
                    return FilterResult(
                        decision=self._build_defer(
                            current_step=current_step,
                            reasoning=(
                                f"System 1 social-proof gate: tier '{tier}' requires "
                                f"{threshold:.0%} neighbor adoption, currently "
                                f"{adopted_ratio:.0%}."
                            ),
                        ),
                        reason="social_proof_gate",
                    )

        # Pass to System 2 (LLM)
        return FilterResult(decision=None, reason="pass_to_llm")

    def describe(self) -> str:
        return (
            f"System1Filter(budget_headroom={self.budget_headroom_factor:.0%}, "
            f"social_thresholds={self.tier_social_thresholds})"
        )

    # ── synthetic decision helpers ──────────────────────────

    @staticmethod
    def _neutral_attrs() -> PerceivedAttributes:
        """Neutral Rogers attributes used when System 1 bypasses the LLM."""
        return PerceivedAttributes(
            relative_advantage=0.5,
            compatibility=0.5,
            complexity=0.5,
            trialability=0.5,
            observability=0.5,
        )

    def _build_reject(
        self,
        reasoning: str,
        price_acceptable: bool,
        key_concerns: list,
    ) -> PurchaseDecisionOutput:
        return PurchaseDecisionOutput(
            decision=PurchaseDecision.REJECT,
            confidence=0.90,
            reasoning=reasoning,
            perceived_attributes=self._neutral_attrs(),
            price_acceptable=price_acceptable,
            key_concerns=key_concerns,
            social_influence_weight=0.0,
            deferred_until=None,
        )

    def _build_defer(
        self, current_step: int, reasoning: str
    ) -> PurchaseDecisionOutput:
        return PurchaseDecisionOutput(
            decision=PurchaseDecision.DEFER,
            confidence=0.75,
            reasoning=reasoning,
            perceived_attributes=self._neutral_attrs(),
            price_acceptable=True,
            key_concerns=[],
            social_influence_weight=0.0,
            deferred_until=current_step + 2,  # re-check two steps later
        )
