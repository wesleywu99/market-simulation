"""
Post-purchase experience sampling — the "blind-box" model.

Replaces the Phase 1 single-scalar quality rating with a per-buyer,
per-dimension stochastic experience that surfaces concrete textual
defects and praises.  These ground the WOM prompt in specifics rather
than abstract numerical scores.

See ``docs/30_product_quality.md`` for the full design rationale.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent
    from environment.product import Product


# ─────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────

@dataclass
class ExperienceProfile:
    """A single buyer's grounded experience with the product.

    Attributes
    ----------
    overall_score : float
        0–10 aggregate, weighted by the buyer's dimension preferences.
    dimension_scores : Dict[str, float]
        Per-dimension observed quality after blind-box noise.
    surfaced_defects : List[str]
        Concrete defect descriptions (one per low-bucket dimension).
    surfaced_praises : List[str]
        Concrete praise descriptions (one per high-bucket dimension).
    """
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    surfaced_defects: List[str] = field(default_factory=list)
    surfaced_praises: List[str] = field(default_factory=list)

    @property
    def overall_experience(self) -> str:
        if self.overall_score >= 7.0:
            return "positive"
        if self.overall_score <= 4.0:
            return "negative"
        return "mixed"


# ─────────────────────────────────────────────────────────────
# STRATEGY ABC
# ─────────────────────────────────────────────────────────────

class ExperienceSampler(ABC):
    """Abstract base for post-purchase experience sampling."""

    @abstractmethod
    def sample(
        self,
        product: "Product",
        agent: "ConsumerAgent",
        rng: random.Random,
    ) -> ExperienceProfile:
        """Produce a grounded ExperienceProfile for this buyer."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for reports."""


# ─────────────────────────────────────────────────────────────
# DEFAULT IMPL — BLIND BOX
# ─────────────────────────────────────────────────────────────

class BlindBoxExperience(ExperienceSampler):
    """Per-dimension Gaussian noise + threshold-based defect/praise surfacing.

    Parameters
    ----------
    noise_sigma : float
        Stddev of the per-dimension noise applied to latent quality.
        Higher = more variance between buyers (more "blind box" feel).
    low_threshold : float
        Observed score below this triggers a defect surfacing.
    high_threshold : float
        Observed score above this triggers a praise surfacing.
    expectation_shift_by_income : Dict[str, float]
        Income-driven expectation bias.  High-income buyers are stricter
        (negative shift), low-income buyers are pleasantly surprised.
    """

    def __init__(
        self,
        noise_sigma: float = 0.15,
        low_threshold: float = 0.4,
        high_threshold: float = 0.7,
        expectation_shift_by_income: Optional[Dict[str, float]] = None,
    ) -> None:
        self.noise_sigma = noise_sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.expectation_shift_by_income = expectation_shift_by_income or {
            "high":     -0.05,   # stricter
            "affluent": -0.05,
            "medium":    0.00,
            "low":       0.03,   # pleasantly surprised
        }

    def sample(
        self,
        product: "Product",
        agent: "ConsumerAgent",
        rng: random.Random,
    ) -> ExperienceProfile:
        # Fall back to legacy scalar quality when no dimensions are configured.
        if not product.quality_dimensions:
            return self._sample_scalar_fallback(product, agent, rng)

        weights = self._resolve_weights(product, agent)
        shift = self.expectation_shift_by_income.get(
            agent.state.profile.income_level, 0.0
        )

        dim_scores: Dict[str, float] = {}
        defects: List[str] = []
        praises: List[str] = []

        for dim, latent in product.quality_dimensions.items():
            observed = max(0.0, min(1.0, latent + shift + rng.gauss(0, self.noise_sigma)))
            dim_scores[dim] = observed

            if observed < self.low_threshold:
                snippet = self._pick(product.defect_bank, dim, "low", rng)
                if snippet:
                    defects.append(snippet)
            elif observed > self.high_threshold:
                snippet = self._pick(product.praise_bank, dim, "high", rng)
                if snippet:
                    praises.append(snippet)
            # mid-bucket: no surfaced text (intentionally quiet)

        # Persona-weighted aggregate, scaled to 0-10
        if weights:
            total_w = sum(weights.get(d, 0.0) for d in dim_scores) or 1.0
            weighted = sum(dim_scores[d] * weights.get(d, 0.0) for d in dim_scores) / total_w
        else:
            weighted = sum(dim_scores.values()) / len(dim_scores)

        return ExperienceProfile(
            overall_score=round(weighted * 10, 2),
            dimension_scores=dim_scores,
            surfaced_defects=defects,
            surfaced_praises=praises,
        )

    def describe(self) -> str:
        return (
            f"BlindBoxExperience(sigma={self.noise_sigma}, "
            f"low<{self.low_threshold}, high>{self.high_threshold})"
        )

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _pick(bank: dict, dim: str, severity: str, rng: random.Random) -> Optional[str]:
        snippets = bank.get((dim, severity), [])
        return rng.choice(snippets) if snippets else None

    def _resolve_weights(
        self, product: "Product", agent: "ConsumerAgent"
    ) -> Dict[str, float]:
        """Per-buyer weights — agent override or uniform fallback."""
        agent_weights = agent.state.profile.dimension_weights
        if agent_weights:
            # Restrict to dimensions actually present on the product
            return {d: w for d, w in agent_weights.items() if d in product.quality_dimensions}
        return {}

    def _sample_scalar_fallback(
        self,
        product: "Product",
        agent: "ConsumerAgent",
        rng: random.Random,
    ) -> ExperienceProfile:
        """Backward-compat path for products with no quality_dimensions."""
        base = product.quality * 10
        shift = self.expectation_shift_by_income.get(
            agent.state.profile.income_level, 0.0
        ) * 10
        noise = rng.gauss(0, self.noise_sigma * 10)
        score = max(0.0, min(10.0, base + shift + noise))
        return ExperienceProfile(overall_score=round(score, 2))
