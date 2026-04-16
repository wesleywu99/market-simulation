from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Product:
    """
    Product model.

    Combines objective attributes (set at launch) with perceived attributes
    (subjectively evaluated per-agent by the LLM) and dynamic market metrics
    (updated each simulation step).

    Quality is multi-dimensional (see ``quality_dimensions``).  The legacy
    scalar ``quality`` field is retained for backward compatibility but is
    derived as a uniform-weighted average of the dimensions when possible.
    See docs/30_product_quality.md for the full design rationale.
    """

    # ── Identity ──────────────────────────────────────────────
    product_id: str
    name: str
    category: str           # e.g. "consumer_drone" | "smartphone" | "ev_vehicle"
    brand: str

    # ── Objective attributes (set at launch, can change via events) ──
    price: float
    features: List[str]     # e.g. ["4K camera", "obstacle avoidance", "30min flight time"]
    quality: float          # 0.0–1.0; aggregate build quality (auto-derived if dimensions provided)
    brand_reputation: float # 0.0–1.0 in this market at launch
    distribution_channels: List[str]  # e.g. ["online", "flagship_store", "authorized_dealer"]

    # ── Multi-dimensional quality (semantic blind-box model) ──
    # Maps dimension name → latent quality in [0, 1].  Per-category vocabulary
    # (e.g. apparel: fit, material, color_accuracy; drone: camera, battery).
    quality_dimensions: Dict[str, float] = field(default_factory=dict)

    # Pre-authored review snippets keyed by (dimension, severity).
    # Severity values: "low" | "mid" | "high".  See docs/30_product_quality.md.
    defect_bank: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    praise_bank: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)

    # ── Rogers' 5 perceived attributes (market-level baseline) ──
    # These are starting estimates; each agent adjusts them individually via LLM
    perceived_relative_advantage: float = 0.5
    perceived_compatibility: float = 0.5
    perceived_complexity: float = 0.5       # higher = harder to use
    perceived_trialability: float = 0.5
    perceived_observability: float = 0.5

    # ── Dynamic market metrics (updated each step) ────────────
    cumulative_sales: int = 0
    current_step_sales: int = 0
    market_share: float = 0.0       # fraction of total addressable market
    nps_score: float = 0.0          # Net Promoter Score (-100 to 100)
    awareness_rate: float = 0.0     # fraction of agents aware of the product
    adoption_rate: float = 0.0      # fraction of agents who have purchased

    # ── Time metadata ─────────────────────────────────────────
    launch_step: int = 0
    maturity_level: float = 0.0     # 0.0 = just launched, 1.0 = fully mature

    def record_sale(self, price_paid: float) -> None:
        self.cumulative_sales += 1
        self.current_step_sales += 1

    def reset_step_counters(self) -> None:
        self.current_step_sales = 0

    def update_nps(self, ratings: List[float]) -> None:
        """ratings: list of 0–10 scores from post-purchase events."""
        if not ratings:
            return
        promoters = sum(1 for r in ratings if r >= 9)
        detractors = sum(1 for r in ratings if r <= 6)
        self.nps_score = (promoters - detractors) / len(ratings) * 100

    def __post_init__(self) -> None:
        """If quality_dimensions provided, derive scalar `quality` as the mean.

        Dimensions are the source of truth for the new model — the scalar
        ``quality`` is kept only for callers that read it directly (e.g.
        legacy reports).  Callers using dimensions can pass any placeholder
        for ``quality``; it will be overwritten here.
        """
        if self.quality_dimensions:
            self.quality = sum(self.quality_dimensions.values()) / len(self.quality_dimensions)

    def summary(self) -> str:
        return (
            f"{self.name} ({self.brand}) | "
            f"Price: {self.price} | "
            f"Sales: {self.cumulative_sales} | "
            f"Market share: {self.market_share:.1%} | "
            f"NPS: {self.nps_score:.0f}"
        )
