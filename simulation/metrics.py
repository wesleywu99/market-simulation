"""
Simulation metrics collection and reporting.

``MetricsCollector`` records per-step snapshots and provides query helpers
for adoption curves, tier breakdowns, and formatted summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# PER-STEP SNAPSHOT
# ─────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    step: int
    total_agents: int

    # decision counts (this step only)
    evaluators: int
    buyers: int
    deferrers: int
    rejecters: int

    # cumulative
    cumulative_buyers: int
    adoption_rate: float            # cumulative_buyers / total_agents

    # WOM
    wom_messages_generated: int
    wom_receptions: int             # successful receptions reaching a target

    # per-tier adoption (cumulative)
    tier_adoption: Dict[str, Dict] = field(default_factory=dict)
    # e.g. {"innovator": {"total": 1, "adopted": 1, "rate": 1.0}, ...}


# ─────────────────────────────────────────────────────────────
# COLLECTOR
# ─────────────────────────────────────────────────────────────

TIERS = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]


class MetricsCollector:
    """Accumulates ``StepMetrics`` across the simulation and provides summaries."""

    def __init__(self) -> None:
        self.steps: List[StepMetrics] = []

    # ── recording ─────────────────────────────────────────────

    def record_step(
        self,
        *,
        step: int,
        total_agents: int,
        evaluators: int,
        buyers: int,
        deferrers: int,
        rejecters: int,
        cumulative_buyers: int,
        wom_generated: int,
        wom_received: int,
        agents: List[ConsumerAgent],
        adopted_ids: set,
    ) -> StepMetrics:
        tier_adoption: Dict[str, Dict] = {}
        for tier in TIERS:
            tier_agents = [a for a in agents if a.state.profile.adopter_tier == tier]
            adopted = sum(1 for a in tier_agents if a.agent_id in adopted_ids)
            total = len(tier_agents)
            tier_adoption[tier] = {
                "total": total,
                "adopted": adopted,
                "rate": adopted / total if total else 0.0,
            }

        sm = StepMetrics(
            step=step,
            total_agents=total_agents,
            evaluators=evaluators,
            buyers=buyers,
            deferrers=deferrers,
            rejecters=rejecters,
            cumulative_buyers=cumulative_buyers,
            adoption_rate=cumulative_buyers / total_agents if total_agents else 0.0,
            wom_messages_generated=wom_generated,
            wom_receptions=wom_received,
            tier_adoption=tier_adoption,
        )
        self.steps.append(sm)
        return sm

    # ── queries ───────────────────────────────────────────────

    def adoption_curve(self) -> List[float]:
        """Return adoption rate at each step (for plotting)."""
        return [s.adoption_rate for s in self.steps]

    def final_adoption(self) -> float:
        return self.steps[-1].adoption_rate if self.steps else 0.0

    def cumulative_sales(self) -> int:
        return self.steps[-1].cumulative_buyers if self.steps else 0

    def total_wom_receptions(self) -> int:
        return sum(s.wom_receptions for s in self.steps)

    # ── formatted output ──────────────────────────────────────

    def print_adoption_curve(self) -> None:
        """Print a text-based adoption curve bar chart."""
        if not self.steps:
            return
        bar_width = 30
        print()
        print("  Adoption Curve")
        print("  " + "-" * 50)
        for sm in self.steps:
            filled = int(sm.adoption_rate * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)
            print(
                f"  Step {sm.step:>2}: [{bar}]  "
                f"{sm.adoption_rate:>5.0%}  ({sm.cumulative_buyers}/{sm.total_agents})"
            )

    def print_tier_breakdown(self) -> None:
        """Print final per-tier adoption table."""
        if not self.steps:
            return
        final = self.steps[-1].tier_adoption
        print()
        print(f"  {'Tier':<16} {'Total':>5}  {'Adopted':>7}  {'Rate':>6}")
        print(f"  {'-'*16} {'-----':>5}  {'-------':>7}  {'------':>6}")
        for tier in TIERS:
            info = final.get(tier, {})
            total = info.get("total", 0)
            adopted = info.get("adopted", 0)
            rate = info.get("rate", 0.0)
            if total == 0:
                continue
            print(f"  {tier:<16} {total:>5}  {adopted:>7}  {rate:>6.0%}")

    def print_wom_summary(self) -> None:
        """Print cumulative WOM statistics."""
        total_gen = sum(s.wom_messages_generated for s in self.steps)
        total_rec = sum(s.wom_receptions for s in self.steps)
        print()
        print(f"  WOM Summary: {total_gen} messages generated, "
              f"{total_rec} receptions across {len(self.steps)} steps")

    # ── serialisation ─────────────────────────────────────────

    def to_dicts(self) -> List[dict]:
        """Export all step metrics as a list of plain dicts (JSON-friendly)."""
        out = []
        for sm in self.steps:
            out.append({
                "step": sm.step,
                "total_agents": sm.total_agents,
                "evaluators": sm.evaluators,
                "buyers": sm.buyers,
                "deferrers": sm.deferrers,
                "rejecters": sm.rejecters,
                "cumulative_buyers": sm.cumulative_buyers,
                "adoption_rate": round(sm.adoption_rate, 4),
                "wom_messages_generated": sm.wom_messages_generated,
                "wom_receptions": sm.wom_receptions,
                "tier_adoption": sm.tier_adoption,
            })
        return out
