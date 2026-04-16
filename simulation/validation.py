"""
Output validation suite -- automated sanity checks on simulation results.

Implements the success criteria from the seed-case design docs as
executable checks.  Run after a simulation or sweep to flag anomalies
before relying on the numbers for real decisions.

Checks
------
* ``check_rogers_ordering``   -- innovators adopt before laggards.
* ``check_adoption_monotonic``-- cumulative adoption never decreases.
* ``check_scurve_shape``      -- adoption curve is concave-then-convex (S).
* ``check_tier_coverage``     -- every tier has at least one agent.
* ``check_price_monotonicity``-- higher price -> lower adoption (sweep only).
* ``check_ci_width``          -- MC confidence interval below threshold.
* ``run_all``                 -- run every applicable check and print a report.

Each check returns a ``ValidationResult``.  A failing check does NOT mean
the simulation is broken -- it means the output deviates from theoretical
expectations and warrants human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from simulation.monte_carlo import MonteCarloReport
    from simulation.runner import SimulationReport
    from simulation.sweep import SweepReport


# ─────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


@dataclass
class ValidationReport:
    """Collection of validation results."""
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def print_report(self) -> None:
        print()
        print("=" * 64)
        print("  VALIDATION REPORT")
        print("=" * 64)
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")
            if r.details:
                for line in r.details.strip().split("\n"):
                    print(f"         {line}")
        print("-" * 64)
        print(f"  {self.n_passed} passed, {self.n_failed} failed")
        print("=" * 64)


# ─────────────────────────────────────────────────────────────
# SINGLE-RUN CHECKS
# ─────────────────────────────────────────────────────────────

TIER_ORDER = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]


def check_rogers_ordering(report: "SimulationReport") -> ValidationResult:
    """Verify that earlier Rogers tiers adopt at higher rates than later ones.

    This is a soft check: we require that the adoption rate is
    non-increasing across tiers (ties are OK, since small samples
    can produce equal rates).  A strict monotonic decrease is too
    strong for small N.
    """
    tier_data = report.tier_adoption
    rates = []
    detail_lines = []
    for tier in TIER_ORDER:
        info = tier_data.get(tier, {})
        rate = info.get("rate", 0.0)
        total = info.get("total", 0)
        adopted = info.get("adopted", 0)
        rates.append(rate)
        detail_lines.append(f"{tier}: {adopted}/{total} = {rate:.0%}")

    # Check non-increasing (allow ties)
    violations = []
    for i in range(len(rates) - 1):
        if rates[i] < rates[i + 1]:
            violations.append(
                f"{TIER_ORDER[i]} ({rates[i]:.0%}) < {TIER_ORDER[i+1]} ({rates[i+1]:.0%})"
            )

    if not violations:
        return ValidationResult(
            name="Rogers ordering",
            passed=True,
            message="Tier adoption rates are non-increasing (innovator >= ... >= laggard)",
            details="\n".join(detail_lines),
        )
    return ValidationResult(
        name="Rogers ordering",
        passed=False,
        message=f"{len(violations)} tier ordering violation(s)",
        details="\n".join(detail_lines + ["Violations:"] + violations),
    )


def check_adoption_monotonic(report: "SimulationReport") -> ValidationResult:
    """Verify cumulative adoption never decreases step-over-step."""
    curve = report.adoption_curve
    decreases = []
    for i in range(1, len(curve)):
        if curve[i] < curve[i - 1] - 1e-9:  # tolerance for float rounding
            decreases.append(f"step {i}: {curve[i-1]:.4f} -> {curve[i]:.4f}")

    if not decreases:
        return ValidationResult(
            name="Adoption monotonic",
            passed=True,
            message="Cumulative adoption is non-decreasing across all steps",
        )
    return ValidationResult(
        name="Adoption monotonic",
        passed=False,
        message=f"Adoption decreased in {len(decreases)} step(s)",
        details="\n".join(decreases),
    )


def check_scurve_shape(
    report: "SimulationReport",
    min_steps: int = 8,
) -> ValidationResult:
    """Check if the adoption curve has an S-curve shape.

    An S-curve has a period of accelerating growth (concave up) followed
    by decelerating growth (concave down).  We approximate this by
    checking that the second derivative changes sign at least once.

    Requires enough steps to be meaningful (default: 8+).
    """
    curve = report.adoption_curve
    if len(curve) < min_steps:
        return ValidationResult(
            name="S-curve shape",
            passed=True,  # skip, not fail
            message=f"Skipped: only {len(curve)} steps (need {min_steps}+)",
        )

    # First differences (velocity of adoption)
    deltas = [curve[i] - curve[i - 1] for i in range(1, len(curve))]
    if not any(d > 0 for d in deltas):
        return ValidationResult(
            name="S-curve shape",
            passed=False,
            message="No adoption growth detected (flat curve)",
        )

    # Second differences (acceleration)
    accels = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]

    # Look for sign change in acceleration (+ to - = inflection point)
    has_positive = any(a > 1e-9 for a in accels)
    has_negative = any(a < -1e-9 for a in accels)

    if has_positive and has_negative:
        return ValidationResult(
            name="S-curve shape",
            passed=True,
            message="Inflection point detected (acceleration changes sign)",
        )
    if has_positive and not has_negative:
        return ValidationResult(
            name="S-curve shape",
            passed=False,
            message="Pure acceleration (exponential, no saturation) -- may need more steps",
        )
    return ValidationResult(
        name="S-curve shape",
        passed=False,
        message="Pure deceleration or flat -- adoption started fast and stalled",
    )


def check_tier_coverage(report: "SimulationReport") -> ValidationResult:
    """Verify every Rogers tier has at least one agent."""
    tier_data = report.tier_adoption
    missing = [t for t in TIER_ORDER if tier_data.get(t, {}).get("total", 0) == 0]

    if not missing:
        return ValidationResult(
            name="Tier coverage",
            passed=True,
            message="All 5 Rogers tiers represented in population",
        )
    return ValidationResult(
        name="Tier coverage",
        passed=False,
        message=f"Missing tier(s): {', '.join(missing)}",
    )


# ─────────────────────────────────────────────────────────────
# MONTE CARLO CHECKS
# ─────────────────────────────────────────────────────────────

def check_ci_width(
    mc_report: "MonteCarloReport",
    max_width: float = 0.15,
) -> ValidationResult:
    """Check that the 95% CI at the final step is below a threshold.

    Default threshold: 15 percentage points, per the Phase 1.5 plan.
    """
    width = mc_report.final_ci_width
    if width <= max_width:
        return ValidationResult(
            name="CI width",
            passed=True,
            message=f"Final 95% CI width {width:.1%} <= {max_width:.0%} threshold",
        )
    return ValidationResult(
        name="CI width",
        passed=False,
        message=f"Final 95% CI width {width:.1%} exceeds {max_width:.0%} threshold",
        details=f"CI: [{mc_report.final_ci_lower:.1%}, {mc_report.final_ci_upper:.1%}]",
    )


# ─────────────────────────────────────────────────────────────
# SWEEP CHECKS
# ─────────────────────────────────────────────────────────────

def check_price_monotonicity(sweep_report: "SweepReport") -> ValidationResult:
    """For a price sweep, verify higher price -> lower adoption.

    Only applies when the sweep has exactly one axis named ``product.price``.
    """
    price_axes = [a for a in sweep_report.axes if a.param == "product.price"]
    if not price_axes:
        return ValidationResult(
            name="Price monotonicity",
            passed=True,
            message="Skipped: no product.price axis in sweep",
        )

    # Sort grid results by price
    price_results = sorted(
        sweep_report.grid_results,
        key=lambda gp: gp.params.get("product.price", 0),
    )

    prices = [gp.params["product.price"] for gp in price_results]
    means = [gp.final_mean for gp in price_results]

    violations = []
    for i in range(1, len(means)):
        if means[i] > means[i - 1] + 0.01:  # small tolerance
            violations.append(
                f"price {prices[i-1]} ({means[i-1]:.1%}) < "
                f"price {prices[i]} ({means[i]:.1%})"
            )

    details = "\n".join(
        f"price={p}: {m:.1%}" for p, m in zip(prices, means)
    )

    if not violations:
        return ValidationResult(
            name="Price monotonicity",
            passed=True,
            message="Higher price -> lower (or equal) adoption across all points",
            details=details,
        )
    return ValidationResult(
        name="Price monotonicity",
        passed=False,
        message=f"{len(violations)} violation(s) of price-adoption monotonicity",
        details=details + "\nViolations:\n" + "\n".join(violations),
    )


# ─────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────

def validate_run(report: "SimulationReport") -> ValidationReport:
    """Run all single-run checks and return a ValidationReport."""
    vr = ValidationReport()
    vr.add(check_rogers_ordering(report))
    vr.add(check_adoption_monotonic(report))
    vr.add(check_scurve_shape(report))
    vr.add(check_tier_coverage(report))
    return vr


def validate_mc(mc_report: "MonteCarloReport", max_ci_width: float = 0.15) -> ValidationReport:
    """Run Monte Carlo-specific checks."""
    vr = ValidationReport()
    vr.add(check_ci_width(mc_report, max_width=max_ci_width))
    return vr


def validate_sweep(sweep_report: "SweepReport") -> ValidationReport:
    """Run sweep-specific checks."""
    vr = ValidationReport()
    vr.add(check_price_monotonicity(sweep_report))
    return vr
