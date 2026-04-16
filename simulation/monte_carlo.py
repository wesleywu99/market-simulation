"""
MonteCarloRunner — repeated simulations with seed sweep + CI bands.

A single simulation run produces one adoption curve.  Stochastic gates
(network rewiring, Independent-Cascade reception, LLM temperature)
introduce variance, so a single curve is misleading on its own.  The
Monte Carlo wrapper sweeps a list of seeds, runs the same configuration
N times, and aggregates the per-step adoption rates into a mean curve
plus a 95% confidence interval band.

Outputs (under ``<out_dir>/<run_id>/``):

* ``adoption.csv`` — one row per step with ``mean``, ``ci_lower``,
  ``ci_upper``, and one column per individual run.
* ``summary.json`` — top-level aggregates (final-step mean, CI, per-run
  finals, configuration label).
* ``adoption.png`` — matplotlib figure with mean line + shaded CI band
  (only written if matplotlib is importable).

Design choices
--------------
* **Sequential, not concurrent.**  Each ``SimulationRunner.run_async()``
  already parallelises within-step LLM calls.  Running multiple
  simulations concurrently would multiply the LLM-call concurrency and
  make console logs unreadable; it does not buy meaningful wall-clock
  improvement once the dispatcher's semaphore is saturated.
* **Per-run config cloning.**  Strategy objects (network builder, WOM
  engine, …) are deep-copied so a per-run ``seed`` override does not
  mutate the user's template.
* **Tracing disabled by default for child runs.**  Per-run trace dirs
  multiply file-system noise; the MC report is the artefact of interest.
  Set ``per_run_trace=True`` to opt back in.
* **Matplotlib is optional.**  If the import fails, the CSV/JSON still
  write and a warning prints — the rest of the project does not depend
  on matplotlib.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import csv
import io
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from simulation.runner import SimulationConfig, SimulationReport, SimulationRunner


# ─────────────────────────────────────────────────────────────
# CI HELPERS
# ─────────────────────────────────────────────────────────────

# Two-sided 95% critical values of Student's t for df = n - 1.
# Avoids a scipy dependency for a tiny lookup.  Falls back to 1.96
# (normal approx) for larger n.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}


def _t_critical(n: int) -> float:
    """Return the two-sided 95% t critical value for df = n - 1."""
    df = max(1, n - 1)
    if df in _T_95:
        return _T_95[df]
    if df < 30:
        # Linear interp between nearest tabulated dfs (cheap, good enough)
        keys = sorted(_T_95)
        lower = max(k for k in keys if k <= df)
        upper = min(k for k in keys if k >= df)
        if lower == upper:
            return _T_95[lower]
        frac = (df - lower) / (upper - lower)
        return _T_95[lower] + frac * (_T_95[upper] - _T_95[lower])
    return 1.96  # normal approximation for large n


def _ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for a sample.

    With fewer than two samples the CI collapses to (mean, mean, mean).
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if n < 2:
        return mean, mean, mean
    sd = statistics.stdev(values)
    margin = _t_critical(n) * sd / math.sqrt(n)
    return mean, mean - margin, mean + margin


# ─────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────

@dataclass
class MonteCarloReport:
    """Aggregated Monte Carlo result.

    All curve-shaped fields are aligned: index ``i`` corresponds to step
    ``i + 1`` (steps are 1-indexed in the runner).
    """
    run_id: str
    n_runs: int
    seeds: List[int]
    n_steps: int
    n_agents: int
    product_name: str
    config_label: str
    adoption_curves: List[List[float]]      # one inner list per run
    mean_curve: List[float]
    ci_lower_curve: List[float]
    ci_upper_curve: List[float]
    final_adoption_rates: List[float]
    final_mean: float
    final_ci_lower: float
    final_ci_upper: float
    csv_path: Optional[str] = None
    summary_json_path: Optional[str] = None
    figure_path: Optional[str] = None

    # ── handy derived metrics ──
    @property
    def final_ci_width(self) -> float:
        return self.final_ci_upper - self.final_ci_lower


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

class MonteCarloRunner:
    """Repeat a simulation across a seed sweep and aggregate results.

    Parameters
    ----------
    config_template : SimulationConfig
        Template config — copied per run with the seed overridden.
        The template's ``trace_dir`` is ignored for child runs unless
        ``per_run_trace=True``.
    seeds : Sequence[int]
        Seeds to sweep.  Each seed produces one full simulation.
    out_dir : str | Path
        Parent directory for run subdirs (default ``"monte_carlo"``).
    run_id : str | None
        Output subdir name; auto-generated if omitted.
    quiet : bool
        Suppress per-step console output from child runs.  The MC
        wrapper still prints a one-line progress summary per run.
    per_run_trace : bool
        If True, propagate ``config_template.trace_dir`` to each child
        run.  Off by default to keep the file system tidy.
    """

    def __init__(
        self,
        config_template: SimulationConfig,
        seeds: Sequence[int],
        out_dir: str | Path = "monte_carlo",
        run_id: Optional[str] = None,
        quiet: bool = True,
        per_run_trace: bool = False,
    ) -> None:
        if not seeds:
            raise ValueError("MonteCarloRunner requires at least one seed.")

        self.config_template = config_template
        self.seeds = list(seeds)
        self.quiet = quiet
        self.per_run_trace = per_run_trace

        self.run_id = run_id or datetime.now().strftime("mc_%Y%m%d_%H%M%S")
        self.run_dir = Path(out_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────

    def run(self) -> MonteCarloReport:
        """Synchronous wrapper around :py:meth:`run_async`."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> MonteCarloReport:
        adoption_curves: List[List[float]] = []
        finals: List[float] = []
        config_labels: List[str] = []

        self._print_header()

        for i, seed in enumerate(self.seeds, 1):
            print(f"\n[MC {i}/{len(self.seeds)}] seed={seed} -> running...", flush=True)
            cfg = self._clone_config(seed)
            runner = SimulationRunner(cfg)

            if self.quiet:
                # Swallow per-step console output but keep our own status line.
                with contextlib.redirect_stdout(io.StringIO()):
                    report = await runner.run_async()
            else:
                report = await runner.run_async()

            adoption_curves.append(report.adoption_curve)
            finals.append(report.final_adoption_rate)
            if not config_labels:
                config_labels.append(self._config_label(report))

            print(
                f"[MC {i}/{len(self.seeds)}] seed={seed} done "
                f"-> final adoption {report.final_adoption_rate:.0%} "
                f"({report.total_sales}/{report.n_agents})",
                flush=True,
            )

        # Aggregate per-step
        n_steps = max((len(c) for c in adoption_curves), default=0)
        mean_curve: List[float] = []
        ci_lo: List[float] = []
        ci_hi: List[float] = []
        for s in range(n_steps):
            values = [c[s] for c in adoption_curves if s < len(c)]
            mean, lo, hi = _ci95(values)
            mean_curve.append(mean)
            ci_lo.append(lo)
            ci_hi.append(hi)

        final_mean, final_lo, final_hi = _ci95(finals)

        report = MonteCarloReport(
            run_id=self.run_id,
            n_runs=len(self.seeds),
            seeds=list(self.seeds),
            n_steps=n_steps,
            n_agents=self.config_template.n_agents,
            product_name=self.config_template.product.name,
            config_label=config_labels[0] if config_labels else "(unknown)",
            adoption_curves=adoption_curves,
            mean_curve=mean_curve,
            ci_lower_curve=ci_lo,
            ci_upper_curve=ci_hi,
            final_adoption_rates=finals,
            final_mean=final_mean,
            final_ci_lower=final_lo,
            final_ci_upper=final_hi,
        )

        report.csv_path = str(self._write_csv(report))
        report.summary_json_path = str(self._write_summary(report))
        report.figure_path = self._write_figure(report)

        self._print_summary(report)
        return report

    # ── per-run config plumbing ─────────────────────────────

    def _clone_config(self, seed: int) -> SimulationConfig:
        """Deep-copy the template and re-seed every stochastic component."""
        product = copy.deepcopy(self.config_template.product)

        network_builder = self._reseed(self.config_template.network_builder, seed)
        wom_engine = self._reseed(self.config_template.wom_engine, seed)
        # Filters / experience samplers / seeders / belief updaters are
        # typically stateless or carry no RNG of their own — deep-copy is
        # cheap insurance and avoids accidental cross-run mutation.
        decision_filter = copy.deepcopy(self.config_template.decision_filter)
        experience_sampler = copy.deepcopy(self.config_template.experience_sampler)
        influencer_seeding = copy.deepcopy(self.config_template.influencer_seeding)
        belief_updater = copy.deepcopy(self.config_template.belief_updater)

        return SimulationConfig(
            product=product,
            n_agents=self.config_template.n_agents,
            n_steps=self.config_template.n_steps,
            seed=seed,
            network_builder=network_builder,
            wom_engine=wom_engine,
            decision_filter=decision_filter,
            experience_sampler=experience_sampler,
            influencer_seeding=influencer_seeding,
            belief_updater=belief_updater,
            step_duration_days=self.config_template.step_duration_days,
            budget_refresh_interval=self.config_template.budget_refresh_interval,
            sentiment_decay=self.config_template.sentiment_decay,
            intent_decay=self.config_template.intent_decay,
            event_schedule=copy.deepcopy(self.config_template.event_schedule),
            market_context=self.config_template.market_context,
            population_spec=self.config_template.population_spec,
            trace_dir=self.config_template.trace_dir if self.per_run_trace else None,
            run_id=None,
        )

    @staticmethod
    def _reseed(strategy, seed: int):
        """Deep-copy a strategy and overwrite its ``seed`` attribute if present."""
        if strategy is None:
            return None
        clone = copy.deepcopy(strategy)
        if hasattr(clone, "seed"):
            clone.seed = seed
        return clone

    # ── output writers ──────────────────────────────────────

    def _write_csv(self, report: MonteCarloReport) -> Path:
        path = self.run_dir / "adoption.csv"
        run_cols = [f"run_{i}_seed{seed}" for i, seed in enumerate(report.seeds)]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "mean", "ci_lower", "ci_upper", *run_cols])
            for s in range(report.n_steps):
                row = [
                    s + 1,
                    round(report.mean_curve[s], 4),
                    round(report.ci_lower_curve[s], 4),
                    round(report.ci_upper_curve[s], 4),
                ]
                for curve in report.adoption_curves:
                    row.append(round(curve[s], 4) if s < len(curve) else "")
                writer.writerow(row)
        return path

    def _write_summary(self, report: MonteCarloReport) -> Path:
        path = self.run_dir / "summary.json"
        payload = {
            "run_id": report.run_id,
            "n_runs": report.n_runs,
            "n_steps": report.n_steps,
            "n_agents": report.n_agents,
            "seeds": report.seeds,
            "product_name": report.product_name,
            "config_label": report.config_label,
            "final_mean": round(report.final_mean, 4),
            "final_ci_lower": round(report.final_ci_lower, 4),
            "final_ci_upper": round(report.final_ci_upper, 4),
            "final_ci_width": round(report.final_ci_width, 4),
            "final_adoption_rates": [round(v, 4) for v in report.final_adoption_rates],
            "mean_curve": [round(v, 4) for v in report.mean_curve],
            "ci_lower_curve": [round(v, 4) for v in report.ci_lower_curve],
            "ci_upper_curve": [round(v, 4) for v in report.ci_upper_curve],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def _write_figure(self, report: MonteCarloReport) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless-safe
            import matplotlib.pyplot as plt
        except ImportError:
            print("    [warn] matplotlib not installed -- skipping adoption.png")
            return None

        steps = list(range(1, report.n_steps + 1))
        fig, ax = plt.subplots(figsize=(8, 5))

        # Faint individual runs for transparency
        for curve in report.adoption_curves:
            ax.plot(steps, curve, color="#888", alpha=0.25, linewidth=0.8)

        # CI band + mean
        ax.fill_between(
            steps, report.ci_lower_curve, report.ci_upper_curve,
            color="#1f77b4", alpha=0.20, label="95% CI",
        )
        ax.plot(steps, report.mean_curve, color="#1f77b4", linewidth=2.0, label="Mean")

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative adoption rate")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"{report.product_name} — Monte Carlo ({report.n_runs} runs, "
            f"{report.n_agents} agents)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        path = self.run_dir / "adoption.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    # ── console output ──────────────────────────────────────

    def _print_header(self) -> None:
        print()
        print("=" * 64)
        print(f"  MONTE CARLO - {self.config_template.product.name}")
        print(f"  {len(self.seeds)} runs x {self.config_template.n_steps} steps x "
              f"{self.config_template.n_agents} agents")
        print(f"  Seeds: {list(self.seeds)}")
        print(f"  Output dir: {self.run_dir}")
        print("=" * 64)

    def _print_summary(self, report: MonteCarloReport) -> None:
        print()
        print("=" * 64)
        print(f"  MONTE CARLO REPORT - {report.product_name}")
        print("=" * 64)
        print(f"  Runs          : {report.n_runs}")
        print(f"  Config        : {report.config_label}")
        print(
            f"  Final adoption: mean {report.final_mean:.1%}  "
            f"95% CI [{report.final_ci_lower:.1%}, {report.final_ci_upper:.1%}]  "
            f"(width {report.final_ci_width:.1%})"
        )
        print(f"  Per-run finals: "
              f"{[f'{v:.0%}' for v in report.final_adoption_rates]}")
        print()
        print("  Mean adoption curve (with 95% CI)")
        print("  " + "-" * 50)
        bar_width = 30
        for s, mean in enumerate(report.mean_curve, 1):
            lo = report.ci_lower_curve[s - 1]
            hi = report.ci_upper_curve[s - 1]
            filled = int(max(0.0, min(1.0, mean)) * bar_width)
            bar = "#" * filled + "." * (bar_width - filled)
            print(
                f"  Step {s:>2}: [{bar}]  "
                f"{mean:>5.0%}  CI[{lo:>5.0%}, {hi:>5.0%}]"
            )
        print()
        print(f"  CSV     : {report.csv_path}")
        print(f"  Summary : {report.summary_json_path}")
        if report.figure_path:
            print(f"  Figure  : {report.figure_path}")
        print("=" * 64)

    @staticmethod
    def _config_label(report: SimulationReport) -> str:
        return (
            f"net={report.network_type} | wom={report.wom_type} | "
            f"filter={report.filter_type} | seeding={report.seeding_type} | "
            f"belief={report.belief_updater_type}"
        )
