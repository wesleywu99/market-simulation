"""
ParameterSweep -- grid search over simulation parameters with MC at each point.

Monte Carlo sweeps seeds (stochastic variance) at a fixed configuration.
ParameterSweep sweeps *configuration parameters* (price, quality, network
topology, ...) and runs a Monte Carlo batch at each grid point.  This
directly answers business questions like:

* "What's the price elasticity inflection?"
* "How much does quality need to improve before adoption takes off?"
* "Does a denser social network meaningfully change diffusion speed?"

Design
------
* ``SweepAxis`` names one parameter path + a list of values.
* ``ParameterSweep`` takes a template config + one or more axes.
* For each grid point, it clones the config, sets the parameter(s),
  runs a short Monte Carlo (default 5 seeds), and collects results.
* Output: ``sweep_results.csv`` with one row per grid point, plus
  individual MC artefacts in subdirectories.

Parameter paths use dot notation to navigate nested objects:
  - ``"product.price"`` -> ``config.product.price``
  - ``"sentiment_decay"`` -> ``config.sentiment_decay``
  - ``"product.brand_reputation"`` -> ``config.product.brand_reputation``
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product as cartesian_product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simulation.monte_carlo import MonteCarloReport, MonteCarloRunner
from simulation.runner import SimulationConfig


# ─────────────────────────────────────────────────────────────
# SWEEP AXIS
# ─────────────────────────────────────────────────────────────

@dataclass
class SweepAxis:
    """One dimension of a parameter sweep.

    Parameters
    ----------
    param : str
        Dot-delimited path into SimulationConfig.
        Examples: ``"product.price"``, ``"sentiment_decay"``.
    values : list
        Concrete values to sweep.

    Example
    -------
    >>> SweepAxis("product.price", [199, 299, 399, 499])
    """
    param: str
    values: List[Any]

    @property
    def name(self) -> str:
        """Short label for display and CSV headers."""
        return self.param.split(".")[-1]


# ─────────────────────────────────────────────────────────────
# GRID POINT RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class GridPointResult:
    """Outcome of one Monte Carlo batch at a specific parameter combination."""
    params: Dict[str, Any]
    mc_report: MonteCarloReport
    label: str = ""

    @property
    def final_mean(self) -> float:
        return self.mc_report.final_mean

    @property
    def final_ci_lower(self) -> float:
        return self.mc_report.final_ci_lower

    @property
    def final_ci_upper(self) -> float:
        return self.mc_report.final_ci_upper


# ─────────────────────────────────────────────────────────────
# SWEEP REPORT
# ─────────────────────────────────────────────────────────────

@dataclass
class SweepReport:
    """Aggregated results across the full parameter grid."""
    sweep_id: str
    axes: List[SweepAxis]
    grid_results: List[GridPointResult]
    n_seeds_per_point: int
    csv_path: Optional[str] = None
    summary_json_path: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# PARAMETER SWEEP RUNNER
# ─────────────────────────────────────────────────────────────

class ParameterSweep:
    """Grid search over simulation parameters with Monte Carlo at each point.

    Parameters
    ----------
    config_template : SimulationConfig
        Base configuration.  Cloned for each grid point.
    axes : list of SweepAxis
        Parameter dimensions to sweep.  Multiple axes produce a
        Cartesian product grid.
    mc_seeds : Sequence[int]
        Seeds for the Monte Carlo batch at each grid point.
        Default: 5 seeds (100-104).
    out_dir : str | Path
        Root output directory.
    quiet : bool
        Suppress per-step output from child runs.

    Example
    -------
    >>> sweep = ParameterSweep(
    ...     config_template=my_config,
    ...     axes=[
    ...         SweepAxis("product.price", [199, 299, 399]),
    ...         SweepAxis("product.brand_reputation", [0.3, 0.5, 0.7]),
    ...     ],
    ...     mc_seeds=range(100, 105),
    ... )
    >>> report = sweep.run()
    """

    def __init__(
        self,
        config_template: SimulationConfig,
        axes: List[SweepAxis],
        mc_seeds: Sequence[int] = range(100, 105),
        out_dir: str | Path = "sweeps",
        sweep_id: Optional[str] = None,
        quiet: bool = True,
    ) -> None:
        if not axes:
            raise ValueError("ParameterSweep requires at least one SweepAxis.")

        self.config_template = config_template
        self.axes = axes
        self.mc_seeds = list(mc_seeds)
        self.quiet = quiet

        self.sweep_id = sweep_id or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
        self.sweep_dir = Path(out_dir) / self.sweep_id
        self.sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────

    def run(self) -> SweepReport:
        """Run the full parameter grid.  Synchronous entry point."""
        grid = self._build_grid()
        results: List[GridPointResult] = []

        self._print_header(grid)

        for i, combo in enumerate(grid, 1):
            label = ", ".join(f"{k}={v}" for k, v in combo.items())
            print(f"\n{'='*50}")
            print(f"  Grid point {i}/{len(grid)}: {label}")
            print(f"{'='*50}")

            cfg = self._apply_params(combo)
            point_dir = self.sweep_dir / f"point_{i:03d}"

            mc = MonteCarloRunner(
                config_template=cfg,
                seeds=self.mc_seeds,
                out_dir=str(point_dir),
                run_id="mc",
                quiet=self.quiet,
            )
            mc_report = mc.run()

            results.append(GridPointResult(
                params=combo,
                mc_report=mc_report,
                label=label,
            ))

            print(
                f"  -> final adoption {mc_report.final_mean:.1%} "
                f"[{mc_report.final_ci_lower:.1%}, {mc_report.final_ci_upper:.1%}]"
            )

        report = SweepReport(
            sweep_id=self.sweep_id,
            axes=self.axes,
            grid_results=results,
            n_seeds_per_point=len(self.mc_seeds),
        )

        report.csv_path = str(self._write_csv(report))
        report.summary_json_path = str(self._write_summary(report))
        self._print_summary(report)

        return report

    # ── grid construction ──────────────────────────────────

    def _build_grid(self) -> List[Dict[str, Any]]:
        """Cartesian product of all axes."""
        if len(self.axes) == 1:
            axis = self.axes[0]
            return [{axis.param: v} for v in axis.values]

        keys = [a.param for a in self.axes]
        value_lists = [a.values for a in self.axes]
        return [
            dict(zip(keys, combo))
            for combo in cartesian_product(*value_lists)
        ]

    def _apply_params(self, params: Dict[str, Any]) -> SimulationConfig:
        """Deep-copy the template and set each param by dot-path."""
        cfg = copy.deepcopy(self.config_template)
        for path, value in params.items():
            _set_nested(cfg, path, value)
        return cfg

    # ── output ─────────────────────────────────────────────

    def _write_csv(self, report: SweepReport) -> Path:
        """One row per grid point with param values + MC results."""
        path = self.sweep_dir / "sweep_results.csv"
        param_cols = [a.param for a in report.axes]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                *param_cols,
                "final_mean", "ci_lower", "ci_upper", "ci_width",
                "n_seeds",
            ])
            for gp in report.grid_results:
                writer.writerow([
                    *[gp.params[p] for p in param_cols],
                    round(gp.final_mean, 4),
                    round(gp.final_ci_lower, 4),
                    round(gp.final_ci_upper, 4),
                    round(gp.mc_report.final_ci_width, 4),
                    report.n_seeds_per_point,
                ])
        return path

    def _write_summary(self, report: SweepReport) -> Path:
        path = self.sweep_dir / "sweep_summary.json"
        payload = {
            "sweep_id": report.sweep_id,
            "axes": [
                {"param": a.param, "values": a.values}
                for a in report.axes
            ],
            "n_seeds_per_point": report.n_seeds_per_point,
            "n_grid_points": len(report.grid_results),
            "results": [
                {
                    "params": gp.params,
                    "final_mean": round(gp.final_mean, 4),
                    "ci_lower": round(gp.final_ci_lower, 4),
                    "ci_upper": round(gp.final_ci_upper, 4),
                    "ci_width": round(gp.mc_report.final_ci_width, 4),
                }
                for gp in report.grid_results
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    # ── console output ──────────────────────────────────────

    def _print_header(self, grid: List[Dict[str, Any]]) -> None:
        print()
        print("=" * 64)
        print(f"  PARAMETER SWEEP - {self.config_template.product.name}")
        for axis in self.axes:
            print(f"  {axis.param}: {axis.values}")
        print(f"  Grid points: {len(grid)} | MC seeds per point: {len(self.mc_seeds)}")
        print(f"  Output: {self.sweep_dir}")
        print("=" * 64)

    def _print_summary(self, report: SweepReport) -> None:
        print()
        print("=" * 64)
        print(f"  SWEEP SUMMARY - {self.config_template.product.name}")
        print("=" * 64)

        # Header
        param_names = [a.name for a in report.axes]
        header = "  " + "  ".join(f"{n:>12}" for n in param_names)
        header += "   final_mean   95% CI"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for gp in report.grid_results:
            vals = "  ".join(
                f"{gp.params[a.param]:>12}" for a in report.axes
            )
            print(
                f"  {vals}   "
                f"{gp.final_mean:>9.1%}   "
                f"[{gp.final_ci_lower:.1%}, {gp.final_ci_upper:.1%}]"
            )

        print()
        print(f"  CSV: {report.csv_path}")
        print(f"  JSON: {report.summary_json_path}")
        print("=" * 64)


# ─────────────────────────────────────────────────────────────
# DOT-PATH SETTER
# ─────────────────────────────────────────────────────────────

def _set_nested(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation.

    ``_set_nested(cfg, "product.price", 299)`` is equivalent to
    ``cfg.product.price = 299``.
    """
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
