# 30 — Monte Carlo Wrapper

## Version
v1.0 (2026-04-15) — Initial implementation.

## Status
✅ Implemented in `simulation/monte_carlo.py`.

## Why

A single simulation run is one realisation of a stochastic process. Three sources of randomness live inside the runner:

1. **Network rewiring** — Watts–Strogatz `p` produces a different topology per seed.
2. **Independent-Cascade reception** — Layer-1 WOM gate flips a coin per (source, target) edge.
3. **LLM temperature** — at any non-zero temperature, identical prompts produce different outputs.

Reading a single adoption curve as "the answer" hides this variance. The Monte Carlo wrapper sweeps a list of seeds, runs the same configuration N times, and reports the mean curve plus a 95% confidence interval band.

## Design

```
MonteCarloRunner(config_template, seeds=[100, 101, …])
    │
    ├─ for each seed:
    │     deep-copy config + reseed strategies
    │     SimulationRunner(cfg).run_async()   ← reuses existing runner
    │     collect adoption_curve + final_adoption_rate
    │
    └─ aggregate per step:
          mean
          95% CI via Student's t critical value (df = n-1)
          fall back to 1.96 (normal approx) for n > 30
```

### Per-run config cloning

Strategy objects (`NetworkBuilder`, `WOMEngine`, `BeliefUpdater`, …) hold their own RNG seed in their constructor. Just overriding `SimulationConfig.seed` would leave those internal seeds frozen across runs, defeating the point of a sweep.

`MonteCarloRunner._clone_config(seed)` therefore:

1. `copy.deepcopy` of the `Product` (so accumulated `cumulative_sales` doesn't leak between runs).
2. `copy.deepcopy` of every injected strategy.
3. If the cloned strategy has a `.seed` attribute, set it to the per-run seed.

This convention assumes strategies expose `.seed` as a public attribute. Both `SmallWorldNetwork` and `TrustWeightedWOM` already do; new ABC implementations should follow suit if they need reseeding.

### Sequential, not concurrent at the run level

`SimulationRunner.run_async()` already parallelises within-step LLM calls via `asyncio.gather`. Running multiple simulations concurrently would multiply the LLM-call concurrency and:

- Saturate the dispatcher's `Semaphore` (which exists precisely to cap parallel calls against LongCat).
- Interleave per-step console output, making logs unreadable.
- Not buy meaningful wall-clock improvement once the semaphore is saturated.

So MC runs are sequential at the run level. The `quiet=True` default redirects per-step stdout to a buffer so only one progress line per run reaches the console.

### CI computation

Two-sided 95% confidence intervals using Student's t with df = n − 1:

```
margin = t_critical(df) × stdev / sqrt(n)
CI = [mean − margin, mean + margin]
```

Critical values are baked into a small lookup table (`_T_95`) covering df = 1..30, with linear interpolation for gaps and z = 1.96 for df > 30. This avoids a scipy dependency for ~20 lines of stats.

For n = 1 the CI collapses to `[mean, mean]` (no variance to estimate).

## Outputs

Each MC run writes three artefacts under `monte_carlo/<run_id>/`:

| File | Purpose |
|---|---|
| `adoption.csv` | Per-step rows: `step, mean, ci_lower, ci_upper, run_0_seed100, …` — Pandas-friendly. |
| `summary.json` | Top-level aggregates: per-run finals, mean curve, CI bands, config label. |
| `adoption.png` | Mean curve + shaded 95% CI band + faint individual run traces. Skipped (with warning) if matplotlib is unavailable. |

The console summary also prints a text-bar adoption curve with CI bounds per step.

## Usage

```python
from simulation.monte_carlo import MonteCarloRunner
from simulation.runner import SimulationConfig
# … build product + strategies as in main.py …

template = SimulationConfig(
    product=product,
    n_agents=100,
    n_steps=50,
    seed=42,                       # overridden per run
    network_builder=SmallWorldNetwork(k_neighbors=4, rewire_prob=0.3),
    wom_engine=TrustWeightedWOM(close_trust_threshold=0.6),
    decision_filter=System1Filter(),
    experience_sampler=BlindBoxExperience(),
    influencer_seeding=DegreeBasedSeeding(top_k_fraction=0.1),
)

mc = MonteCarloRunner(
    config_template=template,
    seeds=range(100, 120),         # 20 runs
    out_dir="monte_carlo",
    quiet=True,                    # suppress per-step logs from child runs
)
report = mc.run()
print(f"Final adoption: {report.final_mean:.1%} "
      f"95% CI [{report.final_ci_lower:.1%}, {report.final_ci_upper:.1%}]")
```

## Validation target

From the [Phase 1.5 plan](20_phase1.5_upgrade_plan.md): "95% CI at step 50 < 15 percentage points (tight enough to be useful)." This is the headline check before declaring the simulator's predictions trustworthy enough to inform real product decisions.

## Smoke-test result

3 runs × 3 steps × 10 agents on the SkyView Pro X1 drone (seeds 100–102):

```
Per-run finals: 30%, 30%, 20%
Mean: 26.7%   95% CI [12.3%, 41.0%]   width 28.7%
```

The CI is wide because n=3 with high tier variance — the test confirms aggregation works, not that the simulator has converged. Production runs should use ≥ 20 seeds.

## Cost note

A 20-run × 100-agent × 50-step sweep is ~20 × the per-run cost. With System 1 saving ~70% of LLM calls and the dispatcher's semaphore capping parallelism, expected wall-clock is roughly proportional. Do a small calibration run before launching a full sweep.
