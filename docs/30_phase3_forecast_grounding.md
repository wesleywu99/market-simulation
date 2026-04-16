# 30 -- Phase 3.0: Forecast Grounding

## Version
v1.0 (2026-04-16) -- All four subsystems implemented and smoke-tested.

## Status
Done. Ready for seed-case usage.

## Motivation

Phase 2.0 made agent behavior realistic at scale (bounded memory, decay,
cognitive diversity).  But a deeper assessment identified three gaps that
undermine *forecast reliability* -- the user's primary metric:

1. **Agents decide in a vacuum** -- no knowledge of competing brands,
   category price benchmarks, or typical purchase triggers.  A "skeptical"
   agent rejecting a 300 CNY dress has no idea that Zara charges 500 for
   comparable quality.

2. **Demographics are fixed** -- the same Chinese-city, tech-oriented
   population is used for drones, apparel, EVs.  A women's apparel case
   needs different age ranges, gender distribution, occupations, and
   lifestyle tags.

3. **No parameter sensitivity analysis** -- Monte Carlo sweeps seeds
   (stochastic variance) but not configuration parameters.  Business
   questions like "what's the price elasticity inflection?" require manual
   re-runs at different price points.

4. **No automated output validation** -- success criteria (Rogers ordering,
   S-curve shape, price monotonicity) existed in docs but were checked by
   manual inspection.

---

## P3-1: Category Context Injection

### Problem
Agents reason about a product with zero external reference frame.
Without knowing that "typical consumer drones cost 2000-8000 CNY" or
"the main competitor is DJI Mini 4 Pro at 4199 CNY," the LLM's price
evaluation is unanchored.

### Design
A new `market_context: Optional[str]` field on `SimulationConfig`.
The text is injected into the purchase decision prompt as a "Market context"
section between the risk/budget line and the timeline.

The system prompt instructs the agent to use this context for anchoring:
*"compare the product's price, quality, and features against the competing
brands and typical price ranges described."*

### Usage
```python
config = SimulationConfig(
    product=product,
    market_context="""
This is the consumer drone market in China (2024).
- Price range: budget 1000-2000 CNY, mid-range 2500-5000 CNY, premium 5000-12000 CNY.
- Key competitors: DJI Mini 4 Pro (4199 CNY, market leader), Autel EVO Nano+ (3999 CNY),
  FIMI X8 Mini (2499 CNY, budget alternative).
- Purchase triggers: travel photography, content creation, hobby/recreation.
- Common concerns: battery life, wind resistance, camera quality in low light.
""",
)
```

When `market_context` is `None` or empty, the prompt section is omitted
entirely (Jinja2 conditional block).

### Files changed
- `simulation/runner.py`: `SimulationConfig.market_context` field
- `agents/consumer.py`: `_build_purchase_prompt` and `decide_purchase` accept + pass context
- `prompts/purchase_decision.user.md`: conditional `{% if market_context %}` block
- `prompts/purchase_decision.system.md`: instruction to use market context

---

## P3-2: Configurable Population Profiles (PopulationSpec)

### Problem
Agent demographics are hardcoded in `simulation/controller.py`: ages 22-55,
Chinese city names, tech-oriented occupations, fixed income ranges per tier.
A women's apparel seed case targeting young urbanites needs completely
different population parameters.

### Design
New `PopulationSpec` dataclass in `simulation/population.py`.  Every field
is optional -- `None` means "use the built-in default."  This lets seed
cases override only what matters.

Configurable fields:

| Field | Default | Description |
|-------|---------|-------------|
| `age_range` | (22, 55) | Inclusive age bounds |
| `gender_distribution` | None | `{"female": 0.85, "male": 0.15}` -- injected as lifestyle tag |
| `income_ranges` | TIER_INCOME | Per-tier `(min, max)` monthly disposable |
| `occupations` | 10 tech-oriented | Occupation pool |
| `locations` | 5 Chinese cities | Location pool |
| `lifestyle_tags` | 7 tech tags | Tag pool (2-4 sampled per agent) |
| `education_weights` | Uniform | `{level: fraction}` for weighted sampling |
| `names` | 15 Chinese names | Name pool |
| `goal_template` | Drone goal text | Free-text, injected into agent goals |
| `tier_distribution` | Rogers 2.5/13.5/34/34/16 | Override Rogers fractions |
| `cognitive_style_overrides` | TIER_COGNITIVE_STYLES | Per-tier style pools |

### Wiring
- `SimulationConfig.population_spec: Optional[PopulationSpec]`
- `controller.py:_assign_tiers(n, spec)` uses `spec.tier_distribution` if provided
- `controller.py:make_agent(id, tier, idx, spec)` uses spec for all demographics
- Gender is injected as the *first* lifestyle tag (e.g. `["female", "fashion-forward", ...]`)
  so the LLM sees it naturally in the profile without a separate field.

---

## P3-3: Parameter Sweep Framework

### Problem
Monte Carlo answers "how much variance does this configuration have?"
But business questions ask "how does adoption change *as a function of*
price / quality / network density?" -- that requires sweeping config
parameters, not seeds.

### Design
New `ParameterSweep` class in `simulation/sweep.py`.

```
ParameterSweep(config_template, axes, mc_seeds)
    |
    +-- build Cartesian grid of parameter combos
    |
    +-- for each grid point:
    |     apply params to cloned config
    |     MonteCarloRunner(config, mc_seeds).run()
    |     collect GridPointResult
    |
    +-- write sweep_results.csv + sweep_summary.json
    +-- print comparison table
```

**SweepAxis** names a dot-delimited config path + values:
```python
SweepAxis("product.price", [199, 299, 399, 499])
SweepAxis("product.brand_reputation", [0.3, 0.5, 0.7])
```

Multiple axes produce a Cartesian product grid (e.g., 4 prices x 3
reputations = 12 grid points, each with 5 MC seeds = 60 total runs).

Parameter paths use `_set_nested(obj, "product.price", 299)` to set
attributes on arbitrarily nested config objects.

### Outputs

| File | Content |
|------|---------|
| `sweep_results.csv` | One row per grid point: param values + final_mean + CI |
| `sweep_summary.json` | Full structured results for programmatic analysis |
| `point_NNN/mc/` | Individual MC artefacts per grid point |

### Example
```python
from simulation.sweep import ParameterSweep, SweepAxis

sweep = ParameterSweep(
    config_template=config,
    axes=[SweepAxis("product.price", [199, 299, 399, 499])],
    mc_seeds=range(100, 110),  # 10 seeds per price point
)
report = sweep.run()
# -> sweep_results.csv with price vs adoption
```

---

## P3-4: Output Validation Suite

### Problem
The seed-case doc lists five success criteria (Rogers ordering, S-curve
shape, price monotonicity, etc.) but they were only checkable by manual
inspection of adoption curves.

### Design
New `simulation/validation.py` with modular check functions.  Each returns
a `ValidationResult(name, passed, message, details)`.

### Available checks

| Check | Scope | What it tests |
|-------|-------|---------------|
| `check_rogers_ordering` | Single run | Innovator adoption >= ... >= laggard |
| `check_adoption_monotonic` | Single run | Cumulative adoption never decreases |
| `check_scurve_shape` | Single run (8+ steps) | Second derivative changes sign |
| `check_tier_coverage` | Single run | All 5 tiers have >= 1 agent |
| `check_ci_width` | Monte Carlo | 95% CI width < threshold (default 15pp) |
| `check_price_monotonicity` | Sweep | Higher price -> lower adoption |

### Integration
- `validate_run(report)` runs all single-run checks
- `validate_mc(mc_report)` runs MC checks
- `validate_sweep(sweep_report)` runs sweep checks
- `SimulationRunner.print_report()` automatically calls `validate_run()`
  after printing the report

### A failing check does NOT mean the simulation is broken
It means the output deviates from theoretical expectations and warrants
human review.  Small populations, short runs, and extreme parameter
values can all produce legitimate deviations.

---

## Code cleanup (P3-5)

Also completed during this phase:
- **Removed dead code**: `SimulationController`, `StepResult`, `SimulationResult`
  classes from `controller.py` (replaced by `SimulationRunner` in Phase 1.5)
- **Cleaned imports**: removed unused `Event`, `Product`, `PurchaseDecision`,
  `PurchaseDecisionOutput`, `dataclass`, `field`, `Tuple` from controller
- **Updated docstring**: controller.py now describes itself as "agent factory"
- **MC clone_config**: now forwards all Phase 2.0 + 3.0 config fields
  (step_duration_days, decay, event_schedule, market_context, population_spec)
