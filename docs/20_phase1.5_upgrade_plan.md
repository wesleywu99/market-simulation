# 20 — Phase 1.5 Upgrade Plan

## Version
v1.2 (2026-04-16) — All phases through 3.0 complete.

## Status
All Phase 1.5 (13 items), Phase 2.0 (5 subsystems), and Phase 3.0 (4 subsystems) complete.
- Phase 2.0: [30_phase2_agent_resilience.md](30_phase2_agent_resilience.md)
- Phase 3.0: [30_phase3_forecast_grounding.md](30_phase3_forecast_grounding.md)

## Motivation

Phase 1 MVP works end-to-end at 15 agents × 4 steps. The user's target scale is **100 agents × 50 steps**. Three blockers prevent naïve scaling:

1. **Cost** — 100 agents × 50 steps ≈ 5,000 decisions × some WOM = ~8,000 LLM calls per run. At ~$0.003/call this is ~$24/run before any Monte Carlo.
2. **Latency** — sync calls at ~3s each ≈ 6.7 hours per run. Unusable.
3. **Fidelity** — the flat-formula belief update and one-LLM-per-decision design miss the nuance the user wants (negative WOM asymmetry, cognitive updates, social proof thresholds).

Phase 1.5 addresses all three.

## Implementation priority (highest ROI first)

| # | Upgrade | Status | Unblocks | Est. effort |
|---|---|---|---|---|
| 1 | **Async LLM dispatcher** | ✅ done | 100x50 scale (wall-clock) | S |
| 2 | **System 1 pre-filter** | ✅ done | 100x50 scale (cost) | M |
| 3 | **Two-layer WOM (IC + LLM semantic)** | ✅ done | Fidelity | M |
| 4 | **networkx-based topology** | ✅ done | Realism + research | S |
| 5 | **Prompt externalization** (`.md` + Jinja) | ✅ done | Iteration speed | S |
| 6 | **Bug fix: `deferred_until` clamp** | ✅ done | Data quality | XS |
| 7 | **Decision trace log (JSONL)** | ✅ done | Debuggability | S |
| 8 | **Negative WOM asymmetry** | ✅ done (with #3) | Realism | XS |
| 9 | **Awareness decay** | ✅ done | Realism | XS |
| 10 | **Monte Carlo wrapper + CI** | ✅ done | Decisions under uncertainty | M |
| 11 | **Per-step state snapshots** | ✅ done | Post-hoc analysis | S |
| 12 | **Semantic blind-box quality** | ✅ done | WOM realism, generalization | M |
| 13 | **Influencer seeding** | ✅ done | KOL dynamics | S |

All 13 Phase 1.5 items are complete. Items 1-6 were blocking for 100x50 scale. Items 7-11 added analytical depth. Items 12-13 added fidelity for consumer-product simulations.

### Phase 2.0 agent-resilience upgrades (2026-04-16)

| ID | Subsystem | Status | Addresses |
|----|-----------|--------|-----------|
| P2-1 | Time semantics + budget refresh | ✅ done | Timeless decisions, frozen budgets |
| P2-2 | External event queue | ✅ done | Closed-system limitation |
| P2-3 | Cognitive style + per-tier temperature | ✅ done | Agent homogeneity |
| P2-4 | Memory eviction + belief dedup | ✅ done | Unbounded state growth |
| P2-5 | Sentiment & intent decay | ✅ done | Sentiment/intent saturation |

See [30_phase2_agent_resilience.md](30_phase2_agent_resilience.md) for full design documentation.

### Phase 3.0 forecast-grounding upgrades (2026-04-16)

| ID | Subsystem | Status | Addresses |
|----|-----------|--------|-----------|
| P3-1 | Category context injection | ✅ done | Agents decide in a vacuum |
| P3-2 | Configurable population (PopulationSpec) | ✅ done | Fixed demographics across cases |
| P3-3 | Parameter sweep framework | ✅ done | No sensitivity analysis |
| P3-4 | Output validation suite | ✅ done | Manual-only output checking |
| P3-5 | Code cleanup + hardening | ✅ done | Dead code, unused imports |

See [30_phase3_forecast_grounding.md](30_phase3_forecast_grounding.md) for full design documentation.

See:
- [30_product_quality.md](30_product_quality.md) for the blind-box design.
- [30_influencer_seeding.md](30_influencer_seeding.md) for the seeding strategy.
- [30_two_layer_wom.md](30_two_layer_wom.md) for the IC + belief-updater split.
- [30_prompt_externalization.md](30_prompt_externalization.md) for the Jinja2 prompt loader.
- [30_monte_carlo.md](30_monte_carlo.md) for the seed-sweep / CI wrapper.

### Validated results so far (15×4 smoke test)

- Async: 5 parallel decisions complete in ~single-call wall time.
- System 1: **73%** of LLM calls short-circuited in a high-price scenario.
- Bug fix: no more `deferred_until: 2024` artefacts.
- networkx topology: exact `n × k / 2` edge count, average degree matches k.

## Upgrade 1 — Async LLM dispatcher

**Why first:** every other upgrade that adds LLM calls (two-layer WOM) makes latency worse. Fix the substrate first.

**Design:**
- Replace `openai.OpenAI` with `openai.AsyncOpenAI` in `llm/dispatcher.py`.
- `call_llm` becomes `async def call_llm(...)`.
- In the runner, collect all evaluators for a step, then `await asyncio.gather(*[agent.decide_purchase(...) for agent in evaluators])`.
- Same treatment for WOM generation within a step.
- Preserve the retry logic; retries are per-task.

**Constraint:** LongCat API rate limits are unknown. We add a `asyncio.Semaphore` (configurable, default 10) to cap concurrency.

**Non-goal:** we are **not** making the simulation itself async (agents don't interleave within a step). We're only parallelizing the within-phase LLM calls.

## Upgrade 2 — System 1 pre-filter (code-layer)

**Why:** ~60-70% of decisions in the Phase 1 run were REJECT or DEFER for reasons the LLM didn't need to think about (budget, no social proof). Skipping these LLM calls is pure cost savings.

**Design:**
- New ABC: `DecisionFilter` in `agents/filters.py`.
- Default impl `System1Filter` with two gates:
  1. **Budget gate:** if `product.price > agent.budget × budget_headroom_factor (0.5)` → reject, skip LLM. Rationale: nobody spends >50% of disposable income on a single durable item on impulse.
  2. **Social proof gate:** read agent's Rogers tier. If `adopted_neighbors / total_neighbors < tier_threshold`, return DEFER. Thresholds:
     - innovator: 0.0 (no threshold — always evaluates)
     - early_adopter: 0.0
     - early_majority: 0.15
     - late_majority: 0.40
     - laggard: 0.65
- `ConsumerAgent.decide_purchase()` calls `filter.check(product, self)` first; if filter returns a decision, return it without LLM call.
- Filter is injectable via `SimulationConfig.decision_filter`.

**Metrics instrumentation:** track `system1_rejects`, `system1_defers`, `system2_calls` per step. Target: ≥50% reduction in LLM calls vs pure Phase 1.

## Upgrade 3 — Two-layer WOM (Independent Cascade + LLM semantic)

**Why:** the current flat `sentiment_delta × trust` belief update is a crude approximation. Real people don't just shift sentiment — they update *which* attributes they care about, *what* they now believe about the product, and *how* they now feel about the source. Only an LLM can do this nuanced update cheaply.

**Design:**

```
Layer 1 (code, cheap): Independent Cascade gate
    activation_prob = f(source.influence, edge.trust, wom.emotional_intensity)
    coin flip → include or drop target

Layer 2 (LLM, semantic): BeliefUpdater ABC
    Only runs for targets that pass Layer 1
    Input: target's current BeliefSystem, source's WOM message, edge.trust
    Output: updated BeliefSystem + (optionally) a purchase_intent signal
```

**New ABC:** `BeliefUpdater` in `agents/belief.py`. Default: `LLMBeliefUpdater`. Fallback for cost: `LinearBeliefUpdater` (current formula).

**Key refinement — negative WOM asymmetry:** the linear fallback uses **asymmetric deltas**: positive = +0.10, negative = -0.22. Rationale: loss aversion + negativity bias are well-documented in consumer research. LLM updater doesn't need this hardcoded — it should emerge from the prompt.

## Upgrade 4 — networkx-based topology

**Why:** `new mvp ideas.txt` calls for `nx.watts_strogatz_graph(n=100, k=4, p=0.1)`. Our hand-rolled implementation works but deviates subtly from the canonical Watts-Strogatz rewiring rule. Moving to networkx gives us:
- Canonical algorithm
- Access to every other topology (scale-free, configuration model, etc.) for free in Phase 2
- Standard graph metrics (clustering coefficient, average path length) for validation

**Design:**
- New impl: `NetworkXSmallWorld(NetworkBuilder)` uses `nx.watts_strogatz_graph(n, k, p)`.
- Retain `SmallWorldNetwork` as a fallback for now; deprecate in Phase 2.
- After building the edge set, iterate edges and attach `Relationship` objects with trust + influence as before.

## Upgrade 5 — Prompt externalization

**Why:** prompts currently live in f-strings inside `consumer.py`. Iterating a prompt requires editing Python. This is painful and error-prone.

**Design:**
- New directory: `prompts/`.
- One `.md` file per prompt: `purchase_decision.md`, `wom_generation.md`, `belief_update.md`.
- Jinja2 placeholders (`{{ agent.name }}`, `{{ product.price }}`).
- Thin loader: `llm/prompts.py` → `render_prompt(name, **context) -> str`.
- Tests verify each prompt renders without error given a fixture context.

**Side benefit:** prompts can be version-controlled and A/B tested via the config.

## Upgrade 6 — `deferred_until` clamp bug fix

**Why:** LLM sometimes returns `deferred_until: 2024` (year), breaking evaluator scheduling.

**Design:**
- Prompt: add explicit constraint "deferred_until must be an integer between `current_step + 1` and `current_step + 5`."
- Pydantic validator: clamp to `[current_step + 1, current_step + 5]`. Log a warning on clamp.

## Upgrade 7 — Decision trace log

**Why:** when an adoption curve looks wrong, we need to answer "why did agent X reject in step 3?" in seconds, not minutes.

**Design:**
- Every decision (System 1 or System 2) writes a JSONL row to `traces/run_{run_id}/decisions.jsonl`.
- Fields: step, agent_id, tier, decision, reason, confidence, filter_reason (if System 1), prompt_hash, raw_output.
- WOM trace: `wom.jsonl` — each reception attempt, pass/fail, reason.

## Upgrade 8 — Negative WOM asymmetry

See **Upgrade 3** — folded in as part of the `LinearBeliefUpdater` fallback.

## Upgrade 9 — Awareness decay

**Why:** in reality, if an agent hears about a product in step 3 and nothing refreshes their interest, by step 10 they've forgotten. Without decay, every past WOM equally influences every future decision.

**Design:**
- Each WOM memory in `Memories` gets a `received_step` timestamp.
- At decision time, compute `weight = 0.85 ^ (current_step - received_step)`.
- Apply weight when aggregating WOM evidence in the prompt.

## Upgrade 10 — Monte Carlo wrapper

**Why:** one run produces one adoption curve. It's tempting to read that curve as "the answer," but stochastic gates (network, IC, LLM temperature) introduce variance. We need confidence intervals.

**Design:**
- New module: `simulation/monte_carlo.py`.
- `MonteCarloRunner(config, n_runs=20, seeds=range(100, 120))`.
- Runs N simulations (async), aggregates adoption curves into mean + 95% CI bands.
- Output: CSV + matplotlib figure.

## Upgrade 11 — Per-step state snapshots

**Why:** post-hoc analysis — "how did beliefs of late_majority agents evolve over time?"

**Design:**
- At the end of each step, serialize `{agent_id: state.to_dict()}` to `traces/run_{run_id}/snapshots/step_{N}.json`.
- Lightweight — only capture `beliefs`, `resources.budget`, `memories` (compressed).

## Validation plan

Before declaring Phase 1.5 "done," run this sanity check battery:

1. **Rogers curve shape**: innovators adopt first, laggards last. Rejection when this fails.
2. **100×50 completes in <15 min** with async on.
3. **System 1 rate**: ≥50% of decisions short-circuited.
4. **Cost check**: run cost < $5 per run (estimated via token counter).
5. **Monte Carlo CI width**: 95% CI at step 50 < 15 percentage points (tight enough to be useful).
6. **Deterministic seed**: same seed produces same trajectory (given fixed LLM temperature ≤ 0.3).

## After Phase 1.5

- Load the first real seed case — affordable women's apparel. See [40_seed_case_womens_apparel.md](40_seed_case_womens_apparel.md).
- If the simulation produces plausible curves for a product with known-ish adoption, we have a useful forecaster.
- Phase 2 expands to competitor products, seed document ingestion, richer agent personas.
