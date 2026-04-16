# 10 — Phase 1 MVP Architecture

## Version
v1.0 (2026-04-15) — Baseline, end-to-end tested. This is what currently runs in `main.py`.

## Status
✅ Implemented and validated. Test run: 15 agents × 4 steps reached 47% adoption with a properly-shaped Rogers curve (innovator 100% → laggard 0%) and working WOM propagation (6 messages → 7 receptions).

## High-level architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SimulationRunner                            │
│  (orchestrates steps, owns state, swappable strategy injection)  │
└──────────────┬──────────────────────────────────────────────────┘
               │
    ┌──────────┼──────────────────────────────────┐
    ▼          ▼                                  ▼
┌────────┐ ┌────────────┐                  ┌────────────┐
│ Agents │ │ Network    │                  │ WOM Engine │
│        │ │ Builder    │                  │            │
│ (LLM-  │ │ (ABC)      │                  │ (ABC)      │
│ backed)│ │            │                  │            │
└────────┘ └────────────┘                  └────────────┘
              │                                 │
              ▼                                 ▼
         Watts-Strogatz                 Trust-weighted
         small-world                    Independent Cascade
         (k=4, p=0.3)                   + LLM semantic update
```

## Module responsibilities

| Module | Responsibility | Key types |
|---|---|---|
| `core/state.py` | Agent state dataclasses (7-variable BDI+) | `AgentState`, `Goals`, `BeliefSystem`, `Resources`, `RelationshipGraph`, `Memories`, `Vulnerabilities`, `Conflicts` |
| `core/events.py` | Event log primitives | `Event`, `EventType` |
| `agents/consumer.py` | The LLM-powered consumer agent | `ConsumerAgent` |
| `environment/product.py` | Product spec + sales counters | `Product` |
| `environment/network.py` | Social network topology (ABC + impl) | `NetworkBuilder`, `SmallWorldNetwork` |
| `environment/wom.py` | WOM propagation strategy (ABC + impl) | `WOMEngine`, `TrustWeightedWOM`, `WOMReception` |
| `llm/schemas.py` | Pydantic validators for LLM outputs | `PurchaseDecisionOutput`, `WOMOutput` |
| `llm/dispatcher.py` | Sync OpenAI-compatible client wrapper | `call_llm()` |
| `simulation/controller.py` | Agent factory + Rogers tier assignment | `make_agent()`, `_assign_tiers()` |
| `simulation/metrics.py` | Per-step metrics + reporting | `MetricsCollector`, `StepMetrics` |
| `simulation/runner.py` | Top-level simulation loop | `SimulationRunner`, `SimulationConfig`, `SimulationReport` |

## Key design decisions

### 1. ABC-based strategy injection
Both `NetworkBuilder` and `WOMEngine` are abstract base classes. The runner accepts them via `SimulationConfig`. This means swapping topologies (e.g. scale-free vs small-world) or WOM models (e.g. linear threshold vs independent cascade) requires writing one class, not touching the runner.

### 2. Four-phase step loop
Every step runs exactly these phases in order:

1. **Identify evaluators** — who re-enters the decision pool this step?
   - Step 1: everyone (product launch)
   - Step N>1: deferred-until-N agents + WOM recipients since last step
2. **Purchase decisions** — each evaluator calls the LLM, returns BUY / DEFER / REJECT
3. **WOM generation + propagation** — new buyers generate WOM; propagated via WOMEngine
4. **Metrics snapshot** — `StepMetrics` captures state for reporting

### 3. Clean separation: WOM computation vs state mutation
`WOMEngine.compute_receptions()` returns a list of `WOMReception` objects. The runner is responsible for applying them (calling `agent.receive_wom()`). This means the WOM strategy can be unit-tested without running agents, and the runner can add telemetry/trace without modifying strategies.

### 4. Deferred-step re-entry
Agents that defer specify a `deferred_until` step. The runner keeps a `Dict[int, Set[str]]` of step → agent IDs and pops the relevant set at the start of each step. Agents can re-enter the pool multiple times.

### 5. WOM recipients re-enter evaluation pool
An agent who receives WOM since their last evaluation is added to `wom_recipients`. At the next step, they re-evaluate — even if they previously rejected or weren't ready. This is the primary mechanism for diffusion.

## The agent decision flow (Phase 1)

```
agent.decide_purchase(product, adopted_ids, current_step):
    1. Assemble prompt with:
       - agent profile (persona, Rogers tier, income)
       - current state (budget, beliefs, memories)
       - product spec
       - social context (adopted_ids ∩ neighbors)
    2. Call LLM → PurchaseDecisionOutput (decision, reason, confidence, deferred_until)
    3. Validate + return
```

**Limitation**: every eligible agent calls the LLM every step they evaluate. No cheap pre-filter. This is the single biggest issue Phase 1.5 addresses.

## The WOM flow (Phase 1)

```
new_buyer.generate_wom(product, quality_rating, price_paid):
    1. LLM generates a short message + sentiment + emotional_intensity
WOMEngine.compute_receptions(source_id, source_graph, wom):
    2. For each neighbor, compute reception_prob = trust × intensity × share_prob
    3. Flip a coin, include neighbor if it passes
runner applies:
    4. target.receive_wom() — updates target's beliefs using a simple delta formula:
       sentiment_delta = {+0.12, 0, -0.12}[sentiment] × trust
       overall_sentiment += sentiment_delta
```

**Limitation**: belief update is a linear formula, not a semantic LLM call. Phase 1.5 adds a second LLM layer.

## Metrics produced

Per step:
- `evaluators`, `buyers`, `deferrers`, `rejecters`
- `cumulative_buyers`, `adoption_rate`
- `wom_messages_generated`, `wom_receptions`
- `tier_adoption`: `{tier: {total, adopted, rate}}` for each Rogers tier

Final report:
- Adoption curve (bar chart, ASCII)
- Per-tier adoption breakdown
- Cumulative WOM stats

## Known issues carried into Phase 1.5

| Issue | Severity | Fix in Phase 1.5 |
|---|---|---|
| LLM sometimes returns `deferred_until: 2024` (year, not step) | Medium | Prompt constraint + validator clamp |
| Every evaluation = 1 LLM call (cost scales linearly) | High | System 1 pre-filter |
| WOM belief update is a flat formula, not semantic | Medium | Two-layer WOM: IC + LLM semantic update |
| Sync LLM calls — wall-clock doesn't scale | High | Async + `asyncio.gather` |
| Prompts live in Python string literals | Low | Externalize to `.md` templates with Jinja |
| No run-to-run variance quantification | Medium | Monte Carlo wrapper |

## Configuration snapshot

```python
SimulationConfig(
    product=Product(...),
    n_agents=15,
    n_steps=4,
    seed=42,
    network_builder=SmallWorldNetwork(k=4, rewire_prob=0.3, seed=42),
    wom_engine=TrustWeightedWOM(close_threshold=0.6, seed=42),
)
```

## LLM configuration (.env)

```
LLM_API_KEY=...
LLM_BASE_URL=https://api.longcat.chat/openai
LLM_MODEL_NAME=LongCat-Flash-Chat
```

`LongCat-Flash-Chat` was chosen over `LongCat-Flash-Thinking-2601` for ~5x latency improvement. Thinking-style models may return for high-stakes decisions in Phase 2 if quality warrants it.
