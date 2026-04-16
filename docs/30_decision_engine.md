# 30 — Decision Engine (System 1 / System 2)

## Version
v1.0 (2026-04-15) — First impl landed in `agents/filters.py`.

## Status
✅ System 1 (code filter) implemented and validated. System 2 (LLM) unchanged from Phase 1.

## The problem

Every purchase decision in Phase 1 was an LLM call, even when the answer was obvious:

- "Can a ¥3000/mo agent afford a ¥3999 product?" → no, no LLM needed.
- "Will a laggard evaluate a product none of their friends have bought?" → no, no LLM needed.

These trivial decisions blew through budget and wall-clock time.

## The architecture

A two-tier decision pipeline inspired by Kahneman's System 1 / System 2:

```
                    evaluators for step N
                             │
                             ▼
                   ┌──────────────────────┐
                   │  DecisionFilter      │  ← System 1 (code)
                   │  .check(agent, ...)  │
                   └──────────────────────┘
                        │             │
                 decision=None   decision=PurchaseDecisionOutput
                        │             │
                        ▼             ▼
               ┌──────────────┐   (skip LLM,
               │ LLM dispatch │    use filter's decision)
               │ (System 2)   │
               └──────────────┘
                        │
                        ▼
                  final decision
```

## The ABC

```python
class DecisionFilter(ABC):
    def check(
        self,
        agent: ConsumerAgent,
        product: Product,
        adopted_ids: set,
        current_step: int,
    ) -> FilterResult:
        """Return FilterResult(decision=None, ...) to pass to LLM,
        or FilterResult(decision=<output>, ...) to short-circuit."""
```

## The default impl — `System1Filter`

Two gates, checked in order. The first to trip wins.

### Gate 1: Budget

```python
if product.price > agent.budget * budget_headroom_factor:
    return REJECT
```

- `budget_headroom_factor` default: `0.5`.
- Rationale: ~nobody spends more than half their disposable income on a single durable item on impulse.
- Tune: lower (0.3) for truly conservative markets, higher (0.8) for credit-financed high-value products.

### Gate 2: Social proof

```python
if adopted_neighbors / total_neighbors < tier_threshold:
    return DEFER
```

- Per-tier thresholds (Rogers-aligned):

| Tier | Threshold | Rationale |
|---|---|---|
| innovator | 0.00 | Leads — never needs permission |
| early_adopter | 0.00 | Early signal seeker — no threshold |
| early_majority | 0.15 | Needs a handful of validating peers |
| late_majority | 0.40 | Needs the trend to be obvious |
| laggard | 0.65 | Only after everyone else |

- Skipped at step 1 (product launch — nobody has adopted anything yet, so all tiers would defer forever).

## Results from first validation run (15 agents × 4 steps)

```
System 1      : 16 rejects, 0 defers (saved 73% of LLM calls)
System 2 (LLM): 6 calls
```

73% reduction in LLM calls — on target for the Phase 1.5 validation bar (≥50%).

The budget gate dominated because the product (¥3999) exceeded 50% headroom for most agents. When we switch to the affordable women's apparel seed case (¥150-400 price range), we expect the **social proof gate** to dominate instead — which will better exercise the tier dynamics.

## Swapping filters

Three supported patterns:

### Disable filtering (baseline / debug)
```python
from agents.filters import NullFilter
SimulationConfig(..., decision_filter=NullFilter())
```

### Custom thresholds
```python
SimulationConfig(..., decision_filter=System1Filter(
    budget_headroom_factor=0.3,   # stricter
    tier_social_thresholds={"laggard": 0.75, ...},
))
```

### New filter logic
Subclass `DecisionFilter`, implement `check()` and `describe()`. No runner changes needed.

## Future filters to consider

- **Awareness gate** — reject if agent has had zero WOM exposure AND tier ≥ late_majority. Forces adoption to happen via WOM, not advertising.
- **Fatigue gate** — defer (with exponential backoff) if the agent has already evaluated this product 3+ times without changing decision.
- **Quality-mismatch gate** — reject if product `quality < tier_min_quality` (laggards tolerate lower quality; innovators do not).

## Non-goals

- The filter is **deterministic** — no stochastic short-circuiting. Keep randomness in the network/WOM layers where it's scientifically meaningful.
- The filter does NOT examine belief state (sentiment, recent memories). Those are System 2 concerns. Keep Gates 1 and 2 orthogonal to each other so they compose cleanly.
