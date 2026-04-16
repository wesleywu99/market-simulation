# 30 — Two-Layer Word-of-Mouth

## Version
v1.0 (2026-04-15) — First spec and implementation. Phase 1.5 upgrade #3.

## Status
✅ Implemented — `LinearBeliefUpdater` is the default; `LLMBeliefUpdater` is opt-in.

## The problem

Phase 1 shipped with a single WOM mechanism that conflated two different
concerns into one pass through the social graph:

1. **Who actually hears the message.** A function of network structure,
   trust, and the source's emotional intensity — cheap, code-only.
2. **What hearing the message does to the listener's beliefs.**
   The *semantic* update — which requires understanding the message
   content in context of the listener's current beliefs.

Phase 1 collapsed (2) into a fixed formula:

```python
sentiment_delta = {"positive": 0.12, "neutral": 0.0, "negative": -0.12}[sentiment]
target.overall_sentiment += sentiment_delta * trust
```

This is coarse — it cannot distinguish "my friend said the camera is
amazing (and I care about photography)" from "my friend said the app is
buggy (which I already know from another contact)." Real WOM updates are
selective and context-sensitive.

## The two-layer model

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Independent Cascade gate  (code, cheap)  │
│  ─────────────────────────────────────────────────  │
│  WOMEngine.compute_receptions(source, wom) →        │
│      list[WOMReception]                             │
│                                                     │
│  Decides WHO receives.  Inputs: trust, emotional    │
│  intensity, share probability.  Stochastic roll.    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2 — Belief updater  (per-reception)          │
│  ─────────────────────────────────────────────────  │
│  BeliefUpdater.update(target, source_id, sentiment, │
│                       message, trust, step) → None  │
│                                                     │
│  Decides HOW the target's beliefs move.             │
│  Two default impls: Linear (formula) or LLM.        │
└─────────────────────────────────────────────────────┘
```

Layer 1 is owned by `environment/wom.py` (the existing `WOMEngine` ABC).
Layer 2 is new — `agents/belief.py` introduces the `BeliefUpdater` ABC.

Both layers are pluggable via `SimulationConfig` (`wom_engine` and
`belief_updater` respectively), so the topology of propagation and the
mechanics of belief revision can evolve independently.

## Layer 2 implementations

### `LinearBeliefUpdater` (default)

A closed-form update. Same shape as the Phase 1 formula but with one
critical change: **asymmetric positive/negative deltas**.

```python
positive_sentiment_delta = +0.10
negative_sentiment_delta = -0.22   # ~2× larger magnitude
```

**Why asymmetric?** Consumer research consistently shows that negative
word-of-mouth has a larger attitudinal impact than equally-intense
positive WOM. Two well-established drivers:

- **Negativity bias** (Baumeister et al., 2001): bad is stronger than good.
- **Loss aversion** (Kahneman & Tversky): perceived risk of a negative
  attribute outweighs comparable perceived gain from a positive one.

The Phase 1 symmetric formula under-represented this asymmetry and
produced artificially optimistic adoption curves in runs with mixed WOM.

### `LLMBeliefUpdater` (opt-in)

One LLM call per reception. The prompt gives the LLM:

- Target's current overall sentiment and purchase intent.
- Target's last few product-related beliefs (to allow reinforcement
  vs. contradiction logic).
- Source's trust.
- The WOM message itself.

The LLM returns bounded deltas plus a new-belief predicate:

```json
{
  "sentiment_delta": -0.14,
  "intent_delta": -0.05,
  "new_belief_predicate": "has battery life well below advertised",
  "belief_confidence": 0.72,
  "memory_importance": 0.65,
  "memory_valence": -0.4,
  "reasoning": "Trusted source contradicts the brand's spec claim directly."
}
```

**Cost note.** At 100 agents × 50 steps × (say) 3 WOM receptions per
agent-step, this is ~15,000 extra LLM calls per run. Not free. The
updater is therefore **opt-in** — enable it selectively for fine-grained
seed-case tuning, use the linear default for scale.

### `describe()` outputs in reports

```
Belief updater: LinearBeliefUpdater(pos=+0.10, neg=-0.22, intent×0.15)
Belief updater: LLMBeliefUpdater(temp=0.5)
```

## Integration with the runner

The runner batches all per-reception belief updates for each buyer's WOM
message and awaits them together via `asyncio.gather`:

```python
update_tasks = []
for reception in receptions:
    target = self.agents_map.get(reception.target_id)
    if target is None or target.state.has_purchased:
        continue
    update_tasks.append(self.belief_updater.update(
        target=target, source_id=buyer.agent_id, sentiment=..., message=..., trust=..., step=step,
    ))
    self.wom_recipients.add(target.agent_id)

if update_tasks:
    await asyncio.gather(*update_tasks, return_exceptions=True)
```

This means `LLMBeliefUpdater` inherits the dispatcher's
`LLM_MAX_CONCURRENCY` semaphore — the same rate-limit gate used for
purchase decisions and WOM generation. No new concurrency primitive is
introduced.

## The legacy `ConsumerAgent.receive_wom` method

Kept for ad-hoc callers and tests, but it is now a thin async wrapper
around `LinearBeliefUpdater().update(...)`. The runner no longer calls
it — it talks to the configured `BeliefUpdater` directly.

## Architecture — what's generic, what's per-impl

| Concern | Generic | Per-impl |
|---|---|---|
| `BeliefUpdater` ABC | ✅ | — |
| `LinearBeliefUpdater` formula | — | ✅ (weights overridable) |
| `LLMBeliefUpdater` prompt | — | ✅ |
| State mutation helpers (`_apply_sentiment`, `_append_memory`, `_append_belief`) | ✅ (module-private) | — |
| Pydantic `BeliefUpdateOutput` schema | ✅ | — |
| Runner batching / `asyncio.gather` | ✅ | — |

The state-mutation helpers are deliberately module-level so that future
updaters (a Bayesian one, a rule-based one) can reuse them without
subclassing.

## Caveats and non-goals

- **No cross-agent state sharing.** Each reception updates exactly one
  target. We do not model "my friend told me, and that changed how I
  hear the next review" as a compound effect within a step; that
  emerges over time across steps.
- **No message fatigue.** Receiving the 10th positive review still
  counts the same as the 1st. Awareness/attention decay is tracked as a
  separate Phase 1.5 item (#9) and will apply an age-based weight when
  the LLM reads memories at decision time.
- **No source-specific learning.** Trust is a static edge attribute in
  Phase 1.5 — a listener does not increase/decrease trust in a source
  based on whether that source's past reviews turned out to be
  accurate. Planned for Phase 2.
- **LLMBeliefUpdater is not a one-size-fits-all upgrade.** It produces
  more realistic updates but adds significant LLM cost. The linear
  default is calibrated to be good enough for most runs; use the LLM
  updater for diagnostic deep-dives, not routine sweeps.
