# 30 — Influencer Seeding

## Version
v1.0 (2026-04-15) — First spec. To be implemented as Phase 1.5 upgrade #13.

## Status
📋 Designed. Implementation in progress.

## The problem

Phase 1 has agents whose `influence` and `social_capital` are drawn from uniform-ish distributions. The network (Watts-Strogatz) gives every agent roughly the same degree (k=4, narrow Poisson distribution after rewiring).

Result: **no real KOLs**. The most "influential" agent in a 100-agent run might have degree 8 vs. median 4 — that's gregarious, not a Karen-with-100k-followers.

Real consumer markets — especially fashion, beauty, electronics — are dominated by a small number of **opinion leaders** whose WOM carries 10-100× the weight of an average user.

## Decoupling structure from influence

Two orthogonal things are easy to confuse:

| Concept | Mechanism | Owned by |
|---|---|---|
| **Network structure** — who talks to whom | Edge generation | `NetworkBuilder` |
| **Influence amplitude** — whose opinion carries weight | Per-agent attribute | `InfluencerSeeding` |

Both contribute to KOL behavior, but they're separable:
- Scale-free topology + uniform influence → KOLs by reach
- Small-world topology + degree-based influence boost → KOLs by both
- Small-world topology + random influence boost → "celebrity" KOLs (unrelated to who they actually know)

Keeping them as separate ABCs lets us mix-and-match.

## The ABC

```python
class InfluencerSeeding(ABC):
    @abstractmethod
    def seed(self, agents: List[ConsumerAgent]) -> List[str]:
        """Amplify selected agents' influence/social_capital.
        Returns the list of agent_ids that were boosted (for telemetry)."""

    @abstractmethod
    def describe(self) -> str: ...
```

Runs **once**, after `network_builder.build()`, before step 1.

## Default impl — `DegreeBasedSeeding`

```python
class DegreeBasedSeeding(InfluencerSeeding):
    """Top-K nodes by degree get an influence + social_capital boost.

    Rationale: in any social network, the people with the most connections
    are *structurally* positioned to be opinion leaders. We amplify what
    the network already implies.
    """
    def __init__(
        self,
        top_k_fraction: float = 0.05,        # top 5% of agents
        influence_boost: float = 0.4,        # added to outgoing influence
        social_capital_boost: float = 0.3,   # added to agent's capital
        outgoing_trust_boost: float = 0.1,   # neighbors trust them more
    ): ...
```

**What it does:**
1. Compute degree for each agent.
2. Pick top-K by degree (ties broken by tier — innovators/early_adopters first).
3. For each selected agent A:
   - `A.state.resources.social_capital += social_capital_boost`
   - For every edge `B → A` in the network: `B's relationship to A`'s `influence += influence_boost`
   - Optionally bump `B → A` `trust += outgoing_trust_boost`
4. Mark `A.state.profile.is_influencer = True` for prompt enrichment.

**Why outgoing-edge boost, not the agent's own attribute?** Influence is a property of *the relationship*, not the person. A KOL has high influence over their followers; their followers have low influence back. The relationship asymmetry is what we want.

## Alternative impls

```python
class NoSeeding(InfluencerSeeding):
    """Identity — purely emergent. Use as A/B baseline."""
    def seed(self, agents): return []

class RandomSeeding(InfluencerSeeding):
    """Pick N at random. Models 'celebrity' KOLs unrelated to network position."""
    def __init__(self, n: int = 5, **boosts): ...

class PreTaggedSeeding(InfluencerSeeding):
    """Read AgentProfile.is_influencer set during agent creation.
    Use when seeding a specific real KOL ('agent_042 is Karen')."""
```

All four are interchangeable via `SimulationConfig.influencer_seeding`.

## Validation

A/B test once impl lands:
- Run with `NoSeeding` → record adoption curve as baseline.
- Run with `DegreeBasedSeeding(top_k_fraction=0.05)` → record adoption curve.
- Expected: faster initial cascade, higher ceiling for medium-trust products. Negligible for budget-gated products (System 1 still rejects).

Sanity check: confirm the boosted agents actually appear as WOM sources more often than baseline (count `wom_event.source == influencer_id` in trace log).

## Phase 2 — scale-free topology option

When we want the *topology itself* to reflect KOL power (KOLs have 100× neighbors, not 2×), add:

```python
class BarabasiAlbertNetwork(NetworkBuilder):
    """nx.barabasi_albert_graph — preferential attachment.
    Produces a true power-law degree distribution."""
```

Combined with `DegreeBasedSeeding`, this gives KOLs both *reach* and *amplitude*. For Phase 1.5 we hold off because (a) it complicates the Watts-Strogatz baseline comparison, and (b) the seeding mechanism alone proves the architecture.

## Non-goals

- **No dynamic influence shifts.** A KOL stays a KOL for the whole run. Real influence rises and falls, but modeling that adds two more strategies (gain function, decay function) for marginal benefit.
- **No reciprocal effects.** A KOL's high outgoing influence doesn't automatically mean their followers' opinions affect *them* less. We model directional edges already; this stays orthogonal.
- **Not a substitute for real persona work.** A KOL needs to *be* somebody — fashion expert, tech reviewer, mom-influencer. That comes from the seed-case persona spec, not from this strategy.
