"""
Influencer seeding strategies — KOL amplification.

In a Watts-Strogatz network, every agent has roughly the same degree.  No
amount of stochastic LLM noise produces real opinion leaders.  This module
amplifies selected agents' influence and social capital so that their WOM
carries the weight of a Karen-with-100k-followers, not just a slightly
gregarious neighbour.

Network *structure* (who connects to whom) is owned by ``NetworkBuilder``.
Influence *amplitude* (whose voice is loud) is owned here — these two
concerns are deliberately decoupled and can be swapped independently.

See ``docs/30_influencer_seeding.md`` for full design rationale.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# STRATEGY ABC
# ─────────────────────────────────────────────────────────────

class InfluencerSeeding(ABC):
    """Abstract base for KOL seeding strategies.

    Runs **once**, after the network is built and before the first step.
    Implementations mutate selected agents' state to amplify their WOM
    influence on neighbours.
    """

    @abstractmethod
    def seed(self, agents: List["ConsumerAgent"]) -> List[str]:
        """Apply the seeding strategy.  Returns the list of agent IDs boosted."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for reports."""


# ─────────────────────────────────────────────────────────────
# NO-OP BASELINE
# ─────────────────────────────────────────────────────────────

class NoSeeding(InfluencerSeeding):
    """Identity strategy — no agents are boosted.

    Use as the A/B baseline to measure the effect of any other seeding.
    """

    def seed(self, agents):
        return []

    def describe(self) -> str:
        return "NoSeeding"


# ─────────────────────────────────────────────────────────────
# DEGREE-BASED SEEDING
# ─────────────────────────────────────────────────────────────

# Tier-based tiebreaker priority (higher = more likely to be selected on ties)
_TIER_PRIORITY = {
    "innovator":       4,
    "early_adopter":   3,
    "early_majority":  2,
    "late_majority":   1,
    "laggard":         0,
}


class DegreeBasedSeeding(InfluencerSeeding):
    """Top-K agents by network degree get an influence + social-capital boost.

    Rationale: in any social network, the agents with the most connections
    are *structurally* positioned to be opinion leaders.  We amplify what
    the network already implies.

    Boosts are applied to **outgoing edges** (B → A) where A is the seeded
    agent.  This models that A's followers (B) attribute more weight to A's
    opinions, not that A's own opinion of B changes.

    Parameters
    ----------
    top_k_fraction : float
        Fraction of total agents to select as KOLs (default 5%).
    influence_boost : float
        Added to each ``B → A`` edge's ``influence`` attribute (clamped to 1.0).
    social_capital_boost : float
        Added to A's own ``social_capital`` (clamped to 1.0).  Affects WOM reach.
    trust_boost : float
        Optional bump to ``B → A`` ``trust``.  Default 0 — KOLs aren't always
        trusted more, just listened to more.
    """

    def __init__(
        self,
        top_k_fraction: float = 0.05,
        influence_boost: float = 0.4,
        social_capital_boost: float = 0.3,
        trust_boost: float = 0.0,
    ) -> None:
        self.top_k_fraction = top_k_fraction
        self.influence_boost = influence_boost
        self.social_capital_boost = social_capital_boost
        self.trust_boost = trust_boost

    def seed(self, agents: List["ConsumerAgent"]) -> List[str]:
        if not agents:
            return []

        k = max(1, int(round(len(agents) * self.top_k_fraction)))

        # Sort by (degree DESC, tier_priority DESC) for deterministic tie-breaking
        ranked = sorted(
            agents,
            key=lambda a: (
                -len(a.state.relationships.relationships),
                -_TIER_PRIORITY.get(a.state.profile.adopter_tier, 0),
                a.agent_id,  # final stable tiebreaker
            ),
        )
        selected = ranked[:k]
        selected_ids = {a.agent_id for a in selected}

        # Build agent_id → agent map for edge lookup
        agents_by_id = {a.agent_id: a for a in agents}

        for kol in selected:
            kol.state.profile.is_influencer = True
            kol.state.resources.social_capital = min(
                1.0, kol.state.resources.social_capital + self.social_capital_boost
            )

            # For each follower B → A, boost B's relationship attributes toward A
            for follower in agents:
                if follower.agent_id == kol.agent_id:
                    continue
                rel = follower.state.relationships.relationships.get(kol.agent_id)
                if rel is None:
                    continue
                rel.influence = min(1.0, rel.influence + self.influence_boost)
                if self.trust_boost:
                    rel.trust = min(1.0, rel.trust + self.trust_boost)

        return sorted(selected_ids)

    def describe(self) -> str:
        return (
            f"DegreeBasedSeeding(top={self.top_k_fraction:.0%}, "
            f"infl+={self.influence_boost}, cap+={self.social_capital_boost})"
        )


# ─────────────────────────────────────────────────────────────
# RANDOM SEEDING — "celebrity" KOLs unrelated to network position
# ─────────────────────────────────────────────────────────────

class RandomSeeding(InfluencerSeeding):
    """Pick N agents uniformly at random.

    Models KOLs whose status comes from outside the network (celebrities,
    media personalities) rather than from structural position.
    """

    def __init__(
        self,
        n: int = 5,
        influence_boost: float = 0.4,
        social_capital_boost: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.n = n
        self.influence_boost = influence_boost
        self.social_capital_boost = social_capital_boost
        self.rng = random.Random(seed)

    def seed(self, agents: List["ConsumerAgent"]) -> List[str]:
        if not agents:
            return []
        n = min(self.n, len(agents))
        selected = self.rng.sample(agents, n)
        for kol in selected:
            kol.state.profile.is_influencer = True
            kol.state.resources.social_capital = min(
                1.0, kol.state.resources.social_capital + self.social_capital_boost
            )
            for follower in agents:
                if follower.agent_id == kol.agent_id:
                    continue
                rel = follower.state.relationships.relationships.get(kol.agent_id)
                if rel is not None:
                    rel.influence = min(1.0, rel.influence + self.influence_boost)
        return sorted(a.agent_id for a in selected)

    def describe(self) -> str:
        return f"RandomSeeding(n={self.n})"


# ─────────────────────────────────────────────────────────────
# PRE-TAGGED SEEDING — explicit ("agent_042 is Karen")
# ─────────────────────────────────────────────────────────────

class PreTaggedSeeding(InfluencerSeeding):
    """Boost agents whose profile.is_influencer is already True at build time.

    Use when the seed case explicitly designates certain agents as KOLs
    (e.g. simulating a specific real-world influencer's reach).
    """

    def __init__(
        self,
        influence_boost: float = 0.4,
        social_capital_boost: float = 0.3,
    ) -> None:
        self.influence_boost = influence_boost
        self.social_capital_boost = social_capital_boost

    def seed(self, agents: List["ConsumerAgent"]) -> List[str]:
        selected = [a for a in agents if a.state.profile.is_influencer]
        for kol in selected:
            kol.state.resources.social_capital = min(
                1.0, kol.state.resources.social_capital + self.social_capital_boost
            )
            for follower in agents:
                if follower.agent_id == kol.agent_id:
                    continue
                rel = follower.state.relationships.relationships.get(kol.agent_id)
                if rel is not None:
                    rel.influence = min(1.0, rel.influence + self.influence_boost)
        return sorted(a.agent_id for a in selected)

    def describe(self) -> str:
        return "PreTaggedSeeding"
