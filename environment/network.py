"""
Social network topology builders.

Extension point: subclass ``NetworkBuilder`` and implement ``build()`` to add
new topologies (scale-free, random, community-clustered, …).  The simulation
runner accepts any NetworkBuilder via ``SimulationConfig.network_builder``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import networkx as nx

from core.state import NetworkPosition, Relationship

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# ABSTRACT INTERFACE
# ─────────────────────────────────────────────────────────────

class NetworkBuilder(ABC):
    """Abstract base for social network topology generators.

    Implementations create bidirectional ``Relationship`` edges between agents
    by mutating each agent's ``SocialGraph``.  The simulation runner calls
    ``build()`` once before the first step.
    """

    @abstractmethod
    def build(self, agents: List[ConsumerAgent]) -> None:
        """Create social connections between *agents* (mutates SocialGraph in place)."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable label for reports (e.g. ``SmallWorld(k=4, p=0.3)``)."""


# ─────────────────────────────────────────────────────────────
# WATTS–STROGATZ SMALL-WORLD NETWORK
# ─────────────────────────────────────────────────────────────

class SmallWorldNetwork(NetworkBuilder):
    """Watts–Strogatz small-world network.

    1. Start with a ring lattice where each node connects to *k* nearest
       neighbours.
    2. Rewire each rightward edge with probability *p*.

    The result has high clustering **and** short average path length — a good
    default for modelling real social networks.

    Parameters
    ----------
    k_neighbors : int
        Ring-lattice neighbourhood size (rounded up to the next even number).
    rewire_prob : float
        Probability of rewiring each edge (0 → regular lattice, 1 → random).
    seed : int | None
        Random seed for reproducible topologies.
    """

    def __init__(
        self,
        k_neighbors: int = 4,
        rewire_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.k = k_neighbors + (k_neighbors % 2)  # must be even
        self.p = rewire_prob
        self.seed = seed

    # ── public interface ──────────────────────────────────────

    def build(self, agents: List[ConsumerAgent]) -> None:
        rng = random.Random(self.seed)
        n = len(agents)

        if n < self.k + 1:
            adj = self._complete_adj(n)
        else:
            adj = self._watts_strogatz(n, rng)

        self._materialise_relationships(agents, adj, rng)
        self._assign_positions(agents, adj)

    def describe(self) -> str:
        return f"SmallWorld(k={self.k}, rewire={self.p})"

    # ── topology construction ─────────────────────────────────

    @staticmethod
    def _complete_adj(n: int) -> Dict[int, Set[int]]:
        """Fully-connected fallback when n ≤ k."""
        return {i: set(range(n)) - {i} for i in range(n)}

    def _watts_strogatz(self, n: int, rng: random.Random) -> Dict[int, Set[int]]:
        half_k = self.k // 2
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

        # Ring lattice: connect each node to k/2 neighbours on each side
        for i in range(n):
            for offset in range(1, half_k + 1):
                j = (i + offset) % n
                adj[i].add(j)
                adj[j].add(i)

        # Rewire: for each node, for each rightward neighbour, rewire with prob p
        for i in range(n):
            for offset in range(1, half_k + 1):
                j = (i + offset) % n
                if j not in adj[i]:
                    continue  # already rewired away by an earlier iteration
                if rng.random() >= self.p:
                    continue
                candidates = [x for x in range(n) if x != i and x not in adj[i]]
                if not candidates:
                    continue
                new_j = rng.choice(candidates)
                # swap edge
                adj[i].discard(j)
                adj[j].discard(i)
                adj[i].add(new_j)
                adj[new_j].add(i)

        return adj

    # ── relationship materialisation ──────────────────────────

    def _materialise_relationships(
        self,
        agents: List[ConsumerAgent],
        adj: Dict[int, Set[int]],
        rng: random.Random,
    ) -> None:
        """Convert adjacency sets into ``Relationship`` objects on each agent."""
        for i, agent in enumerate(agents):
            for j in adj[i]:
                target = agents[j]
                tid = target.agent_id

                if tid in agent.state.relationships.relationships:
                    continue  # already created from the other direction

                trust_a, influence_a = self._edge_params(agent, target, rng)
                trust_b, influence_b = self._edge_params(target, agent, rng)

                # A → B
                agent.state.relationships.relationships[tid] = Relationship(
                    target_id=tid,
                    trust=trust_a,
                    influence=influence_a,
                    dependency=round(rng.uniform(0.05, 0.25), 3),
                    contact_frequency=rng.randint(1, 5),
                )
                # B → A (potentially asymmetric)
                target.state.relationships.relationships[agent.agent_id] = Relationship(
                    target_id=agent.agent_id,
                    trust=trust_b,
                    influence=influence_b,
                    dependency=round(rng.uniform(0.05, 0.25), 3),
                    contact_frequency=rng.randint(1, 5),
                )

    @staticmethod
    def _edge_params(
        source: ConsumerAgent,
        target: ConsumerAgent,
        rng: random.Random,
    ) -> tuple:
        """Compute (trust, influence) for an edge *source → target*.

        Tier-based influence: innovators and early adopters exert more
        influence on their neighbours.  Co-location boosts trust.
        """
        trust = rng.uniform(0.3, 0.85)
        influence = rng.uniform(0.1, 0.6)

        tier = target.state.profile.adopter_tier
        if tier == "innovator":
            influence = min(1.0, influence + 0.25)
        elif tier == "early_adopter":
            influence = min(1.0, influence + 0.15)

        if source.state.profile.location == target.state.profile.location:
            trust = min(1.0, trust + 0.1)

        return round(trust, 3), round(influence, 3)

    # ── network-position assignment ───────────────────────────

    @staticmethod
    def _assign_positions(
        agents: List[ConsumerAgent],
        adj: Dict[int, Set[int]],
    ) -> None:
        n = len(agents)
        if n == 0:
            return
        avg_degree = sum(len(v) for v in adj.values()) / n
        for i, agent in enumerate(agents):
            degree = len(adj[i])
            if degree >= avg_degree * 1.5:
                agent.state.relationships.network_position = NetworkPosition.CENTRAL
            elif degree <= max(1, avg_degree * 0.5):
                agent.state.relationships.network_position = NetworkPosition.ISOLATED
            # else: stays PERIPHERAL (the default)


# ─────────────────────────────────────────────────────────────
# NETWORKX-BACKED WATTS–STROGATZ
# ─────────────────────────────────────────────────────────────

class NetworkXSmallWorld(SmallWorldNetwork):
    """Small-world topology using ``networkx.watts_strogatz_graph``.

    Identical interface and relationship-materialisation as
    :class:`SmallWorldNetwork`, but the adjacency is generated by
    the canonical networkx implementation.

    Using networkx gives us:
      * The textbook Watts–Strogatz rewiring rule.
      * Access to every other standard topology (scale-free, random,
        configuration-model) for free in Phase 2 via other builders.
      * Standard graph metrics (clustering coefficient, avg path length)
        for validation without re-implementing.
    """

    def _watts_strogatz(self, n: int, rng: random.Random) -> Dict[int, Set[int]]:
        # networkx uses numpy-style seeding; derive a deterministic seed.
        nx_seed = rng.randint(0, 2**31 - 1)
        g = nx.watts_strogatz_graph(n=n, k=self.k, p=self.p, seed=nx_seed)
        return {node: set(g.neighbors(node)) for node in g.nodes()}

    def describe(self) -> str:
        return f"NetworkXSmallWorld(k={self.k}, rewire={self.p})"
