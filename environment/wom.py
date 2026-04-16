"""
Word-of-mouth propagation engines.

Extension point: subclass ``WOMEngine`` and implement ``compute_receptions()``
to create alternative propagation strategies (viral, influencer-weighted,
geographic-proximity, …).  The runner injects the engine via
``SimulationConfig.wom_engine``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from core.state import SocialGraph
from llm.schemas import WOMOutput


# ─────────────────────────────────────────────────────────────
# DATA TRANSFER OBJECT
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WOMReception:
    """One successful word-of-mouth reception.

    Returned by ``WOMEngine.compute_receptions()`` to tell the runner
    *who* received the message and *how strongly*.  The runner then
    calls ``target.receive_wom()`` — the engine itself never touches
    agent state, keeping propagation logic and state-mutation separate.
    """
    target_id: str
    trust: float              # edge trust from source → target
    reception_strength: float  # combined probability that drove this reception


# ─────────────────────────────────────────────────────────────
# ABSTRACT INTERFACE
# ─────────────────────────────────────────────────────────────

class WOMEngine(ABC):
    """Abstract base for WOM propagation strategies.

    Implementations decide **which neighbours receive** a WOM message
    and at what strength.  They do *not* mutate agent state — that is
    the runner's job, using the returned ``WOMReception`` list.
    """

    @abstractmethod
    def compute_receptions(
        self,
        source_id: str,
        source_graph: SocialGraph,
        wom: WOMOutput,
    ) -> List[WOMReception]:
        """Return the list of neighbours who actually receive this WOM."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable label for reports."""


# ─────────────────────────────────────────────────────────────
# TRUST-WEIGHTED PROPAGATION
# ─────────────────────────────────────────────────────────────

class TrustWeightedWOM(WOMEngine):
    """Propagate WOM through the social graph, weighted by edge trust.

    Reception probability for each neighbour:

        P(receive) = trust × emotional_intensity × share_probability

    If the WOM targets ``close_friends`` only, neighbours below
    *close_trust_threshold* are skipped.

    Parameters
    ----------
    close_trust_threshold : float
        Minimum trust for a neighbour to count as a "close friend".
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        close_trust_threshold: float = 0.6,
        seed: Optional[int] = None,
    ):
        self.close_threshold = close_trust_threshold
        self.rng = random.Random(seed)

    def compute_receptions(
        self,
        source_id: str,
        source_graph: SocialGraph,
        wom: WOMOutput,
    ) -> List[WOMReception]:
        if wom.target_audience.value == "nobody":
            return []

        receptions: List[WOMReception] = []

        for target_id, rel in source_graph.relationships.items():
            # audience filter
            if (
                wom.target_audience.value == "close_friends"
                and rel.trust < self.close_threshold
            ):
                continue

            # stochastic reception
            prob = rel.trust * wom.emotional_intensity * wom.share_probability
            if self.rng.random() > prob:
                continue

            receptions.append(WOMReception(
                target_id=target_id,
                trust=rel.trust,
                reception_strength=round(prob, 4),
            ))

        return receptions

    def describe(self) -> str:
        return f"TrustWeightedWOM(close_threshold={self.close_threshold})"
