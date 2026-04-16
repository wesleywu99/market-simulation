from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class MemoryType(Enum):
    EPISODIC = "episodic"       # specific events (e.g. "I bought product X")
    SEMANTIC = "semantic"       # abstract knowledge (e.g. "brand X is expensive")
    PROCEDURAL = "procedural"   # how-to knowledge (e.g. "how to compare drones")


class VulnerabilityType(Enum):
    FINANCIAL = "financial"
    RISK_AVERSE = "risk_averse"
    TIME_PRESSURE = "time_pressure"
    SOCIAL_PRESSURE = "social_pressure"


class ConflictStatus(Enum):
    UNRESOLVED = "unresolved"
    DELIBERATING = "deliberating"
    RESOLVED = "resolved"


class NetworkPosition(Enum):
    CENTRAL = "central"         # opinion leader, high reach
    PERIPHERAL = "peripheral"   # average member
    ISOLATED = "isolated"       # minimal social connections


# ─────────────────────────────────────────────────────────────
# 1. GOAL
# ─────────────────────────────────────────────────────────────

@dataclass
class Goal:
    goal_id: str
    description: str            # e.g. "Buy a cost-effective drone for travel"
    priority: float             # 0.0–1.0
    status: GoalStatus = GoalStatus.ACTIVE
    deadline: Optional[int] = None   # simulation step deadline; None = no deadline
    progress: float = 0.0            # 0.0–1.0 completion
    sub_goals: List["Goal"] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# 2. BELIEF
# ─────────────────────────────────────────────────────────────

@dataclass
class Belief:
    belief_id: str
    subject: str       # e.g. "DJI Mini 4 Pro"
    predicate: str     # e.g. "has reliable obstacle avoidance"
    confidence: float  # 0.0–1.0
    source: str        # "personal_experience" | "word_of_mouth" | "media" | "advertising"
    timestamp: int     # step when belief was formed or last updated
    evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)


@dataclass
class BeliefSystem:
    """Tracks product-related beliefs with deduplication.

    ``add_or_update`` replaces an existing belief with the same subject +
    source within a recent window, preventing unbounded growth from
    repeated WOM of the same type.
    """
    beliefs: List[Belief] = field(default_factory=list)
    overall_sentiment: float = 0.0   # -1.0 (very negative) to 1.0 (very positive)
    purchase_intent: float = 0.0     # 0.0–1.0
    max_beliefs: int = 20            # hard cap — evict oldest when exceeded

    def add_or_update(
        self,
        belief: Belief,
        dedup_window: int = 3,
    ) -> None:
        """Append a belief, deduplicating recent same-source entries.

        If a belief with the same ``subject`` and ``source`` already exists
        within ``dedup_window`` steps, it is *replaced* (updated) rather
        than appended.  This prevents 20+ near-identical "positive review
        from trusted contact" entries accumulating over a long run.
        """
        for i, existing in enumerate(self.beliefs):
            if (existing.subject == belief.subject
                    and existing.source == belief.source
                    and belief.timestamp - existing.timestamp <= dedup_window):
                self.beliefs[i] = belief
                return
        self.beliefs.append(belief)
        # Evict oldest if over capacity.
        if len(self.beliefs) > self.max_beliefs:
            self.beliefs.pop(0)


# ─────────────────────────────────────────────────────────────
# 3. RESOURCE
# ─────────────────────────────────────────────────────────────

@dataclass
class ResourcePool:
    budget: float                          # monthly disposable income
    time_budget: int = 100                 # attention steps available this period
    attention: float = 1.0                 # 0.0–1.0; depleted by info processing
    social_capital: float = 0.5            # 0.0–1.0; affects WOM reach
    spent_budget: float = 0.0
    consumed_attention: Dict[str, float] = field(default_factory=dict)  # topic -> attention spent


# ─────────────────────────────────────────────────────────────
# 4. RELATIONSHIP
# ─────────────────────────────────────────────────────────────

@dataclass
class Relationship:
    target_id: str
    trust: float        # 0.0–1.0
    influence: float    # -1.0 to 1.0 (negative = contrarian effect)
    dependency: float   # 0.0–1.0; how much this agent relies on the other
    contact_frequency: int = 1      # avg interactions per N steps
    last_interaction: int = 0       # step of last contact
    shared_experiences: List[str] = field(default_factory=list)


@dataclass
class SocialGraph:
    relationships: Dict[str, Relationship] = field(default_factory=dict)  # agent_id -> Relationship
    network_position: NetworkPosition = NetworkPosition.PERIPHERAL

    def get_neighbors(self) -> List[str]:
        return list(self.relationships.keys())

    def get_adopted_neighbors(self, adopted_ids: set) -> List[str]:
        return [aid for aid in self.relationships if aid in adopted_ids]

    def get_trusted_neighbors(self, threshold: float = 0.6) -> List[str]:
        return [aid for aid, rel in self.relationships.items() if rel.trust >= threshold]


# ─────────────────────────────────────────────────────────────
# 5. MEMORY
# ─────────────────────────────────────────────────────────────

@dataclass
class Memory:
    memory_id: str
    content: str
    memory_type: MemoryType
    importance: float           # 0.0–1.0; affects retrieval weight
    timestamp: int
    emotional_valence: float    # -1.0 (negative) to 1.0 (positive)
    context: str = ""           # situational background
    consequences: str = ""      # what resulted from this event
    lessons: str = ""           # abstracted takeaway
    embedding: Optional[List[float]] = None  # populated when Qdrant is integrated


@dataclass
class MemoryStore:
    """Bounded memory with importance-weighted retrieval.

    When the total number of memories exceeds ``max_memories``, the
    lowest-importance entry is evicted on ``add()``.  ``retrieve_recent``
    ranks by *effective importance* (importance × decay^age) so that
    fresh important memories outrank stale trivial ones.
    """
    episodic: List[Memory] = field(default_factory=list)
    semantic: List[Memory] = field(default_factory=list)
    procedural: List[Memory] = field(default_factory=list)
    recent_access: List[str] = field(default_factory=list)  # memory_ids
    max_memories: int = 30  # cap across all types

    def add(self, memory: Memory) -> None:
        bucket = {
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.PROCEDURAL: self.procedural,
        }[memory.memory_type]
        bucket.append(memory)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop the lowest-importance memory when over capacity."""
        all_mem = self.all_memories()
        if len(all_mem) <= self.max_memories:
            return
        # Find the least important memory and remove it from its bucket.
        weakest = min(all_mem, key=lambda m: m.importance)
        for bucket in (self.episodic, self.semantic, self.procedural):
            if weakest in bucket:
                bucket.remove(weakest)
                break

    def all_memories(self) -> List[Memory]:
        return self.episodic + self.semantic + self.procedural

    def retrieve_recent(
        self,
        n: int = 5,
        current_step: int = 0,
        decay_factor: float = 0.85,
    ) -> List[Memory]:
        """Return top-n memories ranked by effective importance.

        Effective importance = importance × decay_factor^(current_step - timestamp).
        When ``current_step`` is 0 (unknown), falls back to pure timestamp order.
        """
        memories = self.all_memories()
        if not memories:
            return []
        if current_step > 0:
            def _effective(m: Memory) -> float:
                age = max(0, current_step - m.timestamp)
                return m.importance * (decay_factor ** age)
            return sorted(memories, key=_effective, reverse=True)[:n]
        return sorted(memories, key=lambda m: m.timestamp, reverse=True)[:n]

    def retrieve_by_importance(self, top_k: int = 5) -> List[Memory]:
        return sorted(self.all_memories(), key=lambda m: m.importance, reverse=True)[:top_k]

    def retrieve_by_keyword(self, keyword: str, top_k: int = 5) -> List[Memory]:
        keyword = keyword.lower()
        matches = [m for m in self.all_memories() if keyword in m.content.lower()]
        return sorted(matches, key=lambda m: m.importance, reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────
# 6. VULNERABILITY
# ─────────────────────────────────────────────────────────────

@dataclass
class Vulnerability:
    vulnerability_id: str
    type: VulnerabilityType
    severity: float             # 0.0–1.0
    trigger_conditions: List[str] = field(default_factory=list)
    coping_strategy: Optional[str] = None
    active: bool = False


@dataclass
class VulnerabilitySet:
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    risk_tolerance: float = 0.5    # 0.0–1.0
    loss_aversion: float = 2.5     # Kahneman & Tversky; typical human value ~2.5

    def activate(self, trigger: str) -> None:
        for v in self.vulnerabilities:
            if trigger in v.trigger_conditions:
                v.active = True

    def get_active(self) -> List[Vulnerability]:
        return [v for v in self.vulnerabilities if v.active]


# ─────────────────────────────────────────────────────────────
# 7. INTERNAL CONFLICT
# ─────────────────────────────────────────────────────────────

@dataclass
class InternalConflict:
    conflict_id: str
    description: str             # e.g. "wants high quality but budget is tight"
    conflicting_goals: List[str] = field(default_factory=list)  # goal_ids
    resolution_status: ConflictStatus = ConflictStatus.UNRESOLVED
    resolution_strategy: Optional[str] = None
    tension_level: float = 0.5   # 0.0–1.0; higher = slower/less rational decision
    historical_resolutions: List[str] = field(default_factory=list)


@dataclass
class ConflictSet:
    conflicts: List[InternalConflict] = field(default_factory=list)
    dominant_conflict: Optional[str] = None  # conflict_id of the most pressing conflict

    def get_unresolved(self) -> List[InternalConflict]:
        return [c for c in self.conflicts if c.resolution_status != ConflictStatus.RESOLVED]


# ─────────────────────────────────────────────────────────────
# STATIC PROFILE  (immutable agent identity)
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    agent_id: str
    name: str
    age: int
    occupation: str
    income_level: str       # "low" | "medium" | "high" | "affluent"
    income_amount: float    # monthly disposable income in local currency
    education: str          # "high_school" | "bachelor" | "master" | "phd"
    location: str           # city or region
    adopter_tier: str       # "innovator" | "early_adopter" | "early_majority" | "late_majority" | "laggard"
    lifestyle_tags: List[str] = field(default_factory=list)  # e.g. ["tech-savvy", "outdoor", "family"]

    # Per-persona weighting of product quality dimensions.  When empty the
    # blind-box experience sampler falls back to uniform weights.
    # See docs/30_product_quality.md.
    dimension_weights: Dict[str, float] = field(default_factory=dict)

    # Cognitive style — shapes how the agent reasons about decisions.
    # Injected into the LLM system prompt to create genuine behavioral
    # diversity beyond adopter-tier defaults.
    # Values: "analytical" | "emotional" | "social" | "skeptical" | "balanced"
    cognitive_style: str = "balanced"

    # Set by the InfluencerSeeding strategy after network build.
    # See docs/30_influencer_seeding.md.
    is_influencer: bool = False


# ─────────────────────────────────────────────────────────────
# FULL AGENT STATE  (profile + 7 dynamic state variables)
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    profile: AgentProfile

    # 7 dynamic state variables — all persist across simulation steps
    goals: List[Goal] = field(default_factory=list)
    beliefs: BeliefSystem = field(default_factory=BeliefSystem)
    resources: ResourcePool = field(default_factory=lambda: ResourcePool(budget=0.0))
    relationships: SocialGraph = field(default_factory=SocialGraph)
    memories: MemoryStore = field(default_factory=MemoryStore)
    vulnerabilities: VulnerabilitySet = field(default_factory=VulnerabilitySet)
    conflicts: ConflictSet = field(default_factory=ConflictSet)

    # simulation tracking
    current_step: int = 0
    has_purchased: bool = False
    purchase_step: Optional[int] = None
