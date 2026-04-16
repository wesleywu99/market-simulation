"""
Agent factory -- creates consumer agents with configurable demographics.

Provides ``make_agent()`` and ``_assign_tiers()`` used by the runner to
populate the simulation.  Demographics can be overridden per seed case
via ``PopulationSpec``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, TYPE_CHECKING

from agents.consumer import ConsumerAgent
from core.state import AgentProfile, Goal, GoalStatus

if TYPE_CHECKING:
    from simulation.population import PopulationSpec


# ── Adopter tier distribution (Rogers 1962) ──────────────────
TIER_DISTRIBUTION = [
    ("innovator",      0.025),
    ("early_adopter",  0.135),
    ("early_majority", 0.340),
    ("late_majority",  0.340),
    ("laggard",        0.160),
]

# Income ranges by tier (monthly disposable, CNY)
TIER_INCOME = {
    "innovator":      (15000, 30000),
    "early_adopter":  (8000,  18000),
    "early_majority": (5000,  10000),
    "late_majority":  (3000,   7000),
    "laggard":        (2000,   5000),
}

OCCUPATIONS = [
    "Software Engineer", "Product Manager", "Designer", "Teacher",
    "Doctor", "Salesperson", "Entrepreneur", "Accountant",
    "Student", "Freelancer",
]

# Cognitive style distribution per adopter tier.  Innovators skew
# analytical/balanced; laggards skew skeptical/social.  Each agent
# is randomly assigned from the tier's weighted pool.
TIER_COGNITIVE_STYLES = {
    "innovator":      ["analytical", "analytical", "balanced", "emotional"],
    "early_adopter":  ["analytical", "balanced", "emotional", "social"],
    "early_majority": ["balanced", "social", "emotional", "analytical", "skeptical"],
    "late_majority":  ["social", "skeptical", "balanced", "skeptical"],
    "laggard":        ["skeptical", "skeptical", "social", "balanced"],
}

# Per-tier LLM temperature — innovators explore, laggards are rigid.
TIER_TEMPERATURE = {
    "innovator":      0.9,
    "early_adopter":  0.7,
    "early_majority": 0.5,
    "late_majority":  0.3,
    "laggard":        0.2,
}

NAMES = [
    "Wei Chen", "Lin Zhang", "Jing Liu", "Fang Wang", "Ming Li",
    "Xiao Zhao", "Hui Yang", "Rui Xu", "Jun Wu", "Ying Huang",
    "Bo Sun", "Na Ma", "Kai Zhu", "Tao He", "Yu Guo",
]


def _assign_tiers(
    n: int,
    spec: "Optional[PopulationSpec]" = None,
) -> List[str]:
    """Assign adopter tiers to N agents following Rogers distribution.

    If *spec* provides a ``tier_distribution``, use it instead of the
    default Rogers 2.5/13.5/34/34/16 split.
    """
    dist = TIER_DISTRIBUTION
    if spec and spec.tier_distribution:
        dist = spec.tier_distribution

    tiers = []
    for tier, fraction in dist:
        count = max(1, round(n * fraction))
        tiers.extend([tier] * count)
    # trim or pad to exactly n
    tiers = tiers[:n]
    while len(tiers) < n:
        tiers.append("early_majority")
    random.shuffle(tiers)
    return tiers


def _weighted_choice(weights: Dict[str, float]) -> str:
    """Pick a key from {key: weight} dict, weighted by value."""
    keys = list(weights.keys())
    vals = list(weights.values())
    return random.choices(keys, weights=vals, k=1)[0]


def _pick_gender(spec: "Optional[PopulationSpec]") -> Optional[str]:
    """Return a gender label based on PopulationSpec, or None if unspecified."""
    if not spec or not spec.gender_distribution:
        return None
    return _weighted_choice(spec.gender_distribution)


def make_agent(
    agent_id: str,
    tier: str,
    idx: int,
    spec: "Optional[PopulationSpec]" = None,
) -> ConsumerAgent:
    """Create a single ConsumerAgent with demographics from *spec* (or defaults)."""
    # Income
    if spec and spec.income_ranges and tier in spec.income_ranges:
        income_min, income_max = spec.income_ranges[tier]
    else:
        income_min, income_max = TIER_INCOME[tier]
    income = random.uniform(income_min, income_max)

    # Age
    if spec and spec.age_range:
        age = random.randint(spec.age_range[0], spec.age_range[1])
    else:
        age = random.randint(22, 55)

    # Name pool
    names = (spec.names if spec and spec.names else NAMES)
    name = names[idx % len(names)]

    # Occupation
    occupations = (spec.occupations if spec and spec.occupations else OCCUPATIONS)
    occupation = random.choice(occupations)

    # Location
    default_locations = ["Shanghai", "Beijing", "Shenzhen", "Hangzhou", "Chengdu"]
    locations = (spec.locations if spec and spec.locations else default_locations)
    location = random.choice(locations)

    # Education
    if spec and spec.education_weights:
        education = _weighted_choice(spec.education_weights)
    else:
        education = random.choice(["bachelor", "master", "bachelor", "high_school"])

    # Lifestyle tags
    default_tags = ["tech-savvy", "travel", "photography", "outdoor",
                    "family", "budget-conscious", "brand-loyal"]
    tag_pool = (spec.lifestyle_tags if spec and spec.lifestyle_tags else default_tags)
    k = min(random.randint(2, 4), len(tag_pool))
    lifestyle_tags = random.sample(tag_pool, k=k)

    # Gender (injected as a lifestyle tag so the LLM sees it)
    gender = _pick_gender(spec)
    if gender:
        lifestyle_tags = [gender] + lifestyle_tags

    # Cognitive style
    if spec and spec.cognitive_style_overrides and tier in spec.cognitive_style_overrides:
        cog_pool = spec.cognitive_style_overrides[tier]
    else:
        cog_pool = TIER_COGNITIVE_STYLES.get(tier, ["balanced"])
    cognitive_style = random.choice(cog_pool)

    profile = AgentProfile(
        agent_id=agent_id,
        name=name,
        age=age,
        occupation=occupation,
        income_level="high" if income > 12000 else ("medium" if income > 6000 else "low"),
        income_amount=round(income, 0),
        education=education,
        location=location,
        adopter_tier=tier,
        cognitive_style=cognitive_style,
        lifestyle_tags=lifestyle_tags,
    )
    agent = ConsumerAgent(profile)

    # Goal
    default_goal = "Looking for a consumer drone for photography or recreation"
    goal_desc = (spec.goal_template if spec and spec.goal_template else default_goal)
    agent.state.goals.append(Goal(
        goal_id=f"g_{agent_id}",
        description=goal_desc,
        priority=random.uniform(0.5, 0.95),
        status=GoalStatus.ACTIVE,
    ))
    return agent
