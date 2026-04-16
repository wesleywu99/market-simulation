"""
PopulationSpec -- configurable agent demographics per seed case.

The default agent factory in ``simulation/controller.py`` uses hardcoded
Chinese-city demographics, tech-oriented occupations, and fixed income
ranges.  This works for the drone smoke test but is wrong for, say,
women's apparel targeting young urbanites in tier-1 cities.

``PopulationSpec`` lets each seed case declare its own demographic
distributions without modifying the controller.  Pass it via
``SimulationConfig.population_spec``; the runner feeds it into
``make_agent()`` at agent-creation time.

Fields left as ``None`` fall back to the controller's built-in defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PopulationSpec:
    """Demographic configuration for a simulation's agent population.

    Every field is optional.  ``None`` means "use the built-in default
    from controller.py."  This lets you override only what matters for
    a specific seed case.

    Example -- women's apparel in tier-1 Chinese cities::

        PopulationSpec(
            age_range=(18, 40),
            gender_distribution={"female": 0.85, "male": 0.15},
            income_ranges={
                "innovator":      (12000, 25000),
                "early_adopter":  (8000,  18000),
                "early_majority": (5000,  12000),
                "late_majority":  (3000,   8000),
                "laggard":        (2000,   5000),
            },
            occupations=[
                "Fashion Designer", "Marketing Manager", "Teacher",
                "Office Worker", "Nurse", "Freelancer", "Student",
                "Social Media Manager", "Retail Manager", "Accountant",
            ],
            locations=["Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu"],
            lifestyle_tags=[
                "fashion-forward", "brand-conscious", "social-media-active",
                "budget-savvy", "quality-focused", "trend-follower",
                "minimalist", "eco-conscious",
            ],
            education_weights={"bachelor": 0.4, "master": 0.25, "high_school": 0.25, "phd": 0.1},
            goal_template="Looking for stylish, affordable clothing for daily wear",
            tier_distribution=[
                ("innovator",      0.05),
                ("early_adopter",  0.15),
                ("early_majority", 0.35),
                ("late_majority",  0.30),
                ("laggard",        0.15),
            ],
        )
    """

    # Age bounds (inclusive).
    age_range: Optional[Tuple[int, int]] = None

    # Gender distribution as {label: fraction}.  Fractions should sum to 1.0.
    # The label is injected into AgentProfile as a lifestyle tag
    # (e.g. "female", "male") so the LLM sees it.
    gender_distribution: Optional[Dict[str, float]] = None

    # Per-tier income ranges (monthly disposable, local currency).
    # Keys must be the 5 Rogers tiers.  Missing tiers fall back to default.
    income_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    # Occupation pool.  One is picked uniformly at random per agent.
    occupations: Optional[List[str]] = None

    # Location pool.  One is picked uniformly at random per agent.
    locations: Optional[List[str]] = None

    # Lifestyle tag pool.  2-4 tags are sampled per agent.
    lifestyle_tags: Optional[List[str]] = None

    # Education weights as {level: fraction}.  Levels are sampled by weight.
    # Default: {"bachelor": 0.4, "master": 0.25, "high_school": 0.25, "phd": 0.1}
    education_weights: Optional[Dict[str, float]] = None

    # Name pool.  If None, falls back to the built-in NAMES list.
    names: Optional[List[str]] = None

    # Goal description template.  ``{product_name}`` is replaced at creation time.
    # Default: "Looking for a consumer drone for photography or recreation"
    goal_template: Optional[str] = None

    # Override Rogers tier distribution.  List of (tier_name, fraction) pairs.
    # Fractions should sum to ~1.0.  None = standard Rogers (2.5/13.5/34/34/16).
    tier_distribution: Optional[List[Tuple[str, float]]] = None

    # Cognitive style distribution per tier.  Overrides TIER_COGNITIVE_STYLES.
    # Keys are tier names; values are lists (sampled uniformly).
    cognitive_style_overrides: Optional[Dict[str, List[str]]] = None
