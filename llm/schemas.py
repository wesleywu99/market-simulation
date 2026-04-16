"""
Pydantic schemas for all LLM output contracts.

Every LLM call in this system must return JSON that validates against
one of these schemas.  This is the hard boundary between the LLM and
the rest of the simulation — change these carefully.
"""

import re
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, validator


# ─────────────────────────────────────────────────────────────
# SHARED ENUMS
# ─────────────────────────────────────────────────────────────

class PurchaseDecision(str, Enum):
    BUY = "buy"
    DEFER = "defer"
    REJECT = "reject"


class WOMSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class WOMTargetAudience(str, Enum):
    CLOSE_FRIENDS = "close_friends"
    GENERAL_NETWORK = "general_network"
    NOBODY = "nobody"


class CompetitiveResponseType(str, Enum):
    PRICE_ADJUSTMENT = "price_adjustment"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    MARKETING_CAMPAIGN = "marketing_campaign"
    NEW_PRODUCT = "new_product"
    WAIT = "wait"


class Urgency(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─────────────────────────────────────────────────────────────
# PURCHASE DECISION OUTPUT
# ─────────────────────────────────────────────────────────────

class PerceivedAttributes(BaseModel):
    """Rogers' 5 perceived innovation attributes — core adoption drivers."""
    relative_advantage: float = Field(..., ge=0.0, le=1.0,
        description="Perceived advantage over current alternatives")
    compatibility: float = Field(..., ge=0.0, le=1.0,
        description="Fit with the agent's values, habits, and needs")
    complexity: float = Field(..., ge=0.0, le=1.0,
        description="Perceived difficulty to learn and use (higher = harder)")
    trialability: float = Field(..., ge=0.0, le=1.0,
        description="Ease of low-cost trial before full commitment")
    observability: float = Field(..., ge=0.0, le=1.0,
        description="How visible the product's benefits are to the agent's social network")


class PurchaseDecisionOutput(BaseModel):
    """
    Output schema for ConsumerAgent purchase decision calls.
    The LLM must return JSON matching this schema exactly.
    """
    decision: PurchaseDecision
    confidence: float = Field(..., ge=0.0, le=1.0,
        description="How certain the agent is about this decision")
    reasoning: str = Field(...,
        description="Step-by-step reasoning that led to this decision")
    perceived_attributes: PerceivedAttributes
    price_acceptable: bool = Field(...,
        description="Whether the product price falls within the agent's budget tolerance")
    key_concerns: List[str] = Field(default_factory=list,
        description="Main objections or worries about the product")
    social_influence_weight: float = Field(..., ge=0.0, le=1.0,
        description="Fraction of the decision driven by social proof vs personal evaluation")
    deferred_until: Optional[int] = Field(None,
        description="If decision=defer, the simulation step to revisit this decision")

    @validator("deferred_until", pre=True, always=True)
    def coerce_deferred_until(cls, v: Any) -> Optional[int]:
        """Accept int, None, or strings like 'step 4' / '4'."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        # extract first number from strings like "step 4" or "4"
        match = re.search(r"\d+", str(v))
        return int(match.group()) if match else None


# ─────────────────────────────────────────────────────────────
# WORD-OF-MOUTH OUTPUT
# ─────────────────────────────────────────────────────────────

class WOMOutput(BaseModel):
    """
    Output schema for post-purchase WOM generation.
    Determines whether and how the agent shares their experience.
    """
    sentiment: WOMSentiment
    message_content: str = Field(...,
        description="What the agent would actually say to peers — authentic voice")
    share_probability: float = Field(..., ge=0.0, le=1.0,
        description="Likelihood the agent proactively shares without being asked")
    target_audience: WOMTargetAudience
    emotional_intensity: float = Field(..., ge=0.0, le=1.0,
        description="How emotionally charged the sharing is; higher = more persuasive")
    reasoning: str = Field(...,
        description="Why the agent chose to share or not share")


# ─────────────────────────────────────────────────────────────
# BELIEF UPDATE OUTPUT  (Layer 2 of the two-layer WOM pipeline)
# ─────────────────────────────────────────────────────────────

class BeliefUpdateOutput(BaseModel):
    """Output schema for semantic WOM-driven belief updates.

    Produced by ``LLMBeliefUpdater`` — one invocation per successful
    WOM reception.  Deltas are applied to the target agent's
    ``BeliefSystem`` and memory store by the updater.
    """
    sentiment_delta: float = Field(..., ge=-1.0, le=1.0,
        description="Signed shift to target's overall product sentiment")
    intent_delta: float = Field(..., ge=-1.0, le=1.0,
        description="Signed shift to target's purchase intent (0–1 bounded downstream)")
    new_belief_predicate: str = Field(...,
        description="Short claim the target now holds about the product")
    belief_confidence: float = Field(..., ge=0.0, le=1.0,
        description="Confidence attached to the new belief")
    memory_importance: float = Field(..., ge=0.0, le=1.0,
        description="Importance score for the stored WOM memory")
    memory_valence: float = Field(..., ge=-1.0, le=1.0,
        description="Emotional valence attached to the stored memory")
    reasoning: str = Field(...,
        description="One-sentence rationale for the update")


# ─────────────────────────────────────────────────────────────
# COMPETITIVE RESPONSE OUTPUT
# ─────────────────────────────────────────────────────────────

class CompetitiveResponseOutput(BaseModel):
    """Output schema for CompetitorAgent strategic response decisions."""
    response_type: CompetitiveResponseType
    specific_action: str = Field(...,
        description="Concrete action description, e.g. 'reduce price by 10%'")
    expected_impact: str = Field(...,
        description="Predicted market effect of this action")
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: Urgency
    estimated_cost: Optional[float] = Field(None,
        description="Estimated resource cost of the action")
    reasoning: str = Field(...,
        description="Strategic rationale for this response")
