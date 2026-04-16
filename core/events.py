from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    # ── Market events ──────────────────────────────
    PRODUCT_LAUNCH = "product_launch"
    PRICE_CHANGE = "price_change"
    COMPETITOR_ENTRY = "competitor_entry"
    COMPETITOR_EXIT = "competitor_exit"

    # ── Consumer events ────────────────────────────
    PURCHASE_INTENT = "purchase_intent"
    PURCHASE_DECISION = "purchase_decision"
    POST_PURCHASE = "post_purchase"
    WORD_OF_MOUTH = "word_of_mouth"
    ADVERTISING_EXPOSURE = "advertising_exposure"

    # ── External / macro events ────────────────────
    REGULATORY_CHANGE = "regulatory_change"
    ECONOMIC_SHOCK = "economic_shock"
    SOCIAL_TREND = "social_trend"
    MEDIA_COVERAGE = "media_coverage"

    # ── System events (internal bookkeeping) ───────
    STATE_CHANGE = "state_change"
    MEMORY_UPDATE = "memory_update"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class Event:
    """
    Core event unit.  All simulation state changes flow through events.

    content dict conventions by event_type:
      PURCHASE_DECISION  -> {"product_id": str, "decision": "buy|defer|reject",
                              "price_paid": float, "reasoning": str}
      POST_PURCHASE      -> {"product_id": str, "quality_rating": float,
                              "expected_price": float, "price_paid": float}
      WORD_OF_MOUTH      -> {"product_id": str, "sentiment": "positive|negative|neutral",
                              "message": str, "reach_depth": int}
      PRICE_CHANGE       -> {"product_id": str, "old_price": float, "new_price": float,
                              "change_reason": str}
      ADVERTISING_EXPOSURE -> {"product_id": str, "channel": str, "intensity": float}
      STATE_CHANGE       -> {"change_type": str, **kwargs}
      MEMORY_UPDATE      -> {"memory_type": str, "content": str,
                              "emotional_valence": float, "importance": float}
    """
    event_type: EventType
    timestamp: int                   # simulation step when event occurs
    source: str                      # agent_id or "system"
    content: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target: Optional[str] = None     # agent_id; None = broadcast to all
    priority: int = 0                # higher value = processed first
    processed: bool = False
    propagated: bool = False
