"""
External event schedule — inject market shocks into the simulation.

The simulation loop was previously a closed system driven only by WOM.
Real markets face price changes, media campaigns, advertising pushes,
and macro shocks.  This module provides:

* ``ScheduledEvent`` — a (step, type, params) tuple.
* ``EventSchedule``  — an ordered list of scheduled events.
* ``apply_event()``  — mutates the product / agents when an event fires.

Usage: build an ``EventSchedule`` and pass it in ``SimulationConfig``.
The runner calls ``schedule.events_for_step(step)`` at the top of each
step and applies them before agent evaluation begins.

Supported event types
---------------------
* ``PRICE_CHANGE``           — modify ``Product.price`` (absolute or pct).
* ``MEDIA_COVERAGE``         — inject a memory into a random subset of agents
                               (models a news article or review going viral).
* ``ADVERTISING_EXPOSURE``   — boost ``purchase_intent`` for a random subset.

Adding new types: add a handler in ``apply_event()`` and document it here.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.events import EventType
from core.state import Memory, MemoryType

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent
    from environment.product import Product


# ─────────────────────────────────────────────────────────────
# SCHEDULED EVENT
# ─────────────────────────────────────────────────────────────

@dataclass
class ScheduledEvent:
    """One external event scheduled for a specific simulation step.

    Parameters
    ----------
    step : int
        The step at which the event fires.
    event_type : EventType
        Which kind of event (PRICE_CHANGE, MEDIA_COVERAGE, …).
    params : dict
        Type-specific payload.  See ``apply_event`` for schemas.
    description : str
        Human-readable label for trace logs.
    """
    step: int
    event_type: EventType
    params: Dict[str, Any]
    description: str = ""


# ─────────────────────────────────────────────────────────────
# EVENT SCHEDULE
# ─────────────────────────────────────────────────────────────

@dataclass
class EventSchedule:
    """Ordered list of external events to inject during a simulation run.

    Example
    -------
    >>> schedule = EventSchedule(events=[
    ...     ScheduledEvent(step=5, event_type=EventType.PRICE_CHANGE,
    ...                    params={"new_price": 3499.0},
    ...                    description="Early-bird discount ends"),
    ...     ScheduledEvent(step=10, event_type=EventType.MEDIA_COVERAGE,
    ...                    params={"sentiment": "positive",
    ...                            "message": "Tech blog rates it 9/10",
    ...                            "reach_fraction": 0.3},
    ...                    description="Major tech review published"),
    ... ])
    """
    events: List[ScheduledEvent] = field(default_factory=list)

    def events_for_step(self, step: int) -> List[ScheduledEvent]:
        """Return all events scheduled for *step*, in insertion order."""
        return [e for e in self.events if e.step == step]

    def describe(self) -> str:
        if not self.events:
            return "none"
        types = sorted({e.event_type.value for e in self.events})
        return f"{len(self.events)} events ({', '.join(types)})"


# ─────────────────────────────────────────────────────────────
# EVENT APPLICATION
# ─────────────────────────────────────────────────────────────

def apply_event(
    event: ScheduledEvent,
    product: "Product",
    agents: "List[ConsumerAgent]",
    rng: random.Random,
) -> str:
    """Apply a scheduled event, mutating product / agent state.

    Returns a one-line log message describing what happened.
    """
    t = event.event_type
    p = event.params

    if t == EventType.PRICE_CHANGE:
        return _apply_price_change(product, p)

    if t == EventType.MEDIA_COVERAGE:
        return _apply_media_coverage(product, agents, p, rng, event.step)

    if t == EventType.ADVERTISING_EXPOSURE:
        return _apply_advertising(agents, p, rng)

    return f"[event] Unhandled event type: {t.value}"


# ── handlers ─────────────────────────────────────────────────

def _apply_price_change(product: "Product", params: dict) -> str:
    """Change product price.

    Params:
        new_price (float)        — absolute new price.  OR
        pct_change (float)       — relative change, e.g. -0.10 = 10% discount.
        (exactly one must be provided)
    """
    old = product.price
    if "new_price" in params:
        product.price = params["new_price"]
    elif "pct_change" in params:
        product.price = round(old * (1 + params["pct_change"]), 2)
    else:
        return "[event] PRICE_CHANGE: missing new_price or pct_change"
    return f"[event] Price: {old:,.0f} -> {product.price:,.0f}"


def _apply_media_coverage(
    product: "Product",
    agents: "List[ConsumerAgent]",
    params: dict,
    rng: random.Random,
    step: int,
) -> str:
    """Inject a media-sourced memory into a fraction of agents.

    Params:
        sentiment (str)          — "positive" | "negative" | "neutral"
        message (str)            — the headline / review snippet
        reach_fraction (float)   — fraction of non-buyer agents who see it (0-1)
        importance (float)       — memory importance (default 0.6)
    """
    sentiment = params.get("sentiment", "neutral")
    message = params.get("message", "Media coverage about the product")
    reach = params.get("reach_fraction", 0.3)
    importance = params.get("importance", 0.6)

    eligible = [a for a in agents if not a.state.has_purchased]
    n_reached = max(1, int(len(eligible) * reach))
    targets = rng.sample(eligible, min(n_reached, len(eligible)))

    valence = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}.get(sentiment, 0.0)

    for agent in targets:
        agent.state.memories.add(Memory(
            memory_id=str(uuid.uuid4())[:8],
            content=f"Saw media coverage: {message[:120]}",
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            timestamp=step,
            emotional_valence=valence,
            context=f"Media coverage ({sentiment})",
        ))
        # Nudge sentiment
        delta = {"positive": 0.08, "neutral": 0.0, "negative": -0.15}.get(sentiment, 0.0)
        agent.state.beliefs.overall_sentiment = max(
            -1.0, min(1.0, agent.state.beliefs.overall_sentiment + delta)
        )

    return (
        f"[event] Media ({sentiment}): \"{message[:50]}...\" "
        f"-> reached {len(targets)}/{len(eligible)} agents"
    )


def _apply_advertising(
    agents: "List[ConsumerAgent]",
    params: dict,
    rng: random.Random,
) -> str:
    """Boost purchase intent for a fraction of agents.

    Params:
        reach_fraction (float)   — fraction of non-buyer agents exposed (0-1)
        intent_boost (float)     — additive boost to purchase_intent (default 0.08)
    """
    reach = params.get("reach_fraction", 0.4)
    boost = params.get("intent_boost", 0.08)

    eligible = [a for a in agents if not a.state.has_purchased]
    n_reached = max(1, int(len(eligible) * reach))
    targets = rng.sample(eligible, min(n_reached, len(eligible)))

    for agent in targets:
        agent.state.beliefs.purchase_intent = min(
            1.0, agent.state.beliefs.purchase_intent + boost
        )

    return (
        f"[event] Advertising: intent +{boost:.0%} "
        f"-> reached {len(targets)}/{len(eligible)} agents"
    )
