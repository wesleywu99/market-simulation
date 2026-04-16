"""
TraceWriter — JSONL decision/WOM logs + per-step state snapshots.

Writes three artefacts under ``traces/<run_id>/``:

* ``decisions.jsonl``  — one row per purchase decision (System 1 or System 2).
* ``wom.jsonl``        — one row per WOM reception attempt (success only,
  Layer 1 already filtered drops).
* ``snapshots/step_<N>.json`` — compact agent-state dump per step.

The writer is **opt-in** — pass ``trace_dir`` in ``SimulationConfig`` to
enable.  When disabled, the runner uses a ``NoOpTrace`` that swallows
every call so call-sites stay branch-free.

Design choices
--------------
* JSONL (one JSON object per line) is grep-able, append-safe, and
  Pandas-loadable via ``read_json(lines=True)``.
* Per-step snapshots are full JSON files because they need to be
  loaded as a unit, not streamed.
* The writer keeps file handles open across the run and closes them
  via ``close()`` / ``__exit__`` — the runner uses it as a context
  manager.
* Agent snapshots capture **summary** fields, not full state — beliefs
  and memories collapse to counts + aggregate stats.  This keeps a
  100×50 run under ~5 MB.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# RUN ID HELPERS
# ─────────────────────────────────────────────────────────────

def make_run_id(prefix: str = "run") -> str:
    """Timestamp-based run id, e.g. ``run_20260415_143025``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


# ─────────────────────────────────────────────────────────────
# NO-OP IMPL  (default when tracing is disabled)
# ─────────────────────────────────────────────────────────────

class NoOpTrace:
    """Trace sink that drops every call.  Used when tracing is disabled."""

    enabled = False
    run_dir: Optional[Path] = None

    def write_decision(self, **fields: Any) -> None: ...
    def write_wom(self, **fields: Any) -> None: ...
    def write_snapshot(self, step: int, agents: "List[ConsumerAgent]") -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_exc): self.close()


# ─────────────────────────────────────────────────────────────
# TRACE WRITER
# ─────────────────────────────────────────────────────────────

class TraceWriter:
    """Append-mode JSONL writer with a per-step snapshot dir.

    Parameters
    ----------
    base_dir : str | Path
        Parent directory for run subdirs (typically ``traces/``).
    run_id : str | None
        Run identifier; auto-generated via ``make_run_id()`` if omitted.
    """

    enabled = True

    def __init__(
        self,
        base_dir: str | Path = "traces",
        run_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id or make_run_id()
        self.run_dir = Path(base_dir) / self.run_id
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._decisions_fh: IO[str] = (self.run_dir / "decisions.jsonl").open(
            "a", encoding="utf-8"
        )
        self._wom_fh: IO[str] = (self.run_dir / "wom.jsonl").open(
            "a", encoding="utf-8"
        )

    # ── lifecycle ───────────────────────────────────────────

    def close(self) -> None:
        for fh in (self._decisions_fh, self._wom_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── writers ─────────────────────────────────────────────

    def write_decision(self, **fields: Any) -> None:
        """Append a purchase-decision row."""
        self._write(self._decisions_fh, fields)

    def write_wom(self, **fields: Any) -> None:
        """Append a WOM-reception row."""
        self._write(self._wom_fh, fields)

    def write_snapshot(self, step: int, agents: "List[ConsumerAgent]") -> None:
        """Dump compact per-agent state for ``step`` to its own JSON file."""
        path = self.snapshots_dir / f"step_{step:03d}.json"
        payload = {
            "step": step,
            "n_agents": len(agents),
            "agents": [_compact_agent(a) for a in agents],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=None)

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _write(fh: IO[str], obj: dict) -> None:
        json.dump(obj, fh, ensure_ascii=False, default=_json_safe)
        fh.write("\n")
        fh.flush()  # flush per row so a crash mid-run still preserves history


# ─────────────────────────────────────────────────────────────
# AGENT STATE COMPACTOR
# ─────────────────────────────────────────────────────────────

def _compact_agent(agent: "ConsumerAgent") -> dict:
    """Reduce agent state to the fields useful for post-hoc analysis."""
    s = agent.state
    p = s.profile
    memories = s.memories.all_memories()

    avg_valence = (
        sum(m.emotional_valence for m in memories) / len(memories)
        if memories else 0.0
    )

    return {
        "agent_id": agent.agent_id,
        "name": p.name,
        "tier": p.adopter_tier,
        "is_influencer": p.is_influencer,
        "has_purchased": s.has_purchased,
        "purchase_step": s.purchase_step,
        "budget": round(s.resources.budget, 2),
        "spent_budget": round(s.resources.spent_budget, 2),
        "social_capital": round(s.resources.social_capital, 3),
        "purchase_intent": round(s.beliefs.purchase_intent, 3),
        "overall_sentiment": round(s.beliefs.overall_sentiment, 3),
        "n_beliefs": len(s.beliefs.beliefs),
        "n_memories": len(memories),
        "avg_memory_valence": round(avg_valence, 3),
        "n_neighbors": len(s.relationships.relationships),
    }


def _json_safe(obj: Any) -> Any:
    """Fallback serializer for enums / dataclasses we don't want to fully expand."""
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "__dict__"):
        return str(obj)
    return repr(obj)
