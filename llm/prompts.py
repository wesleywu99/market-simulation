"""
Prompt loader — Jinja2-backed templates stored on disk.

Prompts live in ``prompts/<name>.md`` as Markdown with Jinja2
placeholders.  Use ``render_prompt(name, **context)`` to render a
template.  Missing placeholders raise ``UndefinedError`` at render
time — no silent empty strings.

Convention
----------
For an LLM call that needs both a system and a user prompt, author two
files named ``<task>.system.md`` and ``<task>.user.md``.  The loader
has no special knowledge of this split; it just renders whatever name
you pass.

Custom filters
--------------
- ``money``        — ``1234.5`` → ``"¥1,235"``
- ``thousands``    — ``1234.5`` → ``"1,235"``
- ``pct``          — ``0.42``   → ``"42%"``
- ``pct1``         — ``0.425``  → ``"42.5%"``
- ``one_dp``       — ``1.234``  → ``"1.2"``
- ``two_dp``       — ``1.234``  → ``"1.23"``
- ``signed_two_dp``— ``0.1``    → ``"+0.10"``

See ``docs/30_prompt_externalization.md`` for the full design rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # prompts are plain text, never HTML
)

# ── custom filters ────────────────────────────────────────────

_env.filters["money"]         = lambda v: f"¥{v:,.0f}"
_env.filters["thousands"]     = lambda v: f"{v:,.0f}"
_env.filters["pct"]           = lambda v: f"{v:.0%}"
_env.filters["pct1"]          = lambda v: f"{v:.1%}"
_env.filters["one_dp"]        = lambda v: f"{v:.1f}"
_env.filters["two_dp"]        = lambda v: f"{v:.2f}"
_env.filters["signed_two_dp"] = lambda v: f"{v:+.2f}"


def render_prompt(template_name: str, /, **context: Any) -> str:
    """Render ``prompts/<template_name>.md`` with the supplied context.

    The template name is positional-only so that ``name`` (a common
    field in agent prompts) can be passed in ``context`` without
    colliding with this function's own parameter.

    Raises
    ------
    jinja2.TemplateNotFound
        If no file matches ``template_name``.
    jinja2.UndefinedError
        If the template references a variable not present in context.
    """
    template = _env.get_template(f"{template_name}.md")
    return template.render(**context)
