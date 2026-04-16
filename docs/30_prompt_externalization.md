# 30 — Prompt Externalization

## Version
v1.0 (2026-04-15) — First spec and implementation. Phase 1.5 upgrade #5.

## Status
✅ Implemented.

## The problem

Phase 1 stored every prompt as an f-string literal inside the Python
file that called the LLM:

```python
PURCHASE_USER_PROMPT = """Profile: {name}, {age}yo {occupation}, ...
{{ "decision":"buy|defer|reject", ... }}"""
```

Two acute pains came out of this:

1. **Iterating a prompt requires editing Python.** Every comma, every
   pluralisation tweak, every "be more concise" instruction is a code
   change. People who think well about prompts (PMs, domain experts)
   are not always the people who are comfortable opening a `.py`.
2. **JSON-in-f-string is a minefield.** Literal braces have to be
   escaped as `{{` / `}}`. Easy to get wrong, painful to read, hides
   real format placeholders behind visual noise.
3. **No diff readability.** A prompt diff in a Python file is buried
   inside the file's surrounding logic; reviewers can't see the
   prompt change in isolation.

## The model

```
prompts/                            ← canonical, version-controlled
├── purchase_decision.system.md
├── purchase_decision.user.md
├── wom_generation.system.md
├── wom_generation.user.md
├── belief_update.system.md
└── belief_update.user.md

llm/prompts.py                      ← thin loader
   render_prompt(name, **context) -> str
```

### Loader contract

```python
from llm.prompts import render_prompt

system = render_prompt("purchase_decision.system")          # no context needed
user   = render_prompt("purchase_decision.user", **fields)  # all fields required
```

### Strict-undefined safety

The Jinja2 environment uses `StrictUndefined`. A template that
references `{{ foo }}` will raise `UndefinedError` at render time if
`foo` is not in the context. This catches "I renamed a field but
forgot to update the prompt" bugs immediately, instead of producing a
silently-empty `{{ }}` substitution that the LLM ignores.

### Custom filters

A small set of formatting filters keeps templates clean and avoids
sprinkling Python `.format()` calls through the template prose:

| Filter | Example | Output |
|---|---|---|
| `money` | `{{ 1234.5 \| money }}` | `¥1,235` |
| `thousands` | `{{ 1234.5 \| thousands }}` | `1,235` |
| `pct` | `{{ 0.42 \| pct }}` | `42%` |
| `pct1` | `{{ 0.425 \| pct1 }}` | `42.5%` |
| `one_dp` | `{{ 1.234 \| one_dp }}` | `1.2` |
| `two_dp` | `{{ 1.234 \| two_dp }}` | `1.23` |
| `signed_two_dp` | `{{ 0.1 \| signed_two_dp }}` | `+0.10` |

Filters live in `llm/prompts.py`. Add new ones there, not inline in
templates.

### File-naming convention

`<task>.<role>.md`:

- `<task>` matches the call site — `purchase_decision`, `wom_generation`, `belief_update`.
- `<role>` is `system` or `user`.
- Both halves of an LLM call live side-by-side in `prompts/` so a reader
  can see the full instruction at a glance.

The loader has no special knowledge of `system` vs `user` — it just
loads whatever name you pass. Two files per call is convention, not
machinery.

### JSON examples in templates

Single braces in a template are passed through verbatim by Jinja2.
That means the JSON output schema example reads cleanly:

```
{"decision":"buy|defer|reject","confidence":0.0-1.0, ...}
```

…instead of the f-string-escaped version:

```python
{{"decision":"buy|defer|reject","confidence":0.0-1.0, ...}}
```

This was the #1 source of typos in Phase 1.

## Migration

| Old (Phase 1) | New |
|---|---|
| `PURCHASE_SYSTEM_PROMPT` constant in `consumer.py` | `prompts/purchase_decision.system.md` |
| `PURCHASE_USER_PROMPT.format(**fields)` | `render_prompt("purchase_decision.user", **fields)` |
| `WOM_SYSTEM_PROMPT` / `WOM_USER_PROMPT` in `consumer.py` | `prompts/wom_generation.{system,user}.md` |
| `_BELIEF_UPDATE_*_PROMPT` in `belief.py` | `prompts/belief_update.{system,user}.md` |

Call sites switched from `STR.format(...)` to `render_prompt(name, ...)`.
The interface to the dispatcher is unchanged — both still pass plain
strings.

## Architecture — what's generic, what's per-prompt

| Concern | Generic | Per-prompt |
|---|---|---|
| Jinja2 environment + filters | ✅ (`llm/prompts.py`) | — |
| `render_prompt(name, **context)` | ✅ | — |
| Naming convention `<task>.<role>.md` | ✅ | — |
| Strict-undefined behaviour | ✅ | — |
| Prompt content | — | ✅ |
| Context dict at the call site | — | ✅ |

## Caveats and non-goals

- **No partials / inheritance yet.** Jinja2 supports `{% include %}`
  and `{% extends %}`. We don't use either today — every prompt is a
  standalone file. Add inheritance only when the duplication actually
  hurts.
- **No prompt versioning beyond git.** Designs that change behaviour
  meaningfully should bump a version field in the wiki entry that
  cites them, not in the prompt file itself.
- **No A/B harness yet.** A prompt-flag mechanism (e.g. swapping in a
  `purchase_decision.terse.user.md`) is straightforward to add later
  by passing the prompt name through `SimulationConfig`. Not built
  until we actually want to A/B.
- **No template tests.** `StrictUndefined` already catches the
  "missing field" failure mode at first render. Adding a unit test
  per template is overkill until prompts grow in number or complexity.
