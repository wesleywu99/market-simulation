# Market Simulation Wiki

This directory is the canonical, versioned reference for the multi-agent market-forecasting system. Every time a design is **finalized** (implemented or agreed-upon), it is written up here so that new contributors and future-us can understand what exists, what was tried, and why.

## How this wiki is organized

| Prefix | Meaning | Example |
|---|---|---|
| `00_` | Vision, scope, glossary | `00_overview.md` |
| `10_` | Architecture snapshots (per phase) | `10_phase1_mvp.md` |
| `20_` | Upgrade plans (before implementation) | `20_phase1.5_upgrade_plan.md` |
| `30_` | Subsystem deep-dives (one concept each) | `30_wom_engine.md` |
| `40_` | Seed cases (real product configurations) | `40_seed_case_womens_apparel.md` |
| `90_` | Retrospectives, lessons learned | `90_decisions_log.md` |

## Versioning rule

Every architecture or plan doc carries a `## Version` line at the top. When a design **changes**, we either:

1. **Edit in place** and bump the version — use this when the change is a refinement.
2. **Supersede** — add a `## Status: Superseded by [link]` line at top and create a new file. Use this when the design pivots.

This means the wiki is append-only for major decisions but editable for refinements, and a reader can always trace backwards.

## Current status (as of 2026-04-16)

- **Phase 1 MVP**: ✅ Built and end-to-end tested. See [10_phase1_mvp.md](10_phase1_mvp.md).
- **Phase 1.5 upgrades**: ✅ All 13 items complete. See [20_phase1.5_upgrade_plan.md](20_phase1.5_upgrade_plan.md).
- **Phase 2.0 agent resilience**: ✅ 5 subsystems (decay, time, cognitive diversity, memory bounds, events). See [30_phase2_agent_resilience.md](30_phase2_agent_resilience.md).
- **Phase 3.0 forecast grounding**: ✅ 4 subsystems (category context, population spec, parameter sweep, validation). See [30_phase3_forecast_grounding.md](30_phase3_forecast_grounding.md).
- **First seed case**: 🎯 Affordable women's apparel. See [40_seed_case_womens_apparel.md](40_seed_case_womens_apparel.md) (draft).

## Index

- [00_overview.md](00_overview.md) — What this project is, the core mental model
- [10_phase1_mvp.md](10_phase1_mvp.md) — Phase 1 MVP architecture (baseline, working)
- [20_phase1.5_upgrade_plan.md](20_phase1.5_upgrade_plan.md) — Agreed upgrade roadmap
- [30_decision_engine.md](30_decision_engine.md) — System 1 / System 2 decision flow
- [30_product_quality.md](30_product_quality.md) — Semantic blind-box quality model
- [30_influencer_seeding.md](30_influencer_seeding.md) — KOL amplification strategy
- [30_two_layer_wom.md](30_two_layer_wom.md) — Independent Cascade + semantic belief update
- [30_prompt_externalization.md](30_prompt_externalization.md) — Jinja2-backed prompt files
- [30_monte_carlo.md](30_monte_carlo.md) — Seed-sweep wrapper with 95% CI bands
- [30_phase2_agent_resilience.md](30_phase2_agent_resilience.md) — Phase 2.0: decay, time semantics, cognitive diversity, memory bounds, event queue
- [30_phase3_forecast_grounding.md](30_phase3_forecast_grounding.md) — Phase 3.0: category context, population profiles, parameter sweep, validation suite
- [40_seed_case_womens_apparel.md](40_seed_case_womens_apparel.md) — First real product case (draft)
