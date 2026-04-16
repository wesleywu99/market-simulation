# 00 — Project Overview

## Version
v1.0 (2026-04-15) — First pass after Phase 1 MVP completion.

## What this project is

A **multi-agent market forecasting system** that predicts product adoption for new products entering a new market. Each agent is an LLM-powered simulated consumer; together they form a social network whose dynamics approximate real-world diffusion.

The goal is **qualitative forecasting**: instead of regressing on historical data (which doesn't exist for new products in new markets), we simulate *why* people buy, *who* they tell, and *how* that spreads — producing adoption curves, tier breakdowns, and sentiment trajectories that a human analyst can inspect and challenge.

## Core mental model

```
   Product launched
         │
         ▼
   Phase 1 (launch step) — every agent sees the product, runs decision
         │
         ▼
   Some buy, some reject, some defer ──► deferred agents re-enter pool later
         │
         ▼
   Buyers become WOM sources ──► propagate through trust-weighted edges
         │
         ▼
   WOM recipients update beliefs ──► re-enter decision pool
         │
         ▼
   Repeat for N steps ──► adoption curve, sentiment curve, tier breakdown
```

Two things make this more than a spreadsheet model:

1. **Agents reason with an LLM.** Each decision ("should I buy?", "how do I describe this to my friend?") is a prompted LLM call that considers the agent's personality, budget, beliefs, social context, and Rogers' five perceived innovation attributes.
2. **Agents have state.** A 7-variable state vector (goals, beliefs, resources, relationships, memories, vulnerabilities, conflicts) persists across steps, so an agent who hears positive WOM from a trusted friend is *different* next step than one who didn't.

## Why LLM agents, not statistical agents?

Traditional ABM (agent-based modeling) uses hand-coded rules for how agents decide and spread information. This works when you have decades of data to calibrate the rules. For a brand-new product in a brand-new market, you don't.

LLM agents substitute **common-sense reasoning about a persona** for calibrated rules. Given a persona ("38-year-old office worker, practical, budget-conscious, distrusts new tech") and a product description, the LLM produces a plausible decision. We then bind this to structured state (Rogers tier, budget, trust graph) so the behavior is reproducible and measurable — not just vibes.

## Constraints that shape the design

1. **LLM cost dominates runtime and $$$.** Every design choice prioritizes reducing unnecessary LLM calls (System 1 pre-filter, code-layer IC gate, batching, caching).
2. **LLM latency dominates wall-clock time.** Async/parallel dispatch is not optional at scale.
3. **LLM outputs drift.** Every structured output passes through Pydantic validation; failures trigger retry.
4. **Reproducibility matters.** Seeded RNG for network generation and stochastic gates; deterministic prompts.

## Scope for MVP

- **Single product** per simulation run (no competitor modeling).
- **Low-frequency high-value durable goods** (e.g. AI wearables, women's apparel). Buyers become WOM sources; no repeat purchases modeled.
- **Global market** — no geography or physical distribution friction.
- **100 agents × 50 steps** target scale (Phase 1.5).

## Out of scope for MVP

- Competitor products / market share dynamics
- Supply/inventory constraints
- Seasonal effects, macro trends
- Retargeting, advertising, promotions
- Geographic diffusion
- Repurchase / subscription dynamics
- Real-time ingestion of market signals

## Roadmap summary

| Phase | Status | Focus |
|---|---|---|
| 1 (MVP) | ✅ Done | Multi-step loop, small-world network, WOM, metrics |
| 1.5 (Upgrades) | 📋 In progress | networkx topology, System 1/2, two-layer WOM, async, 100×50 scale |
| 2 (Seed) | 🎯 Next | Real product case (women's apparel), seed-document ingestion |
| 3 (Calibration) | Future | Monte Carlo CI, sensitivity analysis, backtesting against real launches |
| 4 (Production) | Future | Dashboard, multi-product, competitor modeling |
