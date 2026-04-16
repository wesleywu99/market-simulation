# 40 — Seed Case: Affordable Women's Apparel

## Version
v0.1 (2026-04-15) — Draft. Product spec + agent persona distribution to be filled in once Phase 1.5 lands.

## Status
🎯 First real seed case. Planned to run once Phase 1.5 upgrades are complete.

## Why this case first

- **Low unit price** (likely ¥100-500 range) — reduces the "budget gate" from a dominant filter to a soft one, so we stress-test the social-proof and sentiment dynamics rather than pure affordability.
- **High WOM sensitivity** — apparel buys are strongly influenced by friends' opinions, visible adoption ("I saw Alice wearing it"), and social media signals. This is a good stress test for the WOM engine.
- **Shorter decision horizon** than AI wearables — faster "defer → decide" cycles mean we see diffusion effects within the 50-step window.
- **Known market shape** — unlike a truly novel category, there are reference brands (Uniqlo, Shein, Zara) we can sanity-check our outputs against.

## Product spec (TBD)

```python
Product(
    product_id="SKU_APPAREL_001",
    name="[TBD]",
    category="women's apparel",
    price=...,       # target ¥150-¥400
    quality=...,     # 0.0-1.0, calibrated to reference brand
    features=[
        # list of differentiating attributes, e.g.
        # "sustainable fabric", "trend-aligned silhouette", "size inclusive",
    ],
    target_persona="...",
)
```

## Agent population (TBD)

For a women's apparel product, the agent population should be:

- **Gender-skewed** — ~85% female agents, ~15% male (for gift-giving scenarios).
- **Age distribution** — primary 22-45 with a long tail.
- **Rogers tier distribution** — default 2.5% / 13.5% / 34% / 34% / 16% unless the category deviates (fast fashion skews to earlier adopters).
- **Income distribution** — calibrated to target market (e.g. tier-1 Chinese cities, if that's the target).
- **Interest clusters** — agents who care about fashion vs. those who don't. This affects initial `awareness` levels and baseline `trust_in_category`.

## Questions the simulation should answer

1. **What's the 50-step adoption ceiling** given price point P and quality Q?
2. **Which tier is the bottleneck?** If early_majority stalls, we're not crossing the chasm.
3. **What does negative WOM do?** A 10% reduction in perceived quality — how much adoption does it cost?
4. **Is there a price elasticity inflection?** Run the sim at 5 price points, compare ceilings.
5. **How fast does the curve reach 50% of ceiling?** (proxy for time-to-market-penetration)

## Success criteria for this seed case

Not "is the forecast accurate?" — we have no ground truth. Instead:

- ✅ Curves are **shaped** like real apparel diffusion (S-curve, not flat, not explosive).
- ✅ **Tier ordering** respects Rogers (innovators first, laggards last).
- ✅ **Price sensitivity is monotonic** (higher price → lower ceiling).
- ✅ **Negative WOM is louder than positive** (asymmetry visible in sensitivity analysis).
- ✅ A human domain expert, shown the output, says "this looks plausible" more often than not.

If all five hold, we've built a useful tool, even without quantitative validation.

## Open questions

- Source of persona distribution — synthetic (LLM-generated from a distribution spec) vs imported (real survey data)?
- How to calibrate `quality` — a single 0.0-1.0 scalar oversimplifies apparel. Phase 2 should add a multi-attribute quality vector (fit, fabric, style, durability).
- Influencer modeling — for apparel, a few high-influence agents (KOLs) dominate WOM. Do we seed them explicitly or emerge them from the network degree distribution?

## Next action

Once Phase 1.5 is complete, fill in the TBD sections of this doc and run the first real simulation.
