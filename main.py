"""
Market Simulation — entry point.

Configures and runs a multi-step market simulation with social-network
effects and word-of-mouth propagation.

The drone product below demonstrates the generalised semantic blind-box
model (see ``docs/30_product_quality.md``): multi-dimensional quality
plus per-dimension defect/praise banks that ground WOM messages in
concrete specifics rather than an abstract scalar score.

Usage:
    python main.py
"""

from agents.experience import BlindBoxExperience
from agents.filters import System1Filter
from agents.influencer import DegreeBasedSeeding
from environment.network import SmallWorldNetwork
from environment.product import Product
from environment.wom import TrustWeightedWOM
from simulation.runner import SimulationConfig, SimulationRunner


def main() -> None:
    # ── Product under test ────────────────────────────────────
    # Multi-dimensional quality replaces the Phase 1 scalar.  The
    # ``quality`` field below is a placeholder — it is auto-derived from
    # ``quality_dimensions`` in ``Product.__post_init__``.
    product = Product(
        product_id="prod_001",
        name="SkyView Pro X1",
        category="consumer_drone",
        brand="SkyView",
        price=3999.0,
        features=["4K/60fps camera", "obstacle avoidance", "35min flight", "under 250g"],
        quality=0.0,  # overwritten from dimensions
        brand_reputation=0.45,
        distribution_channels=["online", "flagship_store"],
        quality_dimensions={
            "camera_quality":     0.85,
            "battery_life":       0.78,
            "obstacle_avoidance": 0.70,
            "app_software":       0.55,   # weak spot
            "build_quality":      0.80,
        },
        defect_bank={
            ("app_software", "low"): [
                "App crashes mid-flight, lost the drone for 5 minutes",
                "Live preview lags 2 seconds — useless for action shots",
                "Had to reinstall the firmware twice before it would pair",
            ],
            ("battery_life", "low"): [
                "Got 22 minutes max, nowhere near the advertised 35",
                "Battery drops 10% just hovering — won't survive real flights",
            ],
            ("obstacle_avoidance", "low"): [
                "Hit a tree on the test flight, sensors didn't trigger",
                "Avoidance is jittery in bright sun, kept false-tripping",
            ],
            ("camera_quality", "low"): [
                "4K footage is noisy in low light, sharpness is oversold",
            ],
            ("build_quality", "low"): [
                "Plastic arms flex worryingly, one hinge already squeaks",
            ],
        },
        praise_bank={
            ("camera_quality", "high"): [
                "4K footage is genuinely cinematic, even in low light",
                "Colours are punchy without looking oversaturated",
            ],
            ("build_quality", "high"): [
                "Feels solid in the hand — premium hinges, no flex",
                "Survived a 2-metre drop onto grass, zero damage",
            ],
            ("battery_life", "high"): [
                "Consistently getting 32+ minutes per charge, close to spec",
            ],
            ("obstacle_avoidance", "high"): [
                "Caught a branch I didn't even see — probably saved the drone",
            ],
            ("app_software", "high"): [
                "Pairing took 10 seconds, intelligent modes just work",
            ],
        },
        perceived_relative_advantage=0.65,
        perceived_compatibility=0.70,
        perceived_complexity=0.35,
        perceived_trialability=0.30,
        perceived_observability=0.75,
    )

    # ── Simulation config (swap strategies here) ──────────────
    config = SimulationConfig(
        product=product,
        n_agents=15,
        n_steps=4,
        seed=42,
        step_duration_days=7,           # 1 step = 1 week
        budget_refresh_interval=4,      # refresh every 4 steps (~monthly)
        # Swap these to change behaviour without touching the runner:
        network_builder=SmallWorldNetwork(k_neighbors=4, rewire_prob=0.3, seed=42),
        wom_engine=TrustWeightedWOM(close_trust_threshold=0.6, seed=42),
        decision_filter=System1Filter(),
        experience_sampler=BlindBoxExperience(),
        influencer_seeding=DegreeBasedSeeding(top_k_fraction=0.1),
        trace_dir="traces",
    )

    # ── Run ───────────────────────────────────────────────────
    runner = SimulationRunner(config)
    report = runner.run()
    runner.print_report(report)


if __name__ == "__main__":
    main()
