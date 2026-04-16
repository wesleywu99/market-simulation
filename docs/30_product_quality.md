# 30 — Product Quality (Semantic Blind-Box Model)

## Version
v1.0 (2026-04-15) — First spec. To be implemented as Phase 1.5 upgrade #12.

## Status
📋 Designed. Implementation in progress.

## The problem with scalar quality

Phase 1 modeled product quality as a single `float in [0, 1]`:

```python
Product(name="SkyView Pro X1", quality=0.82)
```

This collapses every aspect of a product into one number, which loses the **structure that drives word-of-mouth**:

- Two buyers of the same product get the same scalar score → their WOM differs only in random noise.
- The LLM is asked to write WOM from "quality 6.2/10" — generic, sentiment-free, easy to ignore.
- Dimension-specific defects (the loud, viral kind) are erased into the average.
- Personas can't weight what they care about — a fashion-forward agent and a value-shopper agent face the same "quality" signal.

Real product reviews are concrete: "the zipper jams," "battery dies in 22 minutes," "color is nothing like the photo." The blind-box model captures this.

## The model

### 1. Multi-dimensional quality

A `Product` carries a `quality_dimensions` map: dimension name → latent quality `[0, 1]`.

```python
Product(
    name="SkyView Pro X1",
    quality_dimensions={
        "camera_quality":     0.85,
        "battery_life":       0.78,
        "obstacle_avoidance": 0.70,
        "app_software":       0.55,
        "build_quality":      0.80,
    },
    defect_bank={...},
    praise_bank={...},
)
```

The dimensions are **category-specific**. The data structure and the engine are not.

### 2. Defect / praise bank

For each `(dimension, severity)` pair, a list of pre-authored review snippets:

```python
defect_bank = {
    ("app_software", "low"): [
        "App crashes mid-flight, lost the drone for 5 minutes",
        "Live preview lags 2 seconds — useless for action shots",
    ],
    ("battery_life", "low"): [
        "Got 22 minutes max, nowhere near the advertised 35",
    ],
}

praise_bank = {
    ("camera_quality", "high"): [
        "4K footage is genuinely cinematic, even in low light",
    ],
}
```

**Severity buckets:** `low` (dimension < 0.4), `mid` (0.4-0.7), `high` (> 0.7).

The mid bucket is intentionally sparse — the loud reviews are at the extremes.

### 3. Per-buyer "blind box" sampling

When a buyer receives the product, the simulation rolls a stochastic experience:

```python
def open_box(product, agent, rng) -> ExperienceProfile:
    surfaced_defects = []
    surfaced_praises = []
    dim_scores = {}

    for dim, latent in product.quality_dimensions.items():
        # Per-buyer noise (some get a bad unit, some get lucky)
        observed = clamp(latent + rng.gauss(0, 0.15), 0, 1)
        dim_scores[dim] = observed

        severity = bucket(observed)  # low / mid / high
        if severity == "low":
            surfaced_defects.append(rng.choice(product.defect_bank[(dim, "low")]))
        elif severity == "high":
            surfaced_praises.append(rng.choice(product.praise_bank[(dim, "high")]))

    return ExperienceProfile(
        overall_score=weighted_avg(dim_scores, agent.dimension_weights),
        dimension_scores=dim_scores,
        surfaced_defects=surfaced_defects,
        surfaced_praises=surfaced_praises,
    )
```

Two buyers of the same product surface **different** defects, just like in real life.

### 4. Persona-weighted aggregation

`AgentProfile.dimension_weights: Dict[str, float]` lets each persona care about different things.

```python
# Fashion-forward early adopter
weights = {"color_accuracy": 0.4, "fit": 0.3, "material": 0.2, "construction": 0.1}

# Value shopper
weights = {"construction": 0.4, "material": 0.3, "fit": 0.2, "color_accuracy": 0.1}
```

Per-tier defaults live in the seed-case file; individual agents can override.

When weights are missing, the engine falls back to uniform.

### 5. WOM prompt receives concrete text

The new WOM user prompt feeds the LLM the **surfaced defects and praises**, not just an aggregate score:

```
Your purchase experience:
  Overall: 5.8/10
  What stood out (positive): "Build feels solid in the hand"
  What stood out (negative):
    - "App crashes mid-flight, lost the drone for 5 minutes"
    - "Got 22 minutes max, nowhere near the advertised 35"

Write the WOM message you'd actually send.
```

The LLM now has *grounded* material. The output naturally inherits the negativity bias.

## Architecture — generic vs per-category

| Layer | Generic (in code) | Per-category (in seed case) |
|---|---|---|
| `Product.quality_dimensions: Dict[str, float]` | ✅ | dimension names |
| `Product.defect_bank: Dict[(str, str), List[str]]` | ✅ | defect text |
| `Product.praise_bank: Dict[(str, str), List[str]]` | ✅ | praise text |
| `ExperienceProfile` dataclass | ✅ | — |
| `BlindBoxExperience` sampler (ABC + impl) | ✅ | — |
| `AgentProfile.dimension_weights` | ✅ | per-tier weight defaults |
| WOM prompt template | ✅ | — |

## Engine API

```python
# environment/product.py
@dataclass
class Product:
    quality_dimensions: Dict[str, float]
    defect_bank: Dict[Tuple[str, str], List[str]]
    praise_bank: Dict[Tuple[str, str], List[str]]
    # quality property = weighted average across dimensions, for backward compat

# agents/experience.py
@dataclass
class ExperienceProfile:
    overall_score: float            # 0-10 (weighted by persona)
    dimension_scores: Dict[str, float]
    surfaced_defects: List[str]
    surfaced_praises: List[str]

class ExperienceSampler(ABC):
    @abstractmethod
    def sample(self, product, agent, rng) -> ExperienceProfile: ...

class BlindBoxExperience(ExperienceSampler):
    """Default: per-dimension Gaussian noise + threshold sampling."""
    def __init__(self, noise_sigma=0.15, low_thresh=0.4, high_thresh=0.7): ...
```

Inject via `SimulationConfig.experience_sampler`. Default is `BlindBoxExperience`.

## Generality demo — the existing drone case

Migration is a **data-only** change:

```python
# Before
Product(name="SkyView Pro X1", quality=0.82, ...)

# After
Product(
    name="SkyView Pro X1",
    quality_dimensions={
        "camera_quality": 0.85, "battery_life": 0.78,
        "obstacle_avoidance": 0.70, "app_software": 0.55,
        "build_quality": 0.80,
    },
    defect_bank={
        ("app_software", "low"): ["App crashes mid-flight..."],
        ("battery_life", "low"): ["Got 22 minutes max..."],
        ("obstacle_avoidance", "low"): ["Hit a tree on the test flight"],
    },
    praise_bank={
        ("camera_quality", "high"): ["4K footage is genuinely cinematic"],
        ("build_quality", "high"): ["Feels solid, premium materials"],
    },
)
```

Same engine, different vocabulary.

## Caveats and non-goals

- **Severity buckets, not continuous text:** we deliberately use 3 buckets (low/mid/high), not per-value text. Continuous text generation would require LLM calls per buyer per dimension — too expensive.
- **Defect bank is finite and deterministic-ish:** we pick from a curated list. Variety is bounded but reproducible. Phase 2 may add LLM-generated defects for unseen severity-dimension combos.
- **No defect correlations:** in reality, "cheap material" and "shoddy stitching" co-occur. The model treats dimensions as independent. If this matters, a future `CorrelatedBlindBox` impl can sample from a covariance matrix.
- **Physical vs digital products:** the framing is literal for physical goods (different units, different defects). For software/services, "blind box" represents different use cases revealing different limitations. Same code, different mental model.
