# 30 -- Knowledge Retrieval System

## Version
v1.0 (2026-04-16)

## Problem

The original `market_context` field on `SimulationConfig` is a plain string -- fine for a 3-5 sentence category briefing, but unusable when the seed document is a multi-dimensional report spanning competitor crawls, industry reports, government data, and consumer reviews. Injecting the entire document into every agent's prompt would blow token limits and drown out relevance.

## Solution

A three-layer pipeline that chunks, categorizes, and selectively retrieves knowledge per agent:

```
Raw seed doc(s)
    |
    v
DocumentProcessor.process()          -- parse markdown into categorized chunks
    |
    v
SeedKnowledge (indexed by category)  -- the knowledge store
    |
    v
KnowledgeRetriever.retrieve(agent)   -- select relevant chunks per agent
    |
    v
Context string (token-bounded)       -- injected into purchase decision prompt
```

### Layer 1: DocumentProcessor

Parses markdown documents by splitting on `#`/`##`/`###` headers.  Each section is assigned a `KnowledgeCategory` via keyword matching on the header text:

| Keyword in header | Category |
|---|---|
| competitor, competing | `competitor_product` |
| market overview, market size, industry, trend | `market_overview` |
| price, pricing | `pricing_landscape` |
| consumer, customer, demographic | `consumer_insight` |
| channel, distribution, retail | `channel_distribution` |
| brand, reputation, perception | `brand_perception` |
| regulat, government, policy | `regulatory` |
| review, rating, feedback | `product_review` |
| (no match) | `general` |

Sections are further split into paragraph-bounded chunks with a configurable `max_chunk_tokens` (default 300).  Each chunk gets:
- **category** -- from header keyword matching
- **source** -- document label
- **importance** -- default 0.5, overridable with `<!-- importance: 0.8 -->` in the source
- **brand** -- extracted from `Brand: X` patterns in headers
- **tags** -- auto-generated from header words + category
- **token_estimate** -- rough count (Chinese chars x 1.5, English words x 1.3)

Supports `process_multiple({label: text, ...})` for multi-document ingestion.

### Layer 2: SeedKnowledge

Container with query methods:
- `by_category(cat)` -- all chunks of a given category
- `by_brand(brand)` -- all chunks mentioning a specific brand
- `by_tag(tag)` -- free-form tag search
- `describe()` -- summary string with chunk counts and token totals

### Layer 3: KnowledgeRetriever

Selects relevant chunks for each agent based on:

**Cognitive style** -- determines which categories the agent cares about:

| Style | Priority categories |
|---|---|
| analytical | competitor_product, pricing_landscape, market_overview, product_review |
| emotional | brand_perception, product_review, consumer_insight |
| social | consumer_insight, brand_perception, product_review, market_overview |
| skeptical | product_review, regulatory, pricing_landscape, competitor_product |
| balanced | market_overview, competitor_product, pricing_landscape, consumer_insight, product_review |

**Adopter tier** -- determines breadth (how many categories to draw from):

| Tier | Breadth |
|---|---|
| innovator | 5 categories |
| early_adopter | 4 |
| early_majority | 3 |
| late_majority | 2 |
| laggard | 2 |

**Token budget** -- hard cap (default 800 tokens) on total injected context. Chunks are filled in priority order until the budget is exhausted.

**Always-include** -- `pricing_landscape` is included for all agents regardless of style (everyone has some price awareness).

**Global top-5** -- the 5 highest-importance chunks from any category are added as candidates, ensuring critical information reaches all agents.

## Integration

- `SimulationConfig.knowledge_base: Optional[SeedKnowledge]` -- set this instead of (or alongside) `market_context`
- When `knowledge_base` is provided, the runner instantiates a `KnowledgeRetriever` and calls `retriever.retrieve(agent)` per agent in `_decide_with_retry`
- The retrieved context string replaces the raw `market_context` in the purchase decision prompt
- Backward compatible: if `knowledge_base` is None, falls back to `market_context` string
- `MonteCarloRunner` and `ParameterSweep` both forward `knowledge_base` to child configs

## Usage

```python
from simulation.knowledge import DocumentProcessor, SeedKnowledge

# Single document
processor = DocumentProcessor(source_label="industry_report")
knowledge = processor.process(markdown_text)

# Multiple documents
knowledge = processor.process_multiple({
    "industry_report": report_text,
    "competitor_crawl": crawl_text,
    "government_data": policy_text,
})

# Wire into config
config = SimulationConfig(
    product=product,
    knowledge_base=knowledge,
    # market_context is ignored when knowledge_base is set
)
```

## Known Limitations

1. **Flat keyword matching** -- sub-headers like `## Brand: Zara` under `# Competitor Products` match "brand" (-> `brand_perception`) rather than inheriting the parent's "competitor" category. Users can work around this by using explicit headers like `## Competitor: Zara`.
2. **No semantic search** -- retrieval is category-based, not embedding-based. For the current scale (hundreds of chunks, not millions) this is sufficient and avoids a vector DB dependency.
3. **Token estimation is rough** -- the `_estimate_tokens()` function uses character/word heuristics. For exact counts, integrate a tokenizer (e.g. tiktoken) in a future phase.

## File

`simulation/knowledge.py` -- all classes in one module (~530 lines).
