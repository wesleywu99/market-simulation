"""
SeedKnowledge -- chunked market knowledge for grounding agent decisions.

When the user provides a large seed document (competitor crawls, industry
reports, government data), it can't be injected verbatim into every
agent's prompt -- it would blow token limits and dilute relevance.

This module provides:

* ``KnowledgeChunk`` -- one piece of categorized knowledge with metadata.
* ``SeedKnowledge``  -- the full knowledge base (list of chunks + index).
* ``KnowledgeRetriever`` -- selects relevant chunks per agent within a
  token budget, adapting what each agent "knows" to their cognitive style.
* ``DocumentProcessor`` -- parses raw markdown/text into KnowledgeChunks.

Architecture
------------
::

    Raw seed doc(s)
        |
        v
    DocumentProcessor.process()
        |
        v
    SeedKnowledge (categorized chunks)
        |
        v
    KnowledgeRetriever.retrieve(agent, product, step)
        |
        v
    Relevant context string (token-bounded)
        |
        v
    Injected into purchase decision prompt

Knowledge categories
--------------------
Each chunk belongs to one category.  The retriever uses these to match
chunks to agent cognitive styles and decision needs.

* ``competitor_product`` -- specs, pricing, reviews of a competing product
* ``market_overview``    -- market size, growth, trends, segmentation
* ``pricing_landscape``  -- price bands, elasticity data, discounting norms
* ``consumer_insight``   -- demographics, preferences, pain points, surveys
* ``channel_distribution`` -- retail/online channels, market access
* ``brand_perception``   -- reputation data, NPS, social media sentiment
* ``regulatory``         -- government policy, standards, import rules
* ``product_review``     -- real user reviews, ratings, complaints
* ``general``            -- anything that doesn't fit the above
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.consumer import ConsumerAgent


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE CATEGORY
# ─────────────────────────────────────────────────────────────

class KnowledgeCategory(Enum):
    COMPETITOR_PRODUCT = "competitor_product"
    MARKET_OVERVIEW = "market_overview"
    PRICING_LANDSCAPE = "pricing_landscape"
    CONSUMER_INSIGHT = "consumer_insight"
    CHANNEL_DISTRIBUTION = "channel_distribution"
    BRAND_PERCEPTION = "brand_perception"
    REGULATORY = "regulatory"
    PRODUCT_REVIEW = "product_review"
    GENERAL = "general"


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE CHUNK
# ─────────────────────────────────────────────────────────────

@dataclass
class KnowledgeChunk:
    """One piece of categorized market knowledge.

    Parameters
    ----------
    content : str
        The actual text (a paragraph, a table, a bullet list).
    category : KnowledgeCategory
        What kind of knowledge this is.
    source : str
        Where this came from (e.g. "industry_report_2024.pdf", "taobao_crawl").
    importance : float
        0.0-1.0.  Higher = more likely to be selected by the retriever.
    tags : list of str
        Free-form tags for matching (e.g. ["price", "competitor:zara", "quality"]).
    brand : str or None
        If this chunk is about a specific competing brand.
    token_estimate : int
        Approximate token count (rough: len(content) / 2 for Chinese,
        len(content.split()) * 1.3 for English).
    """
    content: str
    category: KnowledgeCategory
    source: str = ""
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    brand: Optional[str] = None
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = _estimate_tokens(self.content)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate.  Chinese chars ~1.5 tokens each; English ~1.3 per word."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars > len(text) * 0.3:
        # Predominantly Chinese
        return int(len(text) * 1.5)
    # Predominantly English/mixed
    return int(len(text.split()) * 1.3)


# ─────────────────────────────────────────────────────────────
# SEED KNOWLEDGE (the store)
# ─────────────────────────────────────────────────────────────

@dataclass
class SeedKnowledge:
    """Container for all chunked market knowledge for a seed case.

    Provides category-based indexing and basic query methods.  The
    retriever uses these to select relevant chunks per agent.
    """
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def add(self, chunk: KnowledgeChunk) -> None:
        self.chunks.append(chunk)

    def add_many(self, chunks: List[KnowledgeChunk]) -> None:
        self.chunks.extend(chunks)

    def by_category(self, category: KnowledgeCategory) -> List[KnowledgeChunk]:
        return [c for c in self.chunks if c.category == category]

    def by_brand(self, brand: str) -> List[KnowledgeChunk]:
        brand_lower = brand.lower()
        return [c for c in self.chunks if c.brand and c.brand.lower() == brand_lower]

    def by_tag(self, tag: str) -> List[KnowledgeChunk]:
        tag_lower = tag.lower()
        return [c for c in self.chunks if tag_lower in [t.lower() for t in c.tags]]

    def total_tokens(self) -> int:
        return sum(c.token_estimate for c in self.chunks)

    def describe(self) -> str:
        if not self.chunks:
            return "empty"
        cats = {}
        for c in self.chunks:
            cats[c.category.value] = cats.get(c.category.value, 0) + 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
        return f"{len(self.chunks)} chunks (~{self.total_tokens()} tokens) [{cat_str}]"


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE RETRIEVER
# ─────────────────────────────────────────────────────────────

# Cognitive style -> preferred knowledge categories (ordered by priority)
_STYLE_PREFERENCES: Dict[str, List[KnowledgeCategory]] = {
    "analytical": [
        KnowledgeCategory.COMPETITOR_PRODUCT,
        KnowledgeCategory.PRICING_LANDSCAPE,
        KnowledgeCategory.MARKET_OVERVIEW,
        KnowledgeCategory.PRODUCT_REVIEW,
    ],
    "emotional": [
        KnowledgeCategory.BRAND_PERCEPTION,
        KnowledgeCategory.PRODUCT_REVIEW,
        KnowledgeCategory.CONSUMER_INSIGHT,
    ],
    "social": [
        KnowledgeCategory.CONSUMER_INSIGHT,
        KnowledgeCategory.BRAND_PERCEPTION,
        KnowledgeCategory.PRODUCT_REVIEW,
        KnowledgeCategory.MARKET_OVERVIEW,
    ],
    "skeptical": [
        KnowledgeCategory.PRODUCT_REVIEW,
        KnowledgeCategory.REGULATORY,
        KnowledgeCategory.PRICING_LANDSCAPE,
        KnowledgeCategory.COMPETITOR_PRODUCT,
    ],
    "balanced": [
        KnowledgeCategory.MARKET_OVERVIEW,
        KnowledgeCategory.COMPETITOR_PRODUCT,
        KnowledgeCategory.PRICING_LANDSCAPE,
        KnowledgeCategory.CONSUMER_INSIGHT,
        KnowledgeCategory.PRODUCT_REVIEW,
    ],
}

# Adopter tier -> how many categories to draw from (innovators get more breadth)
_TIER_BREADTH = {
    "innovator": 5,
    "early_adopter": 4,
    "early_majority": 3,
    "late_majority": 2,
    "laggard": 2,
}


class KnowledgeRetriever:
    """Selects relevant knowledge chunks for each agent's decision prompt.

    Adapts what each agent "knows" based on:
    * **Cognitive style** -- analytical agents get competitor specs and
      pricing data; emotional agents get brand perception and reviews.
    * **Adopter tier** -- innovators see broader market context; laggards
      see a narrower slice (they don't research as much).
    * **Token budget** -- hard cap on total injected context.  Chunks are
      filled in priority order until the budget is exhausted.

    Parameters
    ----------
    knowledge : SeedKnowledge
        The full knowledge base.
    token_budget : int
        Maximum tokens to inject per agent prompt.  Default 800
        (~400 Chinese characters or ~600 English words).
    always_include_categories : list of KnowledgeCategory
        Categories that every agent gets regardless of style.
        Default: [PRICING_LANDSCAPE] -- everyone has some price awareness.
    """

    def __init__(
        self,
        knowledge: SeedKnowledge,
        token_budget: int = 800,
        always_include_categories: Optional[List[KnowledgeCategory]] = None,
    ) -> None:
        self.knowledge = knowledge
        self.token_budget = token_budget
        self.always_include = always_include_categories or [
            KnowledgeCategory.PRICING_LANDSCAPE,
        ]

    def retrieve(self, agent: "ConsumerAgent") -> str:
        """Return a context string for this agent, within token budget."""
        if not self.knowledge.chunks:
            return ""

        style = agent.state.profile.cognitive_style
        tier = agent.state.profile.adopter_tier

        # 1. Determine which categories this agent cares about
        preferred = list(_STYLE_PREFERENCES.get(style, _STYLE_PREFERENCES["balanced"]))
        breadth = _TIER_BREADTH.get(tier, 3)
        categories = preferred[:breadth]

        # 2. Add always-include categories (deduplicated)
        for cat in self.always_include:
            if cat not in categories:
                categories.append(cat)

        # 3. Gather candidate chunks in priority order
        candidates: List[KnowledgeChunk] = []
        seen_ids: set = set()

        for cat in categories:
            cat_chunks = self.knowledge.by_category(cat)
            # Sort by importance (highest first)
            cat_chunks.sort(key=lambda c: c.importance, reverse=True)
            for chunk in cat_chunks:
                cid = id(chunk)
                if cid not in seen_ids:
                    candidates.append(chunk)
                    seen_ids.add(cid)

        # 4. Also add high-importance chunks from any category
        all_by_importance = sorted(
            self.knowledge.chunks, key=lambda c: c.importance, reverse=True
        )
        for chunk in all_by_importance[:5]:  # top-5 globally
            cid = id(chunk)
            if cid not in seen_ids:
                candidates.append(chunk)
                seen_ids.add(cid)

        # 5. Fill within token budget
        selected: List[KnowledgeChunk] = []
        used_tokens = 0
        for chunk in candidates:
            if used_tokens + chunk.token_estimate > self.token_budget:
                continue
            selected.append(chunk)
            used_tokens += chunk.token_estimate

        if not selected:
            return ""

        # 6. Format as a readable context block
        return self._format(selected)

    def _format(self, chunks: List[KnowledgeChunk]) -> str:
        """Render selected chunks as a readable text block."""
        lines = []
        current_cat = None
        for chunk in chunks:
            if chunk.category != current_cat:
                current_cat = chunk.category
                label = current_cat.value.replace("_", " ").title()
                lines.append(f"[{label}]")
            lines.append(chunk.content.strip())
            lines.append("")  # blank line between chunks
        return "\n".join(lines).strip()

    def describe(self) -> str:
        return (
            f"KnowledgeRetriever(budget={self.token_budget} tokens, "
            f"knowledge={self.knowledge.describe()})"
        )


# ─────────────────────────────────────────────────────────────
# DOCUMENT PROCESSOR
# ─────────────────────────────────────────────────────────────

# Section header patterns -> category mapping
_SECTION_CATEGORY_MAP: Dict[str, KnowledgeCategory] = {
    "competitor": KnowledgeCategory.COMPETITOR_PRODUCT,
    "competing": KnowledgeCategory.COMPETITOR_PRODUCT,
    "market overview": KnowledgeCategory.MARKET_OVERVIEW,
    "market size": KnowledgeCategory.MARKET_OVERVIEW,
    "industry": KnowledgeCategory.MARKET_OVERVIEW,
    "market trend": KnowledgeCategory.MARKET_OVERVIEW,
    "price": KnowledgeCategory.PRICING_LANDSCAPE,
    "pricing": KnowledgeCategory.PRICING_LANDSCAPE,
    "consumer": KnowledgeCategory.CONSUMER_INSIGHT,
    "customer": KnowledgeCategory.CONSUMER_INSIGHT,
    "demographic": KnowledgeCategory.CONSUMER_INSIGHT,
    "target audience": KnowledgeCategory.CONSUMER_INSIGHT,
    "channel": KnowledgeCategory.CHANNEL_DISTRIBUTION,
    "distribution": KnowledgeCategory.CHANNEL_DISTRIBUTION,
    "retail": KnowledgeCategory.CHANNEL_DISTRIBUTION,
    "brand": KnowledgeCategory.BRAND_PERCEPTION,
    "reputation": KnowledgeCategory.BRAND_PERCEPTION,
    "perception": KnowledgeCategory.BRAND_PERCEPTION,
    "regulat": KnowledgeCategory.REGULATORY,
    "government": KnowledgeCategory.REGULATORY,
    "policy": KnowledgeCategory.REGULATORY,
    "standard": KnowledgeCategory.REGULATORY,
    "review": KnowledgeCategory.PRODUCT_REVIEW,
    "rating": KnowledgeCategory.PRODUCT_REVIEW,
    "feedback": KnowledgeCategory.PRODUCT_REVIEW,
    "complaint": KnowledgeCategory.PRODUCT_REVIEW,
}


def _guess_category(header: str) -> KnowledgeCategory:
    """Infer category from a section header string."""
    h = header.lower().strip()
    for keyword, cat in _SECTION_CATEGORY_MAP.items():
        if keyword in h:
            return cat
    return KnowledgeCategory.GENERAL


def _guess_brand(text: str, header: str) -> Optional[str]:
    """Try to extract a brand name from chunk header/content.

    Looks for patterns like "Brand: Zara" or "## Zara" in headers.
    Returns None if no clear brand is detected.
    """
    # Check header for "Brand: X" or standalone brand name
    brand_match = re.search(r'brand[:\s]+([A-Za-z\u4e00-\u9fff]+)', header, re.IGNORECASE)
    if brand_match:
        return brand_match.group(1).strip()
    return None


class DocumentProcessor:
    """Parse raw markdown/text into categorized KnowledgeChunks.

    Splits on markdown headers (``#``, ``##``, ``###``) and assigns
    categories based on header text.  Each section becomes one or more
    chunks (split further if too long).

    Parameters
    ----------
    max_chunk_tokens : int
        Maximum tokens per chunk.  Sections longer than this are split
        at paragraph boundaries.  Default 300 (~150 Chinese chars).
    default_importance : float
        Base importance for all chunks.  Override per-section with
        ``<!-- importance: 0.8 -->`` in the source doc.
    source_label : str
        Label attached to all chunks from this document.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 300,
        default_importance: float = 0.5,
        source_label: str = "seed_doc",
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.default_importance = default_importance
        self.source_label = source_label

    def process(self, text: str) -> SeedKnowledge:
        """Parse a markdown document into a SeedKnowledge store."""
        sections = self._split_sections(text)
        knowledge = SeedKnowledge()

        for header, body in sections:
            if not body.strip():
                continue

            category = _guess_category(header)
            brand = _guess_brand(body, header)
            importance = self._extract_importance(body)
            tags = self._extract_tags(header, category)

            paragraphs = self._split_paragraphs(body)
            chunk_group: List[str] = []
            group_tokens = 0

            for para in paragraphs:
                para_tokens = _estimate_tokens(para)
                if group_tokens + para_tokens > self.max_chunk_tokens and chunk_group:
                    # Flush current group as a chunk
                    knowledge.add(KnowledgeChunk(
                        content="\n".join(chunk_group),
                        category=category,
                        source=self.source_label,
                        importance=importance,
                        tags=tags,
                        brand=brand,
                    ))
                    chunk_group = []
                    group_tokens = 0

                chunk_group.append(para)
                group_tokens += para_tokens

            # Flush remaining
            if chunk_group:
                knowledge.add(KnowledgeChunk(
                    content="\n".join(chunk_group),
                    category=category,
                    source=self.source_label,
                    importance=importance,
                    tags=tags,
                    brand=brand,
                ))

        return knowledge

    def process_multiple(
        self,
        documents: Dict[str, str],
    ) -> SeedKnowledge:
        """Process multiple documents into a single SeedKnowledge.

        Parameters
        ----------
        documents : dict
            {source_label: text_content} pairs.
        """
        combined = SeedKnowledge()
        for label, text in documents.items():
            self.source_label = label
            doc_knowledge = self.process(text)
            combined.add_many(doc_knowledge.chunks)
        return combined

    # ── internal helpers ─────────────────────────────────────

    def _split_sections(self, text: str) -> List[tuple]:
        """Split markdown into (header, body) pairs."""
        # Match # Header, ## Header, ### Header
        pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            # No headers -- treat entire text as one section
            return [("document", text)]

        sections = []
        # Content before first header
        if matches[0].start() > 0:
            pre = text[:matches[0].start()].strip()
            if pre:
                sections.append(("preamble", pre))

        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((header, body))

        return sections

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs on double newlines."""
        raw = re.split(r'\n\s*\n', text)
        return [p.strip() for p in raw if p.strip()]

    def _extract_importance(self, text: str) -> float:
        """Look for <!-- importance: 0.8 --> directives in text."""
        match = re.search(r'<!--\s*importance:\s*([\d.]+)\s*-->', text)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
        return self.default_importance

    def _extract_tags(self, header: str, category: KnowledgeCategory) -> List[str]:
        """Generate tags from header text and category."""
        tags = [category.value]
        # Add significant words from header as tags
        words = re.findall(r'[A-Za-z\u4e00-\u9fff]{2,}', header.lower())
        tags.extend(w for w in words if w not in {"the", "and", "for", "from", "with"})
        return tags
