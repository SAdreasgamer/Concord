"""
Heuristic page classifier — determines page value BEFORE touching the LLM.

Classifies each PDF page into categories based on lexical and structural
signals so the pipeline can:
  1. Skip junk pages (TOC, cover, disclaimer) → saves API calls
  2. Route table-heavy pages to local table extraction → no LLM needed
  3. Send only narrative / complex pages to the LLM

Zero external dependencies. All rules are deterministic and fast.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PageType(str, Enum):
    """Classification of a PDF page by content type."""

    FINANCIAL_TABLE = "financial_table"   # Structured tables with numbers → local extraction
    NARRATIVE_DATA = "narrative_data"     # Prose containing facts → LLM extraction
    TABLE_OF_CONTENTS = "toc"            # TOC / index pages → skip
    COVER_PAGE = "cover"                 # Title / cover pages → skip
    LEGAL_BOILERPLATE = "boilerplate"    # Disclaimers, risk factors → skip
    SPARSE_PAGE = "sparse"              # Too little text to be useful → skip
    DIRECTOR_TABLE = "director_table"    # Director/officer listings → skip
    UNKNOWN = "unknown"                  # Could not classify → send to LLM as fallback


@dataclass
class PageClassification:
    """Result of classifying a single page."""

    page_number: int
    page_type: PageType
    confidence: float = 0.5
    reason: str = ""
    word_count: int = 0
    has_tables: bool = False
    financial_keyword_density: float = 0.0
    should_send_to_llm: bool = True
    should_extract_tables_locally: bool = False


# --- Keyword sets for classification ---

FINANCIAL_KEYWORDS = {
    "revenue", "ebitda", "profit", "loss", "margin", "crore", "million",
    "billion", "income", "expense", "operating", "net", "gross", "earnings",
    "growth", "yoy", "qoq", "quarter", "fiscal", "fy", "q1", "q2", "q3", "q4",
    "turnover", "cash", "debt", "equity", "assets", "liabilities", "capital",
    "depreciation", "amortization", "tax", "pbt", "pat", "eps", "dividend",
    "capex", "working", "inventory", "receivable", "payable", "gdp",
    "inflation", "interest", "rate", "percent", "basis", "points", "bps",
    "consolidated", "standalone", "restated", "audited", "unaudited",
    "share", "diluted", "weighted", "average", "shipment", "volume",
}

BOILERPLATE_PATTERNS = [
    r"risk\s+factors?",
    r"disclaimer",
    r"forward[- ]looking\s+statements?",
    r"safe\s+harbor",
    r"this\s+(prospectus|document|report)\s+(does\s+not|is\s+not|should\s+not)",
    r"no\s+representation\s+or\s+warranty",
    r"neither\s+the\s+company\s+nor",
    r"past\s+performance\s+is\s+not",
    r"subject\s+to\s+market\s+risks?",
    r"read\s+the\s+offer\s+document",
    r"general\s+information\s+document",
]

TOC_PATTERNS = [
    r"table\s+of\s+contents?",
    r"contents?\s*$",
    r"^\s*(?:section|chapter|part)\s+\w+\s*\.{2,}",  # "Section A ............ 45"
    r"(?:\.\s*){3,}\d+\s*$",  # Dotted leaders to page numbers
]

DIRECTOR_PATTERNS = [
    r"name,?\s+designation,?\s+address",
    r"din\b",  # Director Identification Number
    r"date\s+of\s+birth",
    r"other\s+directorships?",
    r"key\s+managerial\s+personnel",
]

COVER_PATTERNS = [
    r"^\s*(red\s+herring|prospectus|offer\s+document)",
    r"^[A-Z\s]{20,}$",  # All-caps title lines
    r"securities\s+and\s+exchange\s+board",
    r"registrar\s+and\s+(share\s+)?transfer",
]


def _count_number_tokens(text: str) -> int:
    """Count tokens that look like financial numbers (with commas, decimals, ₹, %)."""
    return len(re.findall(
        r"(?:₹|rs\.?|inr|usd|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*(?:%|cr|crore|mn|million|billion|lakh|bps))?",
        text.lower(),
    ))


def _financial_keyword_density(text: str) -> float:
    """Fraction of words that are financial keywords."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in FINANCIAL_KEYWORDS)
    return hits / len(words)


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any regex pattern."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _count_dotted_leader_lines(text: str) -> int:
    """Count lines that look like TOC entries (text....page_number)."""
    lines = text.strip().split("\n")
    return sum(1 for line in lines if re.search(r"\.{3,}\s*\d+\s*$", line))


def classify_page(
    text: str,
    page_number: int,
    has_tables: bool = False,
    table_row_count: int = 0,
) -> PageClassification:
    """
    Classify a single page of PDF text using heuristic rules.

    This is a fast, deterministic pre-filter that runs BEFORE any LLM call.
    Pages classified as junk are skipped entirely, saving API costs.

    Args:
        text: Raw extracted text from the page.
        page_number: 1-based page number.
        has_tables: Whether PyMuPDF detected structured tables on this page.
        table_row_count: Total number of rows across all detected tables.

    Returns:
        PageClassification with routing decision.
    """
    word_count = len(text.split())
    fin_density = _financial_keyword_density(text)
    num_count = _count_number_tokens(text)

    base = PageClassification(
        page_number=page_number,
        page_type=PageType.UNKNOWN,
        word_count=word_count,
        has_tables=has_tables,
        financial_keyword_density=fin_density,
    )

    # --- Rule 1: Very sparse pages (< 30 words) → skip ---
    if word_count < 30:
        base.page_type = PageType.SPARSE_PAGE
        base.confidence = 0.95
        base.reason = f"Only {word_count} words — likely cover/image/blank"
        base.should_send_to_llm = False
        return base

    # --- Rule 2: Cover pages (first 3 pages with cover patterns) ---
    if page_number <= 3 and _matches_any_pattern(text, COVER_PATTERNS):
        base.page_type = PageType.COVER_PAGE
        base.confidence = 0.85
        base.reason = "Matched cover page patterns in early pages"
        base.should_send_to_llm = False
        return base

    # --- Rule 3: Table of Contents ---
    dotted_lines = _count_dotted_leader_lines(text)
    if (
        _matches_any_pattern(text, TOC_PATTERNS)
        or dotted_lines >= 5
    ):
        base.page_type = PageType.TABLE_OF_CONTENTS
        base.confidence = 0.90
        base.reason = f"TOC pattern detected ({dotted_lines} dotted-leader lines)"
        base.should_send_to_llm = False
        return base

    # --- Rule 4: Director / officer listing tables ---
    if _matches_any_pattern(text, DIRECTOR_PATTERNS) and has_tables:
        base.page_type = PageType.DIRECTOR_TABLE
        base.confidence = 0.80
        base.reason = "Director/officer listing table detected"
        base.should_send_to_llm = False
        return base

    # --- Rule 5: Legal boilerplate ---
    if _matches_any_pattern(text, BOILERPLATE_PATTERNS) and fin_density < 0.03:
        base.page_type = PageType.LEGAL_BOILERPLATE
        base.confidence = 0.80
        base.reason = "Boilerplate legal language with low financial keyword density"
        base.should_send_to_llm = False
        return base

    # --- Rule 6: Financial table with structured data → local extraction candidate ---
    if has_tables and table_row_count >= 3 and (fin_density >= 0.05 or num_count >= 8):
        base.page_type = PageType.FINANCIAL_TABLE
        base.confidence = 0.85
        base.reason = (
            f"Structured table ({table_row_count} rows) with financial content "
            f"(keyword density={fin_density:.1%}, numbers={num_count})"
        )
        base.should_send_to_llm = True  # Still send to LLM for richer extraction
        base.should_extract_tables_locally = True  # Also extract locally
        return base

    # --- Rule 7: Narrative with financial content → LLM extraction ---
    if fin_density >= 0.04 or num_count >= 5:
        base.page_type = PageType.NARRATIVE_DATA
        base.confidence = 0.75
        base.reason = f"Financial narrative (keyword density={fin_density:.1%}, numbers={num_count})"
        base.should_send_to_llm = True
        return base

    # --- Rule 8: Low financial density, moderate text → likely boilerplate ---
    if fin_density < 0.02 and word_count > 100:
        base.page_type = PageType.LEGAL_BOILERPLATE
        base.confidence = 0.60
        base.reason = f"Dense text but very low financial keyword density ({fin_density:.1%})"
        base.should_send_to_llm = False
        return base

    # --- Fallback: Unknown → send to LLM to be safe ---
    base.page_type = PageType.UNKNOWN
    base.confidence = 0.50
    base.reason = "Could not classify with high confidence — sending to LLM as fallback"
    base.should_send_to_llm = True
    return base


def classify_document_pages(
    pages: list[dict],
) -> list[PageClassification]:
    """
    Classify all pages in a document.

    Args:
        pages: List of dicts with keys: text, page_number, has_tables, table_row_count

    Returns:
        List of PageClassification objects.
    """
    classifications = []
    for page in pages:
        cls = classify_page(
            text=page["text"],
            page_number=page["page_number"],
            has_tables=page.get("has_tables", False),
            table_row_count=page.get("table_row_count", 0),
        )
        classifications.append(cls)

    # Log summary
    sent_to_llm = sum(1 for c in classifications if c.should_send_to_llm)
    skipped = len(classifications) - sent_to_llm
    local_tables = sum(1 for c in classifications if c.should_extract_tables_locally)

    logger.info(
        "Page classification: %d total → %d to LLM, %d skipped, %d local table candidates",
        len(classifications),
        sent_to_llm,
        skipped,
        local_tables,
    )

    return classifications
