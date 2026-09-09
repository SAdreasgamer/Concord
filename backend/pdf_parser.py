"""
PDF parsing module.

Extracts text from PDFs page-by-page using PyMuPDF, preserving page numbers.
Also detects and extracts structured tables for local (non-LLM) fact extraction.
No hardcoded logic — works with any PDF regardless of content or structure.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

import pymupdf  # PyMuPDF

from backend.models import PageChunk


logger = logging.getLogger(__name__)


def extract_page_text(page: pymupdf.Page) -> str:
    """
    Extract text preserving spatial horizontal alignment and reading order.

    In financial PDFs, tables and multi-column rows often place metric labels in one
    text block and numbers in a separate text block. Grouping words by their vertical
    center coordinate (y-axis) keeps labels and their corresponding numbers on the
    same line, preventing footnote detachment and table column jumbling.
    """
    try:
        words = page.get_text("words")
        if not words:
            return page.get_text("text").strip()

        buckets: dict[float, list] = {}
        for w in words:
            # w: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            cy = (w[1] + w[3]) / 2.0
            matched_bucket = None
            for b_y in buckets:
                if abs(cy - b_y) <= 3.5:
                    matched_bucket = b_y
                    break
            if matched_bucket is None:
                matched_bucket = cy
                buckets[matched_bucket] = []
            buckets[matched_bucket].append(w)

        sorted_y = sorted(buckets.keys())
        lines = []
        for y in sorted_y:
            wl = sorted(buckets[y], key=lambda x: x[0])
            line_text = " ".join(w[4] for w in wl).strip()
            if line_text:
                lines.append(line_text)
        spatial_text = "\n".join(lines).strip()
        return spatial_text or page.get_text("text").strip()
    except Exception as e:
        logger.debug("Spatial text extraction fallback: %s", e)
        return page.get_text("text").strip()


def calculate_page_density(
    text: str, table_count: int = 0, table_row_count: int = 0
) -> tuple[float, bool]:
    """
    Calculate statistical information density for a page.
    Returns (density_score, is_front_matter).
    Domain-agnostic: relies on digit density, table lines, and metric tokens.
    """
    if not text or len(text.strip()) < 50:
        return 0.0, False

    import re

    lines = text.strip().split("\n")
    header_sample = "\n".join(lines[:6]).lower()
    is_fm = any(
        k in header_sample
        for k in [
            "table of contents",
            "contents\t",
            "contents\n",
            "abbreviations",
            "list of tables",
            "list of boxes",
            "preface",
            "acknowledgement",
        ]
    )
    if text.count("...") > 8 or text.count(". . .") > 4:
        is_fm = True

    tokens = text.split()
    if not tokens:
        return 0.0, is_fm

    # 1. Number / digit token ratio
    num_digits = sum(1 for t in tokens if re.search(r"\d", t))
    digit_ratio = num_digits / len(tokens)

    # 2. Table-like lines: lines containing 2+ distinct numbers or percentages
    table_lines = sum(
        1 for l in lines if len(re.findall(r"\b\d+(?:\.\d+)?%?\b", l)) >= 2
    )
    table_line_ratio = table_lines / max(1, len(lines))

    # 3. Key quantitative/financial metric tokens
    metric_count = len(
        re.findall(
            r"(%|₹|\$|€|crore|million|billion|trillion|ratio|bps|growth|deficit|revenue|profit|loss|gdp|inflation|pat|ebitda|cpi)",
            text,
            re.I,
        )
    )
    metric_ratio = metric_count / len(tokens)

    # Composite density score (0 - 100)
    core_metrics = len(
        re.findall(
            r"\b(revenue|profit|loss|pat|pbt|ebitda|gdp|gva|cpi|wpi|inflation|growth|deficit|net worth|assets|liabilities|borrowings)\b",
            text,
            re.I,
        )
    )
    score = (
        (digit_ratio * 35.0)
        + (table_line_ratio * 30.0)
        + (metric_ratio * 20.0)
        + min(15.0, table_count * 5.0)
        + min(10.0, table_row_count * 0.5)
        + min(25.0, core_metrics * 2.5)
    )

    if is_fm:
        score *= 0.05

    return round(score, 2), is_fm


def parse_pdf(
    file_bytes: bytes,
    doc_name: str,
    doc_id: str | None = None,
    max_pages: int | None = None,
) -> list[PageChunk]:
    """
    Parse a PDF file and extract text page-by-page with density metrics.
    """
    doc_id = doc_id or str(uuid.uuid4())
    chunks: list[PageChunk] = []

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error("Failed to open PDF '%s': %s", doc_name, e)
        raise ValueError(f"Could not open PDF '{doc_name}': {e}") from e

    total_pages = len(doc)
    pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
    logger.info("Parsing PDF '%s' (%d of %d pages)", doc_name, pages_to_process, total_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]

        # Extract text preserving spatial layout and horizontal alignment
        text = extract_page_text(page)

        if not text:
            logger.debug(
                "Page %d of '%s' has no extractable text (may be scanned/image)",
                page_num + 1,
                doc_name,
            )
            continue

        # Detect structured grid tables on this page
        table_data: list[list[list[Optional[str]]]] = []
        table_row_count = 0
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for tab in tabs.tables:
                    rows = tab.extract()
                    if rows:
                        table_data.append(rows)
                        table_row_count += len(rows)
        except Exception as e:
            logger.debug("Table detection failed on page %d: %s", page_num + 1, e)

        density_score, is_fm = calculate_page_density(text, len(table_data), table_row_count)

        # Use 1-based page numbering (as displayed in PDF viewers)
        chunk = PageChunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=page_num + 1,
            text=text,
            tables=table_data if table_data else None,
            table_row_count=table_row_count,
            density_score=density_score,
            is_front_matter=is_fm,
        )
        chunks.append(chunk)

    doc.close()

    tables_found = sum(1 for c in chunks if c.tables)
    logger.info(
        "Extracted text from %d/%d pages of '%s' (%d pages with tables)",
        len(chunks),
        total_pages,
        doc_name,
        tables_found,
    )

    return chunks


def get_high_signal_chunks(chunks: list[PageChunk], max_chunks: int = 6) -> list[PageChunk]:
    """
    Select the highest-signal data pages (tables & quantitative summaries).
    Returns chunks in original ascending page order.
    """
    # Filter out blank pages
    valid = [c for c in chunks if len(c.text.strip()) >= 60]
    if len(valid) <= max_chunks:
        return valid

    # Separate non-front-matter from front-matter
    content_chunks = [c for c in valid if not c.is_front_matter]
    candidates = content_chunks if len(content_chunks) >= max_chunks else valid

    # Sort by density score descending and take top N
    top_ranked = sorted(candidates, key=lambda c: c.density_score, reverse=True)[:max_chunks]

    # Re-sort in original page order for logical coherence
    return sorted(top_ranked, key=lambda c: c.page_number)


def get_page_count(file_bytes: bytes) -> int:
    """Get the total number of pages in a PDF without full parsing."""
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0
