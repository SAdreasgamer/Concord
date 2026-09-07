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


def parse_pdf(
    file_bytes: bytes,
    doc_name: str,
    doc_id: str | None = None,
) -> list[PageChunk]:
    """
    Parse a PDF file and extract text page-by-page.

    Args:
        file_bytes: Raw PDF file content.
        doc_name: Original filename of the document.
        doc_id: Optional document ID. Generated if not provided.

    Returns:
        List of PageChunk objects, one per page with extractable text.
    """
    doc_id = doc_id or str(uuid.uuid4())
    chunks: list[PageChunk] = []

    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error("Failed to open PDF '%s': %s", doc_name, e)
        raise ValueError(f"Could not open PDF '{doc_name}': {e}") from e

    total_pages = len(doc)
    logger.info("Parsing PDF '%s' (%d pages)", doc_name, total_pages)

    for page_num in range(total_pages):
        page = doc[page_num]

        # Extract text — use "text" mode for clean output
        text = page.get_text("text").strip()

        if not text:
            logger.debug(
                "Page %d of '%s' has no extractable text (may be scanned/image)",
                page_num + 1,
                doc_name,
            )
            continue

        # Detect structured tables on this page
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

        # Use 1-based page numbering (as displayed in PDF viewers)
        chunk = PageChunk(
            doc_id=doc_id,
            doc_name=doc_name,
            page_number=page_num + 1,
            text=text,
            tables=table_data if table_data else None,
            table_row_count=table_row_count,
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


def get_page_count(file_bytes: bytes) -> int:
    """Get the total number of pages in a PDF without full parsing."""
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0
