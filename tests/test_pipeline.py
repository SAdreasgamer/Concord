"""
Smoke tests for the Concord pipeline.

Tests will be expanded as each phase is implemented.
"""

import pytest


def test_imports():
    """Verify all backend modules can be imported."""
    from backend import config
    from backend import models
    from backend import database


def test_config_paths():
    """Verify config paths are relative to project root, not hardcoded."""
    from backend.config import PROJECT_ROOT, UPLOAD_DIR, DB_PATH

    assert PROJECT_ROOT.exists()
    assert str(UPLOAD_DIR).startswith(str(PROJECT_ROOT))
    assert str(DB_PATH).startswith(str(PROJECT_ROOT))


def test_models():
    """Verify Pydantic models instantiate correctly."""
    from backend.models import ExtractedFact, Fact, Relationship, RelationType

    fact = ExtractedFact(
        subject="Test Corp",
        subject_normalized="test_corp",
        attribute="revenue",
        attribute_normalized="revenue",
        value="100",
        unit="million USD",
        temporal_scope="FY2023",
        evidence_quote="Revenue was 100 million USD in FY2023.",
        page=1,
        confidence=0.9,
    )
    assert fact.subject_normalized == "test_corp"
    assert fact.confidence == 0.9


def test_pdf_parser():
    """Verify PDF parser extracts text with correct page numbers."""
    from pathlib import Path

    from backend.config import PROJECT_ROOT
    from backend.pdf_parser import parse_pdf, get_page_count

    # Use the smallest starter PDF for testing
    pdf_path = (
        PROJECT_ROOT
        / "starter-datasets"
        / "delhivery"
        / "03-delhivery-q4-fy24-earnings-presentation.pdf"
    )
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"

    file_bytes = pdf_path.read_bytes()

    # Test page count
    page_count = get_page_count(file_bytes)
    assert page_count == 27

    # Test full parsing
    chunks = parse_pdf(file_bytes, doc_name="test.pdf", doc_id="test-123")
    assert len(chunks) > 0
    assert all(c.doc_id == "test-123" for c in chunks)
    assert all(c.doc_name == "test.pdf" for c in chunks)
    assert all(c.page_number >= 1 for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)

    # Page numbers should be 1-based and sequential (for pages with text)
    page_numbers = [c.page_number for c in chunks]
    assert page_numbers == sorted(page_numbers)
