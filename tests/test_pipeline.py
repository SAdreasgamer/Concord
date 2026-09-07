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
