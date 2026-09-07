"""
SQLite database layer for Concord.

Uses aiosqlite for async operations. Schema is created on startup.
All IDs are UUIDs generated in Python — no auto-increment, no DB-specific features.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite
import numpy as np

from backend.config import DB_PATH

# --- Schema ---

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    doc_name     TEXT NOT NULL,
    page_count   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id                    TEXT PRIMARY KEY,
    subject               TEXT NOT NULL,
    subject_normalized    TEXT NOT NULL,
    attribute             TEXT NOT NULL,
    attribute_normalized  TEXT NOT NULL,
    value                 TEXT NOT NULL,
    unit                  TEXT,
    temporal_scope        TEXT,
    conditions            TEXT,
    claim_fingerprint     TEXT NOT NULL,
    extraction_group_id   TEXT,
    source_doc            TEXT NOT NULL,
    source_doc_id         TEXT NOT NULL,
    page                  INTEGER NOT NULL,
    evidence_quote        TEXT NOT NULL,
    confidence            REAL NOT NULL DEFAULT 0.8,
    embedding             BLOB,
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_fingerprint
    ON facts(claim_fingerprint);
CREATE INDEX IF NOT EXISTS idx_facts_subject_norm
    ON facts(subject_normalized);
CREATE INDEX IF NOT EXISTS idx_facts_attribute_norm
    ON facts(attribute_normalized);
CREATE INDEX IF NOT EXISTS idx_facts_source_doc_id
    ON facts(source_doc_id);

CREATE TABLE IF NOT EXISTS relationships (
    id                  TEXT PRIMARY KEY,
    fact_id_1           TEXT NOT NULL REFERENCES facts(id),
    fact_id_2           TEXT NOT NULL REFERENCES facts(id),
    relation_type       TEXT NOT NULL,
    reconciling_factor  TEXT NOT NULL DEFAULT 'none',
    explanation         TEXT NOT NULL,
    match_source        TEXT NOT NULL,
    is_intra_document   INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relationships_fact1
    ON relationships(fact_id_1);
CREATE INDEX IF NOT EXISTS idx_relationships_fact2
    ON relationships(fact_id_2);
CREATE INDEX IF NOT EXISTS idx_relationships_type
    ON relationships(relation_type);

CREATE TABLE IF NOT EXISTS normalization_registry (
    id              TEXT PRIMARY KEY,
    key_type        TEXT NOT NULL,
    raw_value       TEXT NOT NULL,
    canonical_value TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_norm_key_type
    ON normalization_registry(key_type);
CREATE INDEX IF NOT EXISTS idx_norm_canonical
    ON normalization_registry(canonical_value);
"""


# --- Database Manager ---


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # --- Documents ---

    async def insert_document(
        self, doc_id: str, doc_name: str, page_count: int
    ) -> None:
        await self.db.execute(
            "INSERT INTO documents (doc_id, doc_name, page_count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, doc_name, page_count, _now_iso()),
        )
        await self.db.commit()

    async def get_documents(self) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT d.doc_id, d.doc_name, d.page_count, d.created_at, "
            "  (SELECT COUNT(*) FROM facts f WHERE f.source_doc_id = d.doc_id) AS fact_count "
            "FROM documents d ORDER BY d.created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_document(self, doc_id: str) -> Optional[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and cascade delete its facts and relationships."""
        doc = await self.get_document(doc_id)
        if not doc:
            return False

        # 1. Delete relationships referencing facts from this document
        await self.db.execute(
            "DELETE FROM relationships WHERE fact_id_1 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "   OR fact_id_2 IN (SELECT id FROM facts WHERE source_doc_id = ?)",
            (doc_id, doc_id),
        )

        # 2. Delete facts from this document
        await self.db.execute("DELETE FROM facts WHERE source_doc_id = ?", (doc_id,))

        # 3. Delete document record
        await self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await self.db.commit()
        return True

    # --- Facts ---

    async def insert_fact(self, fact: dict, embedding: Optional[bytes] = None) -> str:
        fact_id = fact.get("id") or str(uuid.uuid4())
        await self.db.execute(
            "INSERT INTO facts "
            "(id, subject, subject_normalized, attribute, attribute_normalized, "
            " value, unit, temporal_scope, conditions, claim_fingerprint, "
            " extraction_group_id, source_doc, source_doc_id, page, "
            " evidence_quote, confidence, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                fact["subject"],
                fact["subject_normalized"],
                fact["attribute"],
                fact["attribute_normalized"],
                fact["value"],
                fact.get("unit"),
                fact.get("temporal_scope"),
                fact.get("conditions"),
                fact["claim_fingerprint"],
                fact.get("extraction_group_id"),
                fact["source_doc"],
                fact["source_doc_id"],
                fact["page"],
                fact["evidence_quote"],
                fact.get("confidence", 0.8),
                embedding,
                _now_iso(),
            ),
        )
        await self.db.commit()
        return fact_id

    async def get_facts(
        self, source_doc_id: Optional[str] = None, limit: int = 500
    ) -> list[dict]:
        if source_doc_id:
            cursor = await self.db.execute(
                "SELECT * FROM facts WHERE source_doc_id = ? "
                "ORDER BY page, created_at LIMIT ?",
                (source_doc_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM facts ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_fact(self, fact_id: str) -> Optional[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_facts_by_fingerprint_prefix(
        self, subject_normalized: str, attribute_normalized: str
    ) -> list[dict]:
        """Find facts sharing the same (subject, attribute) normalized keys."""
        cursor = await self.db.execute(
            "SELECT * FROM facts "
            "WHERE subject_normalized = ? AND attribute_normalized = ?",
            (subject_normalized, attribute_normalized),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_all_fact_embeddings(
        self, exclude_doc_id: Optional[str] = None
    ) -> list[tuple[str, np.ndarray]]:
        """Get all fact IDs and their embeddings for similarity search."""
        if exclude_doc_id:
            cursor = await self.db.execute(
                "SELECT id, embedding FROM facts "
                "WHERE embedding IS NOT NULL AND source_doc_id != ?",
                (exclude_doc_id,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT id, embedding FROM facts WHERE embedding IS NOT NULL"
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            results.append((row["id"], emb))
        return results

    async def update_fact_embedding(self, fact_id: str, embedding: bytes) -> None:
        """Update the embedding BLOB for an existing fact."""
        await self.db.execute(
            "UPDATE facts SET embedding = ? WHERE id = ?",
            (embedding, fact_id),
        )
        await self.db.commit()

    # --- Relationships ---

    async def insert_relationship(self, rel: dict) -> str:
        rel_id = rel.get("id") or str(uuid.uuid4())
        await self.db.execute(
            "INSERT INTO relationships "
            "(id, fact_id_1, fact_id_2, relation_type, reconciling_factor, "
            " explanation, match_source, is_intra_document, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel_id,
                rel["fact_id_1"],
                rel["fact_id_2"],
                rel["relation_type"],
                rel.get("reconciling_factor", "none"),
                rel["explanation"],
                rel["match_source"],
                1 if rel.get("is_intra_document") else 0,
                _now_iso(),
            ),
        )
        await self.db.commit()
        return rel_id

    async def get_relationships(
        self, relation_type: Optional[str] = None, limit: int = 500
    ) -> list[dict]:
        if relation_type:
            cursor = await self.db.execute(
                "SELECT * FROM relationships WHERE relation_type = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (relation_type, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM relationships ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_relationships_for_fact(self, fact_id: str) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM relationships "
            "WHERE fact_id_1 = ? OR fact_id_2 = ? "
            "ORDER BY created_at DESC",
            (fact_id, fact_id),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_relationship(self, rel_id: str) -> Optional[dict]:
        """Fetch a single relationship by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM relationships WHERE id = ?", (rel_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_relationships_for_document(self, doc_id: str) -> list[dict]:
        """Get all relationships where at least one fact belongs to this document."""
        cursor = await self.db.execute(
            "SELECT r.* FROM relationships r "
            "WHERE r.fact_id_1 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "   OR r.fact_id_2 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "ORDER BY r.created_at DESC",
            (doc_id, doc_id),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def relationship_exists(self, fact_id_1: str, fact_id_2: str) -> bool:
        """Check if a relationship already exists between two facts (in either direction)."""
        cursor = await self.db.execute(
            "SELECT 1 FROM relationships "
            "WHERE (fact_id_1 = ? AND fact_id_2 = ?) "
            "   OR (fact_id_1 = ? AND fact_id_2 = ?) "
            "LIMIT 1",
            (fact_id_1, fact_id_2, fact_id_2, fact_id_1),
        )
        row = await cursor.fetchone()
        return row is not None

    # --- Normalization Registry ---

    async def get_canonical_values(self, key_type: str) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT DISTINCT canonical_value FROM normalization_registry "
            "WHERE key_type = ?",
            (key_type,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def insert_normalization_entry(
        self, key_type: str, raw_value: str, canonical_value: str
    ) -> str:
        entry_id = str(uuid.uuid4())
        await self.db.execute(
            "INSERT INTO normalization_registry "
            "(id, key_type, raw_value, canonical_value, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry_id, key_type, raw_value, canonical_value, _now_iso()),
        )
        await self.db.commit()
        return entry_id

    # --- Stats ---

    async def get_stats(self) -> dict:
        """Get overall statistics."""
        docs = await self.db.execute("SELECT COUNT(*) as c FROM documents")
        facts = await self.db.execute("SELECT COUNT(*) as c FROM facts")
        rels = await self.db.execute("SELECT COUNT(*) as c FROM relationships")
        docs_count = (await docs.fetchone())["c"]
        facts_count = (await facts.fetchone())["c"]
        rels_count = (await rels.fetchone())["c"]
        return {
            "documents": docs_count,
            "facts": facts_count,
            "relationships": rels_count,
        }


def _now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
