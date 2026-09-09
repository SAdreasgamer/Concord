"""
SQLite database layer for Concord.

Uses aiosqlite for async operations. Schema is created on startup.
All IDs are UUIDs generated in Python.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from backend.config import DB_PATH


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
    agreement_strength  REAL NOT NULL DEFAULT 1.0,
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relationships_fact1
    ON relationships(fact_id_1);
CREATE INDEX IF NOT EXISTS idx_relationships_fact2
    ON relationships(fact_id_2);
CREATE INDEX IF NOT EXISTS idx_relationships_type
    ON relationships(relation_type);
"""


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # --- Documents ---

    async def insert_document(self, doc_id: str, doc_name: str, page_count: int) -> None:
        await self.db.execute(
            "INSERT INTO documents (doc_id, doc_name, page_count, created_at) VALUES (?, ?, ?, ?)",
            (doc_id, doc_name, page_count, _now_iso()),
        )
        await self.db.commit()

    async def get_documents(self) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT d.doc_id, d.doc_name, d.page_count, d.created_at, "
            "  (SELECT COUNT(*) FROM facts f WHERE f.source_doc_id = d.doc_id) AS fact_count "
            "FROM documents d ORDER BY d.created_at DESC"
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_document(self, doc_id: str) -> Optional[dict]:
        cursor = await self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def delete_document(self, doc_id: str) -> bool:
        doc = await self.get_document(doc_id)
        if not doc:
            return False
        await self.db.execute(
            "DELETE FROM relationships WHERE fact_id_1 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "   OR fact_id_2 IN (SELECT id FROM facts WHERE source_doc_id = ?)",
            (doc_id, doc_id),
        )
        await self.db.execute("DELETE FROM facts WHERE source_doc_id = ?", (doc_id,))
        await self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        await self.db.commit()
        return True

    async def clear_all_data(self) -> None:
        await self.db.execute("DELETE FROM relationships")
        await self.db.execute("DELETE FROM facts")
        await self.db.execute("DELETE FROM documents")
        await self.db.commit()

    # --- Facts ---

    async def insert_fact(self, fact: dict) -> str:
        fact_id = fact.get("id") or str(uuid.uuid4())
        await self.db.execute(
            "INSERT INTO facts "
            "(id, subject, subject_normalized, attribute, attribute_normalized, "
            " value, unit, temporal_scope, conditions, claim_fingerprint, "
            " extraction_group_id, source_doc, source_doc_id, page, "
            " evidence_quote, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                fact["subject"], fact["subject_normalized"],
                fact["attribute"], fact["attribute_normalized"],
                fact["value"], fact.get("unit"),
                fact.get("temporal_scope"), fact.get("conditions"),
                fact["claim_fingerprint"], fact.get("extraction_group_id"),
                fact["source_doc"], fact["source_doc_id"],
                fact["page"], fact["evidence_quote"],
                fact.get("confidence", 0.8), _now_iso(),
            ),
        )
        await self.db.commit()
        return fact_id

    async def get_facts(self, source_doc_id: Optional[str] = None, limit: int = 500) -> list[dict]:
        if source_doc_id:
            cursor = await self.db.execute(
                "SELECT * FROM facts WHERE source_doc_id = ? ORDER BY page, created_at LIMIT ?",
                (source_doc_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM facts ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_fact(self, fact_id: str) -> Optional[dict]:
        cursor = await self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    # --- Relationships ---

    async def insert_relationship(self, rel: Any) -> str:
        data = rel.model_dump() if hasattr(rel, "model_dump") else rel
        rel_id = data.get("id") or str(uuid.uuid4())
        r_type = data["relation_type"].value if hasattr(data["relation_type"], "value") else str(data["relation_type"])

        r_type = r_type.lower().replace("relationtype.", "")
        r_factor = data.get("reconciling_factor", "none")
        r_factor = r_factor.value if hasattr(r_factor, "value") else str(r_factor)
        r_factor = r_factor.lower().replace("reconcilingfactor.", "")
        m_source = data.get("match_source", "structural")
        m_source = m_source.value if hasattr(m_source, "value") else str(m_source)

        await self.db.execute(
            "INSERT INTO relationships "
            "(id, fact_id_1, fact_id_2, relation_type, reconciling_factor, "
            " explanation, match_source, is_intra_document, agreement_strength, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel_id, data["fact_id_1"], data["fact_id_2"],
                r_type, r_factor,
                data["explanation"], m_source,
                1 if data.get("is_intra_document") else 0,
                float(data.get("agreement_strength", 1.0)), _now_iso(),
            ),
        )

        await self.db.commit()
        return rel_id


    async def get_relationships(self, relation_type: Optional[str] = None, limit: int = 500) -> list[dict]:
        if relation_type:
            cursor = await self.db.execute(
                "SELECT * FROM relationships WHERE relation_type = ? ORDER BY created_at DESC LIMIT ?",
                (relation_type, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM relationships ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_relationships_for_fact(self, fact_id: str) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM relationships WHERE fact_id_1 = ? OR fact_id_2 = ? ORDER BY created_at DESC",
            (fact_id, fact_id),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_relationship(self, rel_id: str) -> Optional[dict]:
        cursor = await self.db.execute("SELECT * FROM relationships WHERE id = ?", (rel_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_relationships_for_document(self, doc_id: str) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT r.* FROM relationships r "
            "WHERE r.fact_id_1 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "   OR r.fact_id_2 IN (SELECT id FROM facts WHERE source_doc_id = ?) "
            "ORDER BY r.created_at DESC",
            (doc_id, doc_id),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def relationship_exists(self, fact_id_1: str, fact_id_2: str) -> bool:
        cursor = await self.db.execute(
            "SELECT 1 FROM relationships "
            "WHERE (fact_id_1 = ? AND fact_id_2 = ?) OR (fact_id_1 = ? AND fact_id_2 = ?) LIMIT 1",
            (fact_id_1, fact_id_2, fact_id_2, fact_id_1),
        )
        return (await cursor.fetchone()) is not None

    # --- Stats ---

    async def get_stats(self) -> dict:
        docs = await self.db.execute("SELECT COUNT(*) as c FROM documents")
        facts = await self.db.execute("SELECT COUNT(*) as c FROM facts")
        rels = await self.db.execute("SELECT COUNT(*) as c FROM relationships")
        return {
            "documents": (await docs.fetchone())["c"],
            "facts": (await facts.fetchone())["c"],
            "relationships": (await rels.fetchone())["c"],
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
