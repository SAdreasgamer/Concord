"""
Relationship judging module.

Two-step approach:
1. Deterministic: if facts share the same fingerprint and same value → CORROBORATES (0 API calls)
2. LLM: send pair to LLM for judgment (corroborates / contradicts / reconciled)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Optional

import litellm

from backend.config import DEFAULT_LLM_MODEL, GROQ_API_KEY, OLLAMA_API_BASE
from backend.database import Database
from backend.models import (
    CandidatePair,
    ReconcilingFactor,
    Relationship,
    RelationType,
)
from backend.prompts import (
    RELATION_JUDGE_SYSTEM_PROMPT,
    RELATION_JUDGE_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# --- Helpers ---


def _normalize_relation_type(raw: str) -> RelationType:
    t = (raw or "").strip().lower()
    if "corr" in t:
        return RelationType.CORROBORATES
    elif "cont" in t:
        return RelationType.CONTRADICTS
    elif "rec" in t:
        return RelationType.RECONCILED
    return RelationType.CORROBORATES


def _normalize_factor(raw: str, rel_type: RelationType) -> ReconcilingFactor:
    if rel_type != RelationType.RECONCILED:
        return ReconcilingFactor.NONE
    f = (raw or "").strip().lower()
    if "proj" in f or "forecast" in f or "target" in f or "actual" in f:
        return ReconcilingFactor.PROJECTION_VS_ACTUAL
    elif "restate" in f or "basis" in f or "consol" in f or "standalone" in f:
        return ReconcilingFactor.REPORTING_BASIS
    elif "round" in f or "precis" in f or "decimal" in f:
        return ReconcilingFactor.PRECISION_OR_ROUNDING
    elif "temp" in f or "time" in f or "period" in f or "year" in f or "quarter" in f:
        return ReconcilingFactor.TEMPORAL_SCOPE
    elif "entity" in f or "scope" in f or "subsidiary" in f:
        return ReconcilingFactor.ENTITY_SCOPE
    elif "unit" in f or "curr" in f or "scale" in f:
        return ReconcilingFactor.UNIT_DIFFERENCE
    elif "def" in f or "gaap" in f or "adjusted" in f or "standard" in f:
        return ReconcilingFactor.DEFINITION_DIFFERENCE
    return ReconcilingFactor.NONE


def _extract_json(raw_text: str) -> dict[str, Any]:
    """Extract JSON from LLM response with repair for truncation."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try repairing truncated JSON
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        truncated = cleaned[: last_brace + 1].strip()
        for closer in ["]}", "}", "\"}"]:
            try:
                return json.loads(truncated + closer)
            except Exception:
                continue

    # Direct regex field extraction fallback
    rel_type_match = re.search(r'"relation_type"\s*:\s*"([^"]+)"', cleaned, re.I)
    rec_factor_match = re.search(r'"reconciling_factor"\s*:\s*"([^"]+)"', cleaned, re.I)
    exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)', cleaned, re.I)

    if rel_type_match:
        return {
            "relation_type": rel_type_match.group(1),
            "reconciling_factor": rec_factor_match.group(1) if rec_factor_match else "none",
            "explanation": exp_match.group(1) if exp_match else "Extracted relationship.",
        }

    raise ValueError(f"Could not parse JSON: {cleaned[:200]}")


# --- Deterministic Corroboration ---


def try_deterministic_corroboration(pair: CandidatePair) -> Optional[Relationship]:
    """
    If two facts share identical fingerprint and matching values → CORROBORATES instantly.
    Handles float rounding (e.g. -4157.43 vs -4157).
    """
    f1, f2 = pair.fact_1, pair.fact_2

    if not (f1.claim_fingerprint and f2.claim_fingerprint and f1.claim_fingerprint == f2.claim_fingerprint):
        return None

    v1 = f1.value.strip().lower().replace(",", "")
    v2 = f2.value.strip().lower().replace(",", "")
    u1 = (f1.unit or "").strip().lower()
    u2 = (f2.unit or "").strip().lower()

    # Check value equality (exact or float tolerance or Crore/Million conversion)
    is_equal = (v1 == v2)
    unit_ok = not u1 or not u2 or u1 == u2 or ("million" in u1 and "million" in u2) or ("crore" in u1 and "crore" in u2)

    if not is_equal:
        try:
            num1, num2 = float(v1), float(v2)
            if abs(num1 - num2) <= 1.0 or (abs(num1) > 0 and abs(num1 - num2) / abs(num1) < 0.005):
                is_equal = True
            # Check 1 Crore = 10 Million conversion
            elif ("million" in u1 or "mn" in u1) and ("crore" in u2 or "cr" in u2):
                converted = num1 / 10.0
                if abs(converted - num2) <= 1.0 or (num2 > 0 and abs(converted - num2) / num2 < 0.01):
                    is_equal = True
                    unit_ok = True
            elif ("crore" in u1 or "cr" in u1) and ("million" in u2 or "mn" in u2):
                converted = num2 / 10.0
                if abs(num1 - converted) <= 1.0 or (converted > 0 and abs(num1 - converted) / converted < 0.01):
                    is_equal = True
                    unit_ok = True
        except (ValueError, TypeError):
            pass

    if is_equal and unit_ok:

        strength = round(min(float(f1.confidence), float(f2.confidence)), 2)
        return Relationship(
            id=str(uuid.uuid4()),
            fact_id_1=f1.id,
            fact_id_2=f2.id,
            relation_type=RelationType.CORROBORATES,
            reconciling_factor=ReconcilingFactor.NONE,
            explanation=f"Deterministic corroboration: both sources report {f1.subject} {f1.attribute} = {f1.value} {f1.unit or ''} for {f1.temporal_scope or 'unspecified period'}.",
            match_source=pair.match_source,
            is_intra_document=pair.is_intra_document,
            agreement_strength=strength,
        )
    return None


# --- LLM Judging ---


async def judge_single_pair(
    pair: CandidatePair,
    api_key: str,
    model: Optional[str] = None,
) -> Optional[Relationship]:
    """Judge a single candidate pair via LLM."""
    target_model = model or DEFAULT_LLM_MODEL
    f1, f2 = pair.fact_1, pair.fact_2

    prompt = RELATION_JUDGE_USER_PROMPT_TEMPLATE.format(
        doc_1=f1.source_doc, page_1=f1.page,
        subject_1=f1.subject, subject_norm_1=f1.subject_normalized,
        attribute_1=f1.attribute, attribute_norm_1=f1.attribute_normalized,
        value_1=f1.value, unit_1=f1.unit or "",
        temporal_scope_1=f1.temporal_scope or "unspecified",
        conditions_1=f1.conditions or "none",
        evidence_quote_1=f1.evidence_quote,
        doc_2=f2.source_doc, page_2=f2.page,
        subject_2=f2.subject, subject_norm_2=f2.subject_normalized,
        attribute_2=f2.attribute, attribute_norm_2=f2.attribute_normalized,
        value_2=f2.value, unit_2=f2.unit or "",
        temporal_scope_2=f2.temporal_scope or "unspecified",
        conditions_2=f2.conditions or "none",
        evidence_quote_2=f2.evidence_quote,
        match_hint=pair.match_hint or "none",
    )

    max_retries = 4
    for attempt in range(max_retries):
        try:
            extra_kwargs = {}
            if target_model.startswith("ollama/"):
                extra_kwargs["api_base"] = OLLAMA_API_BASE

            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": RELATION_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_key=api_key if not target_model.startswith("ollama/") else None,
                temperature=0.1,
                max_tokens=150,
                timeout=25,
                **extra_kwargs,
            )
            raw_text = response.choices[0].message.content or ""
            data = _extract_json(raw_text)

            rel_type = _normalize_relation_type(data.get("relation_type", "corroborates"))
            factor = _normalize_factor(data.get("reconciling_factor", "none"), rel_type)
            explanation = str(data.get("explanation", "Relationship evaluated by judge.")).strip()
            strength = round(min(float(f1.confidence), float(f2.confidence)), 2)

            return Relationship(
                id=str(uuid.uuid4()),
                fact_id_1=f1.id,
                fact_id_2=f2.id,
                relation_type=rel_type,
                reconciling_factor=factor,
                explanation=explanation,
                match_source=pair.match_source,
                is_intra_document=pair.is_intra_document,
                agreement_strength=strength,
            )

        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg
            if is_rate_limit and attempt < max_retries - 1:
                delay = 3.0 * (attempt + 1)
                m = re.search(r"try again in ([\d\.]+)s", err_msg)
                if m:
                    delay = max(delay, float(m.group(1)) + 1.0)
                logger.warning("Rate limit judging pair (attempt %d/%d). Waiting %.1fs...", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
            else:
                logger.warning("Failed to judge pair (%s vs %s): %s", f1.attribute, f2.attribute, e)
                return None
    return None


# --- Pipeline ---


async def judge_and_store_candidates(
    candidates: list[CandidatePair],
    db: Database,
    api_key: str,
    model: Optional[str] = None,
) -> list[Relationship]:
    """
    Judge all candidates and store results.
    Step 1: Deterministic corroboration (0 API calls)
    Step 2: LLM judging for remaining pairs (1 API call each)
    """
    if not candidates:
        return []

    results: list[Relationship] = []
    pairs_for_llm: list[CandidatePair] = []

    # Step 1: Deterministic
    for pair in candidates:
        det = try_deterministic_corroboration(pair)
        if det:
            if not await db.relationship_exists(det.fact_id_1, det.fact_id_2):
                await db.insert_relationship(det.model_dump())
                results.append(det)
        else:
            pairs_for_llm.append(pair)

    logger.info("%d deterministic corroborations, %d sent to LLM", len(results), len(pairs_for_llm))

    # Step 2: LLM judging (one at a time, with pacing)
    for pair in pairs_for_llm:
        rel = await judge_single_pair(pair, api_key, model)
        if rel and not await db.relationship_exists(rel.fact_id_1, rel.fact_id_2):
            await db.insert_relationship(rel.model_dump())
            results.append(rel)

        target_model = model or DEFAULT_LLM_MODEL
        # Pace for rate limits
        if "groq" in target_model.lower():
            await asyncio.sleep(1.5)

    logger.info(
        "Total: %d relationships (%d corroborates, %d contradicts, %d reconciled)",
        len(results),
        sum(1 for r in results if r.relation_type == RelationType.CORROBORATES),
        sum(1 for r in results if r.relation_type == RelationType.CONTRADICTS),
        sum(1 for r in results if r.relation_type == RelationType.RECONCILED),
    )
    return results
