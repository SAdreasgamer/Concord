"""
LLM relation judging module.

Classifies candidate fact pairs as corroborates, contradicts, or reconciled,
and provides detailed evidence-grounded explanations with explicit reconciling factors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Optional

import litellm

from backend.config import DEFAULT_LLM_MODEL
from backend.database import Database
from backend.models import (
    CandidatePair,
    ReconcilingFactor,
    Relationship,
    RelationType,
)
from backend.prompts import (
    RELATION_JUDGE_BATCH_USER_PROMPT_TEMPLATE,
    RELATION_JUDGE_SYSTEM_PROMPT,
    RELATION_JUDGE_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def normalize_relation_type(raw_type: str) -> RelationType:
    """Normalize raw string to valid RelationType enum."""
    t = (raw_type or "").strip().lower()
    if "corr" in t:
        return RelationType.CORROBORATES
    elif "cont" in t:
        return RelationType.CONTRADICTS
    elif "rec" in t:
        return RelationType.RECONCILED
    return RelationType.CORROBORATES


def normalize_reconciling_factor(raw_factor: str, rel_type: RelationType) -> ReconcilingFactor:
    """Normalize raw string to valid ReconcilingFactor enum."""
    if rel_type != RelationType.RECONCILED:
        return ReconcilingFactor.NONE

    f = (raw_factor or "").strip().lower()
    if "temp" in f or "time" in f or "period" in f or "year" in f or "quarter" in f:
        return ReconcilingFactor.TEMPORAL_SCOPE
    elif "entity" in f or "scope" in f or "subsidiary" in f or "parent" in f:
        return ReconcilingFactor.ENTITY_SCOPE
    elif "unit" in f or "curr" in f or "scale" in f:
        return ReconcilingFactor.UNIT_DIFFERENCE
    elif "def" in f or "gaap" in f or "statutory" in f or "adjusted" in f:
        return ReconcilingFactor.DEFINITION_DIFFERENCE
    return ReconcilingFactor.NONE


def parse_judgment_dict(data: dict[str, Any], pair: CandidatePair) -> Relationship:
    """Convert parsed JSON dict into a validated Relationship model."""
    raw_rel = data.get("relation_type", "corroborates")
    raw_factor = data.get("reconciling_factor", "none")
    explanation = data.get("explanation", "Relationship evaluated by judge.")

    rel_type = normalize_relation_type(raw_rel)
    reconciling_factor = normalize_reconciling_factor(raw_factor, rel_type)

    c1 = getattr(pair.fact_1, "confidence", 1.0) or 1.0
    c2 = getattr(pair.fact_2, "confidence", 1.0) or 1.0
    strength = round(min(float(c1), float(c2)), 2)

    return Relationship(
        id=str(uuid.uuid4()),
        fact_id_1=pair.fact_1.id,
        fact_id_2=pair.fact_2.id,
        relation_type=rel_type,
        reconciling_factor=reconciling_factor,
        explanation=str(explanation).strip(),
        match_source=pair.match_source,
        is_intra_document=pair.is_intra_document,
        agreement_strength=strength,
    )


def extract_json_from_llm(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response text."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError(f"Could not parse valid JSON from LLM response: {cleaned[:200]}")


def format_single_pair_prompt(pair: CandidatePair) -> str:
    """Format single pair prompt using template."""
    f1 = pair.fact_1
    f2 = pair.fact_2

    return RELATION_JUDGE_USER_PROMPT_TEMPLATE.format(
        doc_1=f1.source_doc,
        page_1=f1.page,
        subject_1=f1.subject,
        subject_norm_1=f1.subject_normalized,
        attribute_1=f1.attribute,
        attribute_norm_1=f1.attribute_normalized,
        value_1=f1.value,
        unit_1=f1.unit or "",
        temporal_scope_1=f1.temporal_scope or "unspecified",
        conditions_1=f1.conditions or "none",
        evidence_quote_1=f1.evidence_quote,
        doc_2=f2.source_doc,
        page_2=f2.page,
        subject_2=f2.subject,
        subject_norm_2=f2.subject_normalized,
        attribute_2=f2.attribute,
        attribute_norm_2=f2.attribute_normalized,
        value_2=f2.value,
        unit_2=f2.unit or "",
        temporal_scope_2=f2.temporal_scope or "unspecified",
        conditions_2=f2.conditions or "none",
        evidence_quote_2=f2.evidence_quote,
        match_hint=pair.match_hint or "none",
    )


def format_batch_pairs_prompt(pairs: list[CandidatePair]) -> str:
    """Format multiple pairs into a single prompt for batch judging."""
    blocks = []
    for idx, pair in enumerate(pairs, start=1):
        f1 = pair.fact_1
        f2 = pair.fact_2
        block = f"""=== PAIR {idx} ===
Candidate Match Hint: {pair.match_hint or 'none'}
FACT 1:
  Source: {f1.source_doc} (p. {f1.page})
  Subject: {f1.subject} [{f1.subject_normalized}]
  Attribute: {f1.attribute} [{f1.attribute_normalized}]
  Value: {f1.value} {f1.unit or ''}
  Scope: {f1.temporal_scope or 'unspecified'} | Conditions: {f1.conditions or 'none'}
  Evidence: "{f1.evidence_quote}"

FACT 2:
  Source: {f2.source_doc} (p. {f2.page})
  Subject: {f2.subject} [{f2.subject_normalized}]
  Attribute: {f2.attribute} [{f2.attribute_normalized}]
  Value: {f2.value} {f2.unit or ''}
  Scope: {f2.temporal_scope or 'unspecified'} | Conditions: {f2.conditions or 'none'}
  Evidence: "{f2.evidence_quote}"
"""
        blocks.append(block)

    pairs_text = "\n".join(blocks)
    return RELATION_JUDGE_BATCH_USER_PROMPT_TEMPLATE.format(
        count=len(pairs),
        pairs_text=pairs_text,
    )


async def judge_single_pair(
    pair: CandidatePair,
    api_key: str,
    model: Optional[str] = None,
) -> Relationship:
    """Judge a single candidate pair via LLM call."""
    target_model = model or DEFAULT_LLM_MODEL
    prompt = format_single_pair_prompt(pair)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": RELATION_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
            data = extract_json_from_llm(raw_text)
            return parse_judgment_dict(data, pair)
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                isinstance(e, litellm.RateLimitError)
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1 and "quota" not in err_msg:
                delay = 2.0 * (attempt + 1)
                logger.warning(
                    "Rate limit in relation judge (attempt %d/%d). Backing off for %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise


async def judge_batch_pairs(
    pairs: list[CandidatePair],
    api_key: str,
    model: Optional[str] = None,
) -> list[Relationship]:
    """Judge a batch of candidate pairs in a single LLM call."""
    if not pairs:
        return []
    if len(pairs) == 1:
        return [await judge_single_pair(pairs[0], api_key, model)]

    target_model = model or DEFAULT_LLM_MODEL
    prompt = format_batch_pairs_prompt(pairs)

    batch_max_tokens = 1200 if "groq" in target_model.lower() else 3000
    judgments_raw = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=target_model,
                messages=[
                    {"role": "system", "content": RELATION_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                temperature=0.1,
                max_tokens=batch_max_tokens,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
            data = extract_json_from_llm(raw_text)
            judgments_raw = data.get("judgments", [])
            break
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                isinstance(e, litellm.RateLimitError)
                or "429" in err_msg
                or "quota" in err_msg
                or "resource_exhausted" in err_msg
                or "rate limit" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = 2.5 * (attempt + 1)
                m = re.search(r"try again in ([\d\.]+)s", err_msg)
                if m:
                    delay = max(delay, float(m.group(1)) + 1.0)
                logger.warning(
                    "Rate limit in batch judge (attempt %d/%d). Backing off for %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.warning("Batch judging attempt failed (%s)", e)
                if is_rate_limit:
                    # Do not fall back to per-pair calls if rate limited or quota exhausted
                    return []
                break

    if judgments_raw is not None:
        # Map judgments by pair index
        results: list[Relationship] = []
        for idx, pair in enumerate(pairs, start=1):
            matching_j = next(
                (j for j in judgments_raw if j.get("pair_index") == idx),
                None,
            )
            if matching_j:
                results.append(parse_judgment_dict(matching_j, pair))
            else:
                try:
                    single_rel = await judge_single_pair(pair, api_key, model)
                    results.append(single_rel)
                except Exception:
                    pass
        return results

    # Fallback to individual pair evaluations ONLY for format/JSON errors
    logger.warning("Batch judging format failed, falling back to individual pair evaluations")
    results = []
    for pair in pairs:
        try:
            rel = await judge_single_pair(pair, api_key, model)
            results.append(rel)
        except Exception as inner_e:
            logger.error("Failed to judge pair %s vs %s: %s", pair.fact_1.id, pair.fact_2.id, inner_e)
            if "quota" in str(inner_e).lower() or "resource_exhausted" in str(inner_e).lower():
                break
    return results


def try_deterministic_corroboration(pair: CandidatePair) -> Optional[Relationship]:
    """
    If two facts share identical claim fingerprint (same entity, attribute, temporal scope)
    and identical value and unit, resolve corroboration deterministically with ZERO API calls.
    """
    f1 = pair.fact_1
    f2 = pair.fact_2

    if f1.claim_fingerprint and f2.claim_fingerprint and f1.claim_fingerprint == f2.claim_fingerprint:
        v1 = f1.value.strip().lower().replace(",", "")
        v2 = f2.value.strip().lower().replace(",", "")
        u1 = (f1.unit or "").strip().lower()
        u2 = (f2.unit or "").strip().lower()

        if v1 == v2 and (not u1 or not u2 or u1 == u2):
            strength = round(min(float(f1.confidence), float(f2.confidence)), 2)
            return Relationship(
                id=str(uuid.uuid4()),
                fact_id_1=f1.id,
                fact_id_2=f2.id,
                relation_type=RelationType.CORROBORATES,
                reconciling_factor=ReconcilingFactor.NONE,
                explanation=f"Deterministic match: both sources independently verify {f1.subject} {f1.attribute} as {f1.value} {f1.unit or ''} for {f1.temporal_scope or 'unspecified period'}.",
                match_source=pair.match_source,
                is_intra_document=pair.is_intra_document,
                agreement_strength=strength,
            )
    return None


async def judge_and_store_candidates(
    candidates: list[CandidatePair],
    db: Database,
    api_key: str,
    model: Optional[str] = None,
    batch_size: int = 8,
) -> list[Relationship]:
    """
    Judge all candidate pairs and persist them to the database.
    Resolves identical facts deterministically with 0 API calls, and batches remaining pairs.
    """
    if not candidates:
        return []

    judged_relationships: list[Relationship] = []

    # Step 1: Deterministic resolution for exact fingerprint & value matches (0 API calls)
    pairs_for_llm: list[CandidatePair] = []
    for pair in candidates:
        deterministic_rel = try_deterministic_corroboration(pair)
        if deterministic_rel:
            if not await db.relationship_exists(deterministic_rel.fact_id_1, deterministic_rel.fact_id_2):
                await db.insert_relationship(deterministic_rel.model_dump())
                judged_relationships.append(deterministic_rel)
        else:
            pairs_for_llm.append(pair)

    logger.info(
        "Candidate judging: %d pairs resolved deterministically (0 API calls), %d sent to LLM",
        len(judged_relationships),
        len(pairs_for_llm),
    )

    # Step 2: Batch remaining pairs to LLM
    effective_batch_size = max(batch_size, 8)
    for i in range(0, len(pairs_for_llm), effective_batch_size):
        batch = pairs_for_llm[i : i + effective_batch_size]
        batch_results = await judge_batch_pairs(batch, api_key=api_key, model=model)
        if not batch_results and len(pairs_for_llm) > effective_batch_size:
            logger.warning("Judging batch returned no results (rate limit / quota). Halting further candidate evaluations.")
            break

        for rel in batch_results:
            if not await db.relationship_exists(rel.fact_id_1, rel.fact_id_2):
                await db.insert_relationship(rel.model_dump())
                judged_relationships.append(rel)

    logger.info(
        "Persisted %d judged relationships (%d corroborates, %d contradicts, %d reconciled)",
        len(judged_relationships),
        sum(1 for r in judged_relationships if r.relation_type == RelationType.CORROBORATES),
        sum(1 for r in judged_relationships if r.relation_type == RelationType.CONTRADICTS),
        sum(1 for r in judged_relationships if r.relation_type == RelationType.RECONCILED),
    )
    return judged_relationships
