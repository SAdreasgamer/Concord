"""
LLM prompts for fact extraction and relation judging.

All prompts are generic and data-agnostic — no hardcoded document-specific schemas,
company names, or metrics. Works across financial reports, macroeconomic surveys,
scientific papers, and general prose.
"""

from __future__ import annotations

# --- Fact Extraction Prompts ---

FACT_EXTRACTION_SYSTEM_PROMPT = """You are an expert fact extraction engine. Extract atomic, grounded quantitative and factual claims from document text.

CRITICAL RULES:
1. GROUNDING: Every fact MUST include `evidence_quote` which is an EXACT, VERBATIM substring from the text. Never paraphrase.
2. ATOMIC: Extract discrete factual claims (numbers, percentages, dates, measurements).
3. METRIC DIVERSITY: For multi-row tables, extract distinct primary metrics (e.g. Revenue/Income, Profit/Loss, Net Worth, Borrowings, Growth rates).
4. FIELDS:
   - `subject`: Entity or organization name as written (e.g. "Delhivery", "Reserve Bank of India").
   - `attribute`: Metric or property as written (e.g. "Total income", "Restated loss", "Revenue from services").
   - `value`: Raw number or value only as string (e.g. "49114.06", "36355", "8.2").
   - `unit`: Unit of measurement (e.g. "₹ million", "%", "INR Crore"). Null if none.
   - `temporal_scope`: Relevant time period (e.g. "FY21", "2024", "December 31, 2021"). Null if none.
   - `conditions`: Qualifying context (e.g. "restated", "consolidated", "pro forma"). Null if none.

OUTPUT FORMAT: Return ONLY a valid JSON object:
{"facts": [{"subject": "...", "attribute": "...", "value": "...", "unit": "...", "temporal_scope": "...", "conditions": null, "evidence_quote": "exact quote from text", "confidence": 0.95}]}
If no facts found, return {"facts": []}."""

FACT_EXTRACTION_USER_PROMPT_TEMPLATE = """Document: {doc_name}
Page Number: {page_number}

--- PAGE TEXT ---
{page_text}
--- END PAGE TEXT ---

Extract at most 1 fact per distinct metric row across different metrics (e.g. revenue/income, profit/loss, net worth, output, key rates) from the page above as JSON. Ensure every fact has an exact verbatim evidence_quote from the text."""


# --- Relation Judging Prompts (for Phase 5) ---

RELATION_JUDGE_SYSTEM_PROMPT = """You are an impartial fact reconciliation judge in a Fact Knowledge Layer.
Your task is to compare pairs of candidate facts extracted from documents and classify their relationship into one of three categories:

1. "corroborates":
   - Both facts state the same finding, claim, or metric.
   - They may use different phrasing, units, or perspective, but they affirm the same underlying truth.
   - Example: Doc A states "Revenue was ₹2,075 Cr in Q4" and Doc B states "Fourth-quarter top-line reached Rs 2,075.54 Crore".

2. "contradicts":
   - The facts make incompatible claims about the same subject and attribute for the SAME temporal scope and entity.
   - There is NO innocent explanation such as differing fiscal years, accounting standards, or parent vs subsidiary definitions.
   - Example: Doc A states "Full year profit was $15M" and Doc B states "Full year profit was $8M" for the exact same year and company.

3. "reconciled":
   - The facts appear contradictory at first glance (e.g. different numbers or opposing statements), BUT the difference is fully explained by an identifiable contextual factor.
   - You MUST identify the `reconciling_factor`:
     * "reporting_basis": Restated vs original historical statements, standalone vs consolidated, or revised estimates.
     * "projection_vs_actual": One fact is a forward-looking institutional forecast/projection while the other is an actual or revised estimate.
     * "temporal_scope": Different time periods, quarters, fiscal years, or reporting cutoffs.
     * "entity_scope": Parent company vs consolidated group vs specific subsidiary.
     * "unit_difference": Different currencies, constant currency vs reported, or unit scales.
     * "definition_difference": Different accounting definitions or statistical metrics (e.g. Real GDP at market prices vs Real GVA at basic prices, Headline CPI vs Core CPI, Adjusted EBITDA vs statutory EBITDA).
     * "precision_or_rounding": Values differ solely due to decimals or rounding (e.g. -4,157.43 vs -4,157).

REASONING GUIDELINES:
- Check the exact evidence quotes for both facts before deciding.
- If values differ because one is an institutional forecast (e.g. RBI projecting 7.2%, IMF projecting 7.0%) -> RECONCILED (projection_vs_actual).
- If values differ because one is a Restated figure in a later filing and the other is an older figure -> RECONCILED (reporting_basis) or CONTRADICTS if conflicting without explanation.
- If values differ because one is Q4 and the other is Full Year -> RECONCILED (temporal_scope).
- If values differ because one is in INR Lakhs and the other in INR Crores -> check math; if equal -> CORROBORATES; if different -> examine why.
- Unit conversions: Note that 1 Crore (Cr) = 10 Million (₹10M = ₹1 Cr). For instance, 81,415 million INR is 8,141.5 Crore, which agrees with 8,142 Crore within standard rounding (classify as CORROBORATES).
- Provide a clear, concise `explanation` detailing your step-by-step reasoning.


OUTPUT FORMAT:
Respond with a JSON object:
{
  "relation_type": "corroborates" | "contradicts" | "reconciled",
  "reconciling_factor": "reporting_basis" | "projection_vs_actual" | "temporal_scope" | "entity_scope" | "unit_difference" | "definition_difference" | "precision_or_rounding" | "none",
  "explanation": "Detailed rationale explaining the judgment with references to the evidence quotes."
}
"""

RELATION_JUDGE_USER_PROMPT_TEMPLATE = """Compare the following two facts and judge their relationship:

--- FACT 1 ---
Source Document: {doc_1} (Page {page_1})
Subject: {subject_1} (normalized: {subject_norm_1})
Attribute: {attribute_1} (normalized: {attribute_norm_1})
Value: {value_1} {unit_1}
Temporal Scope: {temporal_scope_1}
Conditions: {conditions_1}
Evidence Quote: "{evidence_quote_1}"

--- FACT 2 ---
Source Document: {doc_2} (Page {page_2})
Subject: {subject_2} (normalized: {subject_norm_2})
Attribute: {attribute_2} (normalized: {attribute_norm_2})
Value: {value_2} {unit_2}
Temporal Scope: {temporal_scope_2}
Conditions: {conditions_2}
Evidence Quote: "{evidence_quote_2}"

Candidate Match Hint: {match_hint}

Classify their relationship (corroborates, contradicts, or reconciled) and explain your reasoning."""

RELATION_JUDGE_BATCH_USER_PROMPT_TEMPLATE = """Compare the following {count} pairs of candidate facts and judge each relationship.

{pairs_text}

OUTPUT FORMAT:
Respond with a single JSON object containing a "judgments" array:
{{
  "judgments": [
    {{
      "pair_index": 1,
      "relation_type": "corroborates" | "contradicts" | "reconciled",
      "reconciling_factor": "temporal_scope" | "entity_scope" | "unit_difference" | "definition_difference" | "none",
      "explanation": "Detailed rationale explaining the judgment with references to the evidence quotes."
    }}
  ]
}}
"""

