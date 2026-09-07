"""
LLM prompts for fact extraction and relation judging.

All prompts are generic and data-agnostic — no hardcoded document-specific schemas,
company names, or metrics. Works across financial reports, macroeconomic surveys,
scientific papers, and general prose.
"""

from __future__ import annotations

# --- Fact Extraction Prompts ---

FACT_EXTRACTION_SYSTEM_PROMPT = """You are an expert fact extraction engine. Your task is to analyze document text and extract atomic, verifiable, grounded facts with precise source citations.

RULES:
1. GROUNDING (CRITICAL):
   - Every fact MUST include an `evidence_quote` which is an EXACT, VERBATIM substring from the provided text.
   - Do NOT paraphrase, summarize, or alter the quote. If the exact words are not in the text, do not extract it.
   - Include the page number provided in the context.

2. ATOMIC DECOMPOSITION:
   - Decompose compound sentences into single atomic facts.
   - For example, if the text says: "In FY24, revenue grew by 15% to $120M from $104M in FY23, while EBITDA margin expanded 200 bps to 8.5%."
     Decompose this into:
     a) Subject: Company | Attribute: revenue | Value: 120 | Unit: million USD | Scope: FY24
     b) Subject: Company | Attribute: revenue | Value: 104 | Unit: million USD | Scope: FY23
     c) Subject: Company | Attribute: revenue growth YoY | Value: 15 | Unit: % | Scope: FY24
     d) Subject: Company | Attribute: EBITDA margin | Value: 8.5 | Unit: % | Scope: FY24
     e) Subject: Company | Attribute: EBITDA margin expansion YoY | Value: 200 | Unit: bps | Scope: FY24
   - Sibling facts decomposed from the same sentence/statement MUST share the SAME `extraction_group_id` (e.g. "g1", "g2").

3. NORMALIZATION:
   - `subject`: The entity or organization name as written (e.g., "Delhivery Limited", "Reserve Bank of India").
   - `subject_normalized`: Lowercase snake_case canonical identifier for the entity (e.g., "delhivery", "rbi"). Strip corporate suffixes like 'ltd', 'limited', 'inc', 'corp', 'pvt' unless essential to distinguish parent vs subsidiary.
   - `attribute`: The property, metric, or statement as written (e.g., "Revenue from operations", "Real GDP Growth", "Adjusted EBITDA").
   - `attribute_normalized`: Lowercase snake_case representation preserving meaningful qualifiers (e.g., "revenue_operations", "gdp_growth_real", "ebitda_adjusted", "net_profit"). Do NOT collapse distinct accounting metrics into a generic name (preserve differences between gross vs net, operating vs total).
   - `value`: The numerical amount, percentage, or concise factual statement as a string. Keep only the raw number/value (e.g., "2075.54", "8.2", "45000", "positive").
   - `unit`: The unit of measurement (e.g., "INR Crore", "%", "million USD", "MT", "bps", "count"). If dimensionless or qualitative, use null.
   - `temporal_scope`: Standardized anchor period if known (e.g., "Q4_FY24", "FY2023", "2024-03-31", "FY25"). If relative (e.g. "last quarter"), resolve it using the document context if possible. If uncertain or not stated, use "unspecified". Never guess.
   - `conditions`: Any conditional clauses, restatements, or qualifying scopes (e.g., "excluding express parcel", "restated", "consolidated", "standalone"). Use null if none.
   - `confidence`: Number between 0.5 and 1.0 reflecting how explicitly the fact is stated.

4. SCOPE & GENERALIZATION:
   - Extract numerical facts, operational metrics, financial data, macroeconomic statistics, and explicit executive statements.
   - Skip trivial narrative filler, table of contents entries, page numbers, or generic disclaimers.

OUTPUT FORMAT:
Respond with a single valid JSON object containing a "facts" array:
{
  "facts": [
    {
      "subject": "Delhivery Limited",
      "subject_normalized": "delhivery",
      "attribute": "Revenue from operations",
      "attribute_normalized": "revenue_operations",
      "value": "2075.54",
      "unit": "INR Crore",
      "temporal_scope": "Q4_FY24",
      "conditions": "consolidated",
      "evidence_quote": "Revenue from operations for Q4 FY24 stood at Rs. 2,075.54 Cr",
      "page": 5,
      "confidence": 0.95,
      "extraction_group_id": "g1"
    }
  ]
}
If no relevant facts are found on the page, return {"facts": []}.
"""

FACT_EXTRACTION_USER_PROMPT_TEMPLATE = """Document: {doc_name}
Page Number: {page_number}

--- PAGE TEXT ---
{page_text}
--- END PAGE TEXT ---

Extract all atomic, verifiable facts from the page above according to the instructions. Ensure every fact has an exact verbatim evidence_quote from the text."""


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
     * "temporal_scope": Different time periods, quarters, fiscal years, or reporting cutoffs.
     * "entity_scope": Parent company vs consolidated group vs specific subsidiary.
     * "unit_difference": Different currencies, gross vs net numbers, constant currency vs reported, or unit scales.
     * "definition_difference": Different accounting definitions (e.g. Adjusted EBITDA vs statutory EBITDA, GAAP vs non-GAAP).

REASONING GUIDELINES:
- Check the exact evidence quotes for both facts before deciding.
- If values differ because one is Q4 and the other is Full Year -> RECONCILED (temporal_scope).
- If values differ because one is Standalone and the other is Consolidated -> RECONCILED (entity_scope).
- If values differ because one is in INR Lakhs and the other in INR Crores -> check math; if equal -> CORROBORATES; if different -> examine why.
- Provide a clear, concise `explanation` detailing your step-by-step reasoning.

OUTPUT FORMAT:
Respond with a JSON object:
{
  "relation_type": "corroborates" | "contradicts" | "reconciled",
  "reconciling_factor": "temporal_scope" | "entity_scope" | "unit_difference" | "definition_difference" | "none",
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

