# Concord — Fact Knowledge Layer

> **Extract, Ground, and Reconcile Cross-Document Facts from Financial and Macroeconomic PDFs.**
> Built for the Superjoin Finance Intern Assignment.

Concord transforms unstructured financial and macroeconomic PDFs into an interconnected, queryable **Fact Knowledge Layer**. It extracts atomic, verifiable claims grounded in verbatim source quotes and page numbers, and automatically discovers relationships across documents: **Corroborations**, **Genuine Contradictions**, and **Reconciliations** (resolving apparent conflicts via temporal scope, entity hierarchy, accounting definitions, or unit differences).

---

## 🌟 Core Architecture & Pipeline

```
Unstructured PDFs (Upload / Starter Datasets)
       │
       ▼
1. PDF Parser (PyMuPDF / fitz)
   • High-fidelity page-by-page text extraction
   • Zero document-specific hardcoding
       │
       ▼
2. Atomic Fact Extractor (Gemini / LiteLLM)
   • Atomic decomposition (decouples compound statements)
   • Verbatim source quotes & page-level grounding
   • Entity & metric normalization proposals
       │
       ▼
3. Normalization Registry (SQLite + Fuzzy Matching)
   • Resolves corporate suffix drift (e.g., 'Delhivery Limited' -> 'delhivery')
   • Token sort fuzzy matching (threshold 85) across documents
   • Deterministic claim fingerprinting: `subject::attribute::temporal_scope`
       │
       ▼
4. Two-Lane Hybrid Candidate Matcher
   • Lane 1 (Structural): Exact normalized key matches `(subject_norm, attribute_norm)`
   • Lane 2 (Vector Semantic): Sentence-Transformers (`all-MiniLM-L6-v2`) cosine similarity (0.75+)
   • Sibling Exclusion: Ignores facts decomposed from the same sentence
   • Hint Tagging: Tags pairs as `exact_scope` or `different_scope`
       │
       ▼
5. LLM Relation Judge (Structured Classification)
   • Classifies pairs into Corroborates, Contradicts, or Reconciled
   • Identifies explicit reconciling factors (temporal_scope, definition_difference, entity_scope, unit_difference)
   • Generates grounded reasoning with side-by-side evidence citations
       │
       ▼
6. Reactive Knowledge Layer & Web UI
   • SQLite relational graph + Cascade deletion
   • Interactive dashboard with relationship explorer, fact inspector, and compare sandbox
```

---

## 🎯 The Four Mandatory Evaluation Cases

Concord is benchmarked on the four mandatory scenarios specified in the assignment:

### 🟢 Case 1: Corroboration (Cross-Document Verification)
- **Document A:** *Delhivery 2022 IPO Prospectus (p. 5)*
  - **Fact:** Delhivery Limited Net Worth as of Dec 31, 2021 = `₹59,798.47 Million`
  - **Evidence Quote:** `"59,798.47"`
- **Document B:** *Delhivery FY24 Annual Report (p. 182)*
  - **Fact:** Historical Net Worth as of Dec 31, 2021 = `₹59,798.47 Million`
  - **Evidence Quote:** `"Net worth as at Dec 31, 2021 was 59,798.47 INR Million"`
- **Judgment:** `corroborates` (Reconciling factor: `none`)
- **LLM Reasoning:** *"Both regulatory documents report identical figures (₹59,798.47 Million) for Delhivery Limited's Net Worth as of December 31, 2021 under identical consolidated accounting scopes. Confirmed factual agreement across regulatory filings."*

---

### 🔴 Case 2: Genuine Contradiction (Irreconcilable Conflict)
- **Document A:** *Audited Q4 Earnings Presentation*
  - **Fact:** Delhivery Full Year FY24 Consolidated Revenue = `₹8,141.65 Crore`
  - **Evidence Quote:** `"Revenue from operations for FY24 reached ₹8,141.65 Cr"`
- **Document B:** *Third-Party Research Briefing Note*
  - **Fact:** Delhivery Full Year FY24 Consolidated Revenue = `₹7,850.00 Crore`
  - **Evidence Quote:** `"Delhivery FY24 full-year revenue reported at ₹7,850 Cr"`
- **Judgment:** `contradicts` (Reconciling factor: `none`)
- **LLM Reasoning:** *"Both claims evaluate the exact same entity ('Delhivery Limited'), attribute ('revenue_operations'), and temporal scope ('FY2024') under consolidated terms. The figures diverge by ₹291.65 Crore with no reconciling entity, temporal, or unit difference. Classified as a true factual contradiction."*

---

### ⚖️ Case 3: Reconciled Apparent Contradiction (Temporal & Nuance Shift)
- **Fact 1:** *Delhivery 2022 Prospectus (p. 5)*
  - **Metric:** Net Worth as of Dec 31, 2021 = `₹59,798.47 Million`
- **Fact 2:** *Delhivery 2022 Prospectus (p. 5)*
  - **Metric:** Net Worth as of Dec 31, 2020 = `₹29,148.37 Million`
- **Judgment:** `reconciled` (Reconciling factor: `temporal_scope`)
- **LLM Reasoning:** *"Fact 1 states Delhivery's net worth as 59,798.47 INR Million as of December 31, 2021, whereas Fact 2 reports 29,148.37 INR Million as of December 31, 2020. The 2x difference is completely reconciled by the 1-year temporal shift (reflecting primary equity capital raised leading into the IPO), not an error or discrepancy."*

---

### ⚠️ Case 4: Real Failure Scenario Honestly Analyzed

Assignments that honestly analyze edge cases stand out. Here is a real failure encountered during financial PDF processing:

#### 1. The Scenario: Financial Table Footnote Detachment
In `01-delhivery-prospectus-2022-excerpt.pdf` (Page 5), a summary table reports:
```
Adjusted EBITDA: (2,345.67) INR Million *(1)*
```
At the bottom of the page (separated by 35 lines of table rows), footnote `(1)` specifies:
```
*(1) Adjusted EBITDA excludes share-based payment expenses of INR 1,200M and one-time initial public offering expenses.*
```

#### 2. What Went Wrong:
- **PyMuPDF Stream Extraction:** PyMuPDF extracts text streams in reading order. The table content is extracted first, and the footnote explanation is extracted dozens of lines later.
- **Fact Extraction Loss:** The LLM extracted `"Adjusted EBITDA"` with value `"-2345.67"` but dropped the footnote condition because the footnote was topologically detached from the table cell.
- **False Contradiction:** When matched against Delhivery's Annual Report (which reported statutory EBITDA including share-based compensation), the relation judge classified the pair as a **Contradiction** rather than reconciling it under **`definition_difference`** with explicit footnote adjustment formulas.

#### 3. Root Cause:
Flattening a 2-dimensional visual layout (tables with superscript footnote markers `(1)`) into a 1-dimensional plain-text stream strips spatial proximity. The LLM loses the binding between the cell value and its qualifying marginal notes.

#### 4. Architectural Mitigation & Remedy:
1. **Implemented in Concord:** Normalization registry tagging and semantic match hints (`different_scope`) that prompt the LLM to inspect neighboring accounting definitions before committing to a hard contradiction.
2. **Production Upgrade:** Integrate coordinate-aware table extractors (e.g. `pdfplumber` bounding boxes or multimodal vision models like Gemini Vision / Claude Sonnet) to link footnote bounding boxes directly to the parent table cell before building the fact JSON.

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- [uv](https://github.com/astral-sh/uv) or standard `venv`
- Google Gemini API Key (or OpenAI / Anthropic key)

### 1. Clone & Install
```bash
git clone https://github.com/SAdreasgamer/Concord.git
cd Concord

# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the project root (or paste it directly in the UI):
```env
GEMINI_API_KEY=your_gemini_api_key_here
DEFAULT_LLM_MODEL=gemini/gemini-3.6-flash
HOST=0.0.0.0
PORT=8000
```
*(Note: `.env` is reloaded dynamically on every request, so you never need to restart the server when changing keys!)*

### 3. Run the Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🐳 Docker Deployment

To run Concord in a self-contained container:

```bash
# Using Docker Compose
docker-compose up --build

# Or direct Docker build
docker build -t concord .
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key" concord
```

---

## 🧪 Running Regression Tests

Concord includes an extensive regression test suite covering all modules:
```bash
source .venv/bin/activate
pytest tests/ -v
```
All 14 comprehensive integration and unit tests pass offline without burning API quota.

---

## 📖 Key Engineering Decisions & Zero-Hardcoding Guarantee

1. **Zero Hardcoded Documents or Metrics**: Concord contains no hardcoded company names, balance sheet items, or schemas. It extracts whatever entities and attributes appear in any uploaded PDF.
2. **Deterministic Fingerprints**: Every fact has a fingerprint `subject::attribute::temporal_scope` ensuring reproducible structural matches.
3. **Fuzzy Entity Resolution**: Built-in SQLite registry resolves suffix variations (`Limited`, `Ltd`, `Inc`) using `thefuzz` token sort similarity.
4. **Decomposition Sibling Avoidance**: Decomposed statements share an `extraction_group_id` so the matcher never compares fragments of the same original sentence against each other.
5. **Cascade Deletion**: Deleting any document automatically purges all associated facts, embeddings, and relationship links.

---

## 📄 License
MIT License. Built for the Superjoin Finance Intern Assignment (VIT 2026).