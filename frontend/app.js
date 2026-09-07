/**
 * Concord — Fact Knowledge Layer Frontend Logic
 * Vanilla JavaScript implementation managing API state, uploads,
 * relationships graph rendering, facts table, and modal interactions.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- State ---
    const state = {
        apiKey: sessionStorage.getItem("concord_api_key") || "",
        model: localStorage.getItem("concord_model") || "gemini/gemini-3.6-flash",
        documents: [],
        facts: [],
        relationships: [],
        canonicals: { subjects: [], attributes: [] },
        activeFilter: "all",
        crossDocOnly: true,
        searchQuery: "",
        selectedDocFilter: "",
    };

    // --- DOM Elements ---
    const elements = {
        // Top Nav
        apiKeyInput: document.getElementById("apiKeyInput"),
        toggleKeyVisibility: document.getElementById("toggleKeyVisibility"),
        validateKeyBtn: document.getElementById("validateKeyBtn"),
        keyStatusBadge: document.getElementById("keyStatusBadge"),
        modelSelect: document.getElementById("modelSelect"),
        serverStatusBadge: document.getElementById("serverStatusBadge"),
        serverStatusText: document.getElementById("serverStatusText"),

        // Stats
        statDocsCount: document.getElementById("statDocsCount"),
        statFactsCount: document.getElementById("statFactsCount"),
        statCorroboratesCount: document.getElementById("statCorroboratesCount"),
        statContradictsCount: document.getElementById("statContradictsCount"),
        statReconciledCount: document.getElementById("statReconciledCount"),

        // Tabs
        tabBtns: document.querySelectorAll(".tab-btn"),
        tabPanes: document.querySelectorAll(".tab-pane"),
        tabRelCount: document.getElementById("tabRelCount"),
        tabFactsCount: document.getElementById("tabFactsCount"),
        tabDocsCount: document.getElementById("tabDocsCount"),

        // Ingestion
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("fileInput"),
        limitPagesCheck: document.getElementById("limitPagesCheck"),
        sampleButtons: document.querySelectorAll(".sample-btn"),
        processingCard: document.getElementById("processingCard"),
        processingTitle: document.getElementById("processingTitle"),
        processingDetail: document.getElementById("processingDetail"),
        progressBarFill: document.getElementById("progressBarFill"),

        // Relationships View
        relationshipsFeed: document.getElementById("relationshipsFeed"),
        filterPills: document.querySelectorAll(".filter-pill"),
        crossDocOnlyToggle: document.getElementById("crossDocOnlyToggle"),
        countRelAll: document.getElementById("countRelAll"),
        countRelCorr: document.getElementById("countRelCorr"),
        countRelCont: document.getElementById("countRelCont"),
        countRelRec: document.getElementById("countRelRec"),
        refreshRelationshipsBtn: document.getElementById("refreshRelationshipsBtn"),

        // Facts View
        factsTableBody: document.getElementById("factsTableBody"),
        factsSearchInput: document.getElementById("factsSearchInput"),
        filterFactDocSelect: document.getElementById("filterFactDocSelect"),
        refreshFactsBtn: document.getElementById("refreshFactsBtn"),

        // Documents View
        documentsGrid: document.getElementById("documentsGrid"),
        refreshDocsBtn: document.getElementById("refreshDocsBtn"),

        // Canonicals View
        canonicalSubjectsList: document.getElementById("canonicalSubjectsList"),
        canonicalAttributesList: document.getElementById("canonicalAttributesList"),
        canonicalSubjectsCount: document.getElementById("canonicalSubjectsCount"),
        canonicalAttributesCount: document.getElementById("canonicalAttributesCount"),

        // Compare Sandbox
        compareFact1Select: document.getElementById("compareFact1Select"),
        compareFact2Select: document.getElementById("compareFact2Select"),
        previewFact1: document.getElementById("previewFact1"),
        previewFact2: document.getElementById("previewFact2"),
        runCompareBtn: document.getElementById("runCompareBtn"),
        compareResultCard: document.getElementById("compareResultCard"),
        compareResultBadge: document.getElementById("compareResultBadge"),
        compareResultBody: document.getElementById("compareResultBody"),

        // Export
        jsonExportPreview: document.getElementById("jsonExportPreview"),
        downloadExportJsonBtn: document.getElementById("downloadExportJsonBtn"),

        // Modal
        factModal: document.getElementById("factModal"),
        closeFactModalBtn: document.getElementById("closeFactModalBtn"),
        modalFactTitle: document.getElementById("modalFactTitle"),
        modalFactScope: document.getElementById("modalFactScope"),
        modalFactQuote: document.getElementById("modalFactQuote"),
        modalFactDoc: document.getElementById("modalFactDoc"),
        modalFactPage: document.getElementById("modalFactPage"),
        modalFactConfidence: document.getElementById("modalFactConfidence"),
        modalFactSubject: document.getElementById("modalFactSubject"),
        modalFactSubjectNorm: document.getElementById("modalFactSubjectNorm"),
        modalFactAttribute: document.getElementById("modalFactAttribute"),
        modalFactAttrNorm: document.getElementById("modalFactAttrNorm"),
        modalFactValue: document.getElementById("modalFactValue"),
        modalFactFingerprint: document.getElementById("modalFactFingerprint"),
        modalRelCount: document.getElementById("modalRelCount"),
        modalRelationshipsList: document.getElementById("modalRelationshipsList"),

        // Toasts
        toastContainer: document.getElementById("toastContainer"),
    };

    // --- Initialization ---
    initApp();

    function initApp() {
        // Restore API Key & Model
        if (state.apiKey) {
            elements.apiKeyInput.value = state.apiKey;
            elements.keyStatusBadge.textContent = "Saved";
            elements.keyStatusBadge.className = "key-status-indicator text-muted";
        }
        if (state.model) {
            elements.modelSelect.value = state.model;
        }

        setupEventListeners();
        checkServerHealth();
        refreshAllData();
    }

    // --- Event Listeners Setup ---
    function setupEventListeners() {
        // API Key input & toggle
        elements.apiKeyInput.addEventListener("input", (e) => {
            state.apiKey = e.target.value.trim();
            sessionStorage.setItem("concord_api_key", state.apiKey);
            elements.keyStatusBadge.textContent = state.apiKey ? "Unverified" : "";
            elements.keyStatusBadge.className = "key-status-indicator text-muted";
        });

        elements.toggleKeyVisibility.addEventListener("click", () => {
            const current = elements.apiKeyInput.getAttribute("type");
            elements.apiKeyInput.setAttribute("type", current === "password" ? "text" : "password");
            elements.toggleKeyVisibility.textContent = current === "password" ? "🔒" : "👁️";
        });

        elements.validateKeyBtn.addEventListener("click", validateApiKey);

        // Model Select
        elements.modelSelect.addEventListener("change", (e) => {
            state.model = e.target.value;
            localStorage.setItem("concord_model", state.model);
            showToast(`Switched model to ${elements.modelSelect.options[elements.modelSelect.selectedIndex].text}`, "info");
        });

        // Tabs
        elements.tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                const target = btn.getAttribute("data-tab");
                switchTab(target);
            });
        });

        // File Dropzone
        elements.dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.dropzone.classList.add("dragover");
        });

        elements.dropzone.addEventListener("dragleave", () => {
            elements.dropzone.classList.remove("dragover");
        });

        elements.dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.dropzone.classList.remove("dragover");
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        elements.fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        // Sample Dataset Buttons
        elements.sampleButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const sampleKey = btn.getAttribute("data-sample");
                handleLoadSample(sampleKey);
            });
        });

        // Relationship Filters
        elements.filterPills.forEach((pill) => {
            pill.addEventListener("click", () => {
                elements.filterPills.forEach((p) => p.classList.remove("active"));
                pill.classList.add("active");
                state.activeFilter = pill.getAttribute("data-filter");
                renderRelationships();
            });
        });

        elements.crossDocOnlyToggle.addEventListener("change", (e) => {
            state.crossDocOnly = e.target.checked;
            renderRelationships();
        });

        elements.refreshRelationshipsBtn.addEventListener("click", () => {
            fetchRelationships();
            showToast("Relationships refreshed", "info");
        });

        // Facts Table Search & Filter
        elements.factsSearchInput.addEventListener("input", (e) => {
            state.searchQuery = e.target.value.toLowerCase();
            renderFacts();
        });

        elements.filterFactDocSelect.addEventListener("change", (e) => {
            state.selectedDocFilter = e.target.value;
            renderFacts();
        });

        elements.refreshFactsBtn.addEventListener("click", () => {
            fetchFacts();
            showToast("Facts refreshed", "info");
        });

        // Documents Refresh
        elements.refreshDocsBtn.addEventListener("click", () => {
            fetchDocuments();
            showToast("Documents refreshed", "info");
        });

        // Compare Sandbox
        elements.compareFact1Select.addEventListener("change", () => updateComparePreview(1));
        elements.compareFact2Select.addEventListener("change", () => updateComparePreview(2));
        elements.runCompareBtn.addEventListener("click", runOnDemandComparison);

        // Export Download
        elements.downloadExportJsonBtn.addEventListener("click", downloadExportJson);

        // Modal Close
        elements.closeFactModalBtn.addEventListener("click", closeFactModal);
        elements.factModal.addEventListener("click", (e) => {
            if (e.target === elements.factModal) {
                closeFactModal();
            }
        });
    }

    // --- Tab Switching ---
    function switchTab(tabName) {
        elements.tabBtns.forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
        });
        elements.tabPanes.forEach((pane) => {
            pane.classList.toggle("active", pane.id === `pane${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`);
        });

        if (tabName === "export") {
            loadExportPreview();
        } else if (tabName === "canonicals") {
            fetchCanonicals();
        } else if (tabName === "compare") {
            populateCompareDropdowns();
        }
    }

    // --- API Key Validation ---
    async function validateApiKey() {
        if (!state.apiKey) {
            showToast("Please paste an API key first", "error");
            elements.apiKeyInput.focus();
            return;
        }

        elements.keyStatusBadge.textContent = "Testing...";
        elements.keyStatusBadge.className = "key-status-indicator text-muted";

        try {
            const resp = await fetch(`/api/validate-key?model=${encodeURIComponent(state.model)}`, {
                method: "POST",
                headers: {
                    "X-API-Key": state.apiKey,
                },
            });
            const data = await resp.json();

            if (resp.ok && data.valid) {
                elements.keyStatusBadge.textContent = "✓ Valid";
                elements.keyStatusBadge.className = "key-status-indicator text-success";
                showToast("API Key validated successfully!", "success");
            } else {
                elements.keyStatusBadge.textContent = "✕ Invalid";
                elements.keyStatusBadge.className = "key-status-indicator text-danger";
                showToast(`Validation failed: ${data.message || "Invalid Key"}`, "error");
            }
        } catch (e) {
            elements.keyStatusBadge.textContent = "✕ Error";
            elements.keyStatusBadge.className = "key-status-indicator text-danger";
            showToast(`Could not connect: ${e.message}`, "error");
        }
    }

    // --- Server Health ---
    async function checkServerHealth() {
        try {
            const resp = await fetch("/health");
            if (resp.ok) {
                const data = await resp.json();
                elements.serverStatusText.textContent = `Online • ${data.stats.documents} docs`;
                updateStatCards(data.stats);
            }
        } catch (e) {
            elements.serverStatusText.textContent = "Offline";
            elements.serverStatusBadge.querySelector(".status-dot").style.backgroundColor = "var(--contradicts)";
        }
    }

    function updateStatCards(stats) {
        if (!stats) return;
        elements.statDocsCount.textContent = stats.documents || 0;
        elements.statFactsCount.textContent = stats.facts || 0;
    }

    // --- Data Fetching ---
    async function refreshAllData() {
        await Promise.all([
            fetchDocuments(),
            fetchFacts(),
            fetchRelationships(),
            fetchCanonicals(),
        ]);
    }

    async function fetchDocuments() {
        try {
            const resp = await fetch("/api/documents");
            const data = await resp.json();
            state.documents = data.documents || [];
            elements.tabDocsCount.textContent = state.documents.length;
            elements.statDocsCount.textContent = state.documents.length;
            renderDocuments();
            updateDocFilterOptions();
        } catch (e) {
            console.error("Error fetching documents:", e);
        }
    }

    async function fetchFacts() {
        try {
            const resp = await fetch("/api/facts?limit=1000");
            const data = await resp.json();
            state.facts = data.facts || [];
            elements.tabFactsCount.textContent = state.facts.length;
            elements.statFactsCount.textContent = state.facts.length;
            renderFacts();
            populateCompareDropdowns();
        } catch (e) {
            console.error("Error fetching facts:", e);
        }
    }

    async function fetchRelationships() {
        try {
            const resp = await fetch("/api/relationships?limit=1000");
            const data = await resp.json();
            state.relationships = data.relationships || [];
            elements.tabRelCount.textContent = state.relationships.length;
            updateRelationshipCounters();
            renderRelationships();
        } catch (e) {
            console.error("Error fetching relationships:", e);
        }
    }

    async function fetchCanonicals() {
        try {
            const resp = await fetch("/api/normalization/canonicals");
            const data = await resp.json();
            state.canonicals = data;
            renderCanonicals();
        } catch (e) {
            console.error("Error fetching canonicals:", e);
        }
    }

    // --- Ingestion: Upload File ---
    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showToast("Only PDF files are supported.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        const maxPages = elements.limitPagesCheck.checked ? 8 : 100;
        const queryParams = new URLSearchParams({
            model: state.model,
            max_pages: maxPages,
        });

        startProgressUI(`Ingesting ${file.name}`);

        try {
            const headers = {};
            if (state.apiKey) {
                headers["X-API-Key"] = state.apiKey;
            }

            const resp = await fetch(`/api/upload?${queryParams.toString()}`, {
                method: "POST",
                headers: headers,
                body: formData,
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Upload failed");
            }

            finishProgressUI();

            if (data.status === "extracted") {
                showToast(`Success! Extracted ${data.fact_count} facts from ${file.name}`, "success");
            } else {
                showToast(`PDF parsed (${data.page_count} pages). Set an API Key to extract facts.`, "info");
            }

            await refreshAllData();
        } catch (e) {
            finishProgressUI();
            showToast(`Processing error: ${e.message}`, "error");
        }
    }

    // --- Ingestion: Load Sample Dataset ---
    async function handleLoadSample(sampleKey) {
        const maxPages = elements.limitPagesCheck.checked ? 8 : 100;
        const queryParams = new URLSearchParams({
            model: state.model,
            max_pages: maxPages,
        });

        startProgressUI(`Loading sample dataset: ${sampleKey}`);

        try {
            const headers = {};
            if (state.apiKey) {
                headers["X-API-Key"] = state.apiKey;
            }

            const resp = await fetch(`/api/sample-datasets/${sampleKey}/load?${queryParams.toString()}`, {
                method: "POST",
                headers: headers,
            });

            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Failed to load sample dataset");
            }

            finishProgressUI();
            showToast(`Loaded ${data.doc_name}! Facts: ${data.fact_count}`, "success");
            await refreshAllData();
        } catch (e) {
            finishProgressUI();
            showToast(`Error: ${e.message}`, "error");
        }
    }

    // --- Progress Stepper UI ---
    function startProgressUI(title) {
        elements.processingCard.classList.remove("hidden");
        elements.processingTitle.textContent = title;
        elements.processingDetail.textContent = "Reading PDF pages and extracting text layer...";
        elements.progressBarFill.style.width = "25%";

        document.getElementById("stepParse").className = "step-badge active";
        document.getElementById("stepExtract").className = "step-badge";
        document.getElementById("stepMatch").className = "step-badge";
        document.getElementById("stepJudge").className = "step-badge";

        setTimeout(() => {
            elements.progressBarFill.style.width = "50%";
            document.getElementById("stepParse").className = "step-badge done";
            document.getElementById("stepExtract").className = "step-badge active";
            elements.processingDetail.textContent = "Extracting atomic grounded facts with LLM...";
        }, 1200);

        setTimeout(() => {
            elements.progressBarFill.style.width = "75%";
            document.getElementById("stepExtract").className = "step-badge done";
            document.getElementById("stepMatch").className = "step-badge active";
            elements.processingDetail.textContent = "Executing Lane 1 structural & Lane 2 embedding matching...";
        }, 3000);

        setTimeout(() => {
            elements.progressBarFill.style.width = "90%";
            document.getElementById("stepMatch").className = "step-badge done";
            document.getElementById("stepJudge").className = "step-badge active";
            elements.processingDetail.textContent = "Judging relationships and contextual reconciliation...";
        }, 5000);
    }

    function finishProgressUI() {
        elements.progressBarFill.style.width = "100%";
        document.getElementById("stepJudge").className = "step-badge done";
        elements.processingDetail.textContent = "Processing complete!";
        setTimeout(() => {
            elements.processingCard.classList.add("hidden");
        }, 1500);
    }

    // --- Render: Relationships Feed ---
    function updateRelationshipCounters() {
        const corr = state.relationships.filter((r) => r.relation_type === "corroborates").length;
        const cont = state.relationships.filter((r) => r.relation_type === "contradicts").length;
        const rec = state.relationships.filter((r) => r.relation_type === "reconciled").length;

        elements.countRelAll.textContent = state.relationships.length;
        elements.countRelCorr.textContent = corr;
        elements.countRelCont.textContent = cont;
        elements.countRelRec.textContent = rec;

        elements.statCorroboratesCount.textContent = corr;
        elements.statContradictsCount.textContent = cont;
        elements.statReconciledCount.textContent = rec;
    }

    function renderRelationships() {
        let rels = state.relationships;

        // Apply filter pill
        if (state.activeFilter !== "all") {
            rels = rels.filter((r) => r.relation_type === state.activeFilter);
        }

        // Apply cross-document toggle
        if (state.crossDocOnly) {
            rels = rels.filter((r) => !r.is_intra_document);
        }

        if (rels.length === 0) {
            elements.relationshipsFeed.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔎</div>
                    <h4>No Relationships Match Current Filter</h4>
                    <p>Try switching to "All" or toggle "Cross-Document Only" off to inspect intra-document statements.</p>
                </div>
            `;
            return;
        }

        elements.relationshipsFeed.innerHTML = rels
            .map((rel) => {
                const f1 = rel.fact_1 || {
                    subject: "Unknown",
                    attribute: "Metric",
                    value: "-",
                    source_doc: "Doc 1",
                    page: 1,
                    evidence_quote: "No quote",
                };
                const f2 = rel.fact_2 || {
                    subject: "Unknown",
                    attribute: "Metric",
                    value: "-",
                    source_doc: "Doc 2",
                    page: 1,
                    evidence_quote: "No quote",
                };

                let badgeClass = "badge-corroborates";
                let badgeIcon = "🤝";
                let badgeText = "Corroborates";

                if (rel.relation_type === "contradicts") {
                    badgeClass = "badge-contradicts";
                    badgeIcon = "⚔️";
                    badgeText = "Contradicts";
                } else if (rel.relation_type === "reconciled") {
                    badgeClass = "badge-reconciled";
                    badgeIcon = "⚖️";
                    badgeText = "Reconciled";
                }

                const factorBadge =
                    rel.reconciling_factor && rel.reconciling_factor !== "none"
                        ? `<span class="rel-factor-chip">Factor: ${rel.reconciling_factor.replace("_", " ")}</span>`
                        : "";

                const matchSourceText =
                    rel.match_source === "both"
                        ? "Lane 1 (Structural) + Lane 2 (Embedding)"
                        : rel.match_source === "structural"
                        ? "Lane 1: Structural Match"
                        : "Lane 2: Embedding Similarity";

                return `
                <div class="relation-card card-${rel.relation_type}">
                    <div class="relation-card-header">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span class="rel-badge ${badgeClass}">${badgeIcon} ${badgeText}</span>
                            ${factorBadge}
                            ${rel.is_intra_document ? '<span class="badge" style="background:rgba(245,158,11,0.15); color:#fcd34d;">Intra-Document</span>' : ''}
                        </div>
                        <span class="match-source-chip">${matchSourceText}</span>
                    </div>

                    <div class="comparison-grid">
                        <div class="fact-box">
                            <div class="fact-box-header">
                                <span class="doc-tag" title="${escapeHtml(f1.source_doc)}">📄 ${escapeHtml(f1.source_doc)}</span>
                                <span class="page-tag">p. ${f1.page}</span>
                            </div>
                            <div class="fact-main-statement">
                                <span class="fact-subject">${escapeHtml(f1.subject)}</span>
                                <span class="fact-attribute">${escapeHtml(f1.attribute)}</span>
                                <span class="fact-value-chip">${escapeHtml(f1.value)} ${escapeHtml(f1.unit || '')}</span>
                            </div>
                            <div class="fact-meta-row">
                                <span>Period: <strong>${escapeHtml(f1.temporal_scope || 'unspecified')}</strong></span>
                                ${f1.conditions ? `<span>Scope: <em>${escapeHtml(f1.conditions)}</em></span>` : ''}
                            </div>
                            <div class="evidence-quote-snippet" title="Verbatim source quote">
                                "${escapeHtml(f1.evidence_quote)}"
                            </div>
                        </div>

                        <div class="bridge-arrow">⇄</div>

                        <div class="fact-box">
                            <div class="fact-box-header">
                                <span class="doc-tag" title="${escapeHtml(f2.source_doc)}">📄 ${escapeHtml(f2.source_doc)}</span>
                                <span class="page-tag">p. ${f2.page}</span>
                            </div>
                            <div class="fact-main-statement">
                                <span class="fact-subject">${escapeHtml(f2.subject)}</span>
                                <span class="fact-attribute">${escapeHtml(f2.attribute)}</span>
                                <span class="fact-value-chip">${escapeHtml(f2.value)} ${escapeHtml(f2.unit || '')}</span>
                            </div>
                            <div class="fact-meta-row">
                                <span>Period: <strong>${escapeHtml(f2.temporal_scope || 'unspecified')}</strong></span>
                                ${f2.conditions ? `<span>Scope: <em>${escapeHtml(f2.conditions)}</em></span>` : ''}
                            </div>
                            <div class="evidence-quote-snippet" title="Verbatim source quote">
                                "${escapeHtml(f2.evidence_quote)}"
                            </div>
                        </div>
                    </div>

                    <div class="relation-reasoning">
                        <span class="reasoning-icon">💡</span>
                        <div class="reasoning-text">
                            <strong>System Reasoning:</strong> ${escapeHtml(rel.explanation)}
                        </div>
                    </div>
                </div>
                `;
            })
            .join("");
    }

    // --- Render: Facts Table ---
    function renderFacts() {
        let facts = state.facts;

        // Filter by selected doc
        if (state.selectedDocFilter) {
            facts = facts.filter((f) => f.source_doc_id === state.selectedDocFilter);
        }

        // Filter by search query
        if (state.searchQuery) {
            const q = state.searchQuery;
            facts = facts.filter(
                (f) =>
                    f.subject.toLowerCase().includes(q) ||
                    f.subject_normalized.toLowerCase().includes(q) ||
                    f.attribute.toLowerCase().includes(q) ||
                    f.attribute_normalized.toLowerCase().includes(q) ||
                    f.value.toLowerCase().includes(q) ||
                    f.evidence_quote.toLowerCase().includes(q)
            );
        }

        if (facts.length === 0) {
            elements.factsTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center py-4 text-muted">No facts found matching criteria.</td>
                </tr>
            `;
            return;
        }

        elements.factsTableBody.innerHTML = facts
            .map((fact) => {
                return `
                <tr>
                    <td>
                        <strong>${escapeHtml(fact.subject)}</strong>
                        <br><span class="code-pill">${escapeHtml(fact.subject_normalized)}</span>
                    </td>
                    <td>
                        ${escapeHtml(fact.attribute)}
                        <br><span class="code-pill">${escapeHtml(fact.attribute_normalized)}</span>
                    </td>
                    <td>
                        <strong class="highlight-value">${escapeHtml(fact.value)}</strong>
                        ${fact.unit ? `<small class="text-muted"> ${escapeHtml(fact.unit)}</small>` : ''}
                    </td>
                    <td>
                        <span class="badge">${escapeHtml(fact.temporal_scope || 'unspecified')}</span>
                    </td>
                    <td style="max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(fact.source_doc)}">
                        ${escapeHtml(fact.source_doc)}
                    </td>
                    <td>
                        <span class="page-tag">p. ${fact.page}</span>
                    </td>
                    <td class="quote-cell" title="${escapeHtml(fact.evidence_quote)}">
                        "${escapeHtml(fact.evidence_quote)}"
                    </td>
                    <td>
                        <button class="btn btn-secondary btn-sm view-fact-btn" data-fact-id="${fact.id}">
                            🔍 Inspect
                        </button>
                    </td>
                </tr>
                `;
            })
            .join("");

        // Attach modal openers
        document.querySelectorAll(".view-fact-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const factId = btn.getAttribute("data-fact-id");
                openFactModal(factId);
            });
        });
    }

    function updateDocFilterOptions() {
        const current = elements.filterFactDocSelect.value;
        elements.filterFactDocSelect.innerHTML = `<option value="">All Documents</option>` +
            state.documents
                .map((d) => `<option value="${d.doc_id}" ${d.doc_id === current ? 'selected' : ''}>${escapeHtml(d.doc_name)}</option>`)
                .join("");
    }

    // --- Render: Documents Grid ---
    function renderDocuments() {
        if (state.documents.length === 0) {
            elements.documentsGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📁</div>
                    <h4>No Documents Ingested</h4>
                    <p>Upload a PDF above to register documents into the knowledge layer.</p>
                </div>
            `;
            return;
        }

        elements.documentsGrid.innerHTML = state.documents
            .map((doc) => {
                const dateStr = doc.created_at ? new Date(doc.created_at).toLocaleDateString() : "Recent";
                return `
                <div class="doc-card">
                    <div class="doc-card-top">
                        <div class="doc-icon">📄</div>
                        <div class="doc-info">
                            <h5 title="${escapeHtml(doc.doc_name)}">${escapeHtml(doc.doc_name)}</h5>
                            <p>Added: ${dateStr}</p>
                        </div>
                    </div>

                    <div class="doc-stats-strip">
                        <span><strong>${doc.page_count}</strong> Pages</span>
                        <span>•</span>
                        <span><strong>${doc.fact_count}</strong> Facts</span>
                    </div>

                    <div class="doc-card-actions">
                        <button class="btn btn-secondary btn-sm process-doc-btn" data-doc-id="${doc.doc_id}">
                            ⚡ Re-process
                        </button>
                        <button class="btn btn-danger btn-sm delete-doc-btn" data-doc-id="${doc.doc_id}">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
                `;
            })
            .join("");

        // Attach action handlers
        document.querySelectorAll(".delete-doc-btn").forEach((btn) => {
            btn.addEventListener("click", async () => {
                const docId = btn.getAttribute("data-doc-id");
                if (confirm("Delete this document and all its associated facts and relationships?")) {
                    await handleDeleteDocument(docId);
                }
            });
        });

        document.querySelectorAll(".process-doc-btn").forEach((btn) => {
            btn.addEventListener("click", async () => {
                const docId = btn.getAttribute("data-doc-id");
                await handleProcessExistingDoc(docId);
            });
        });
    }

    async function handleDeleteDocument(docId) {
        try {
            const resp = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
            if (resp.ok) {
                showToast("Document deleted", "info");
                await refreshAllData();
            } else {
                throw new Error("Failed to delete");
            }
        } catch (e) {
            showToast(`Delete failed: ${e.message}`, "error");
        }
    }

    async function handleProcessExistingDoc(docId) {
        if (!state.apiKey) {
            showToast("Please enter an API key to process this document", "error");
            elements.apiKeyInput.focus();
            return;
        }

        startProgressUI(`Processing document ${docId}`);

        try {
            const resp = await fetch(`/api/documents/${docId}/process?model=${encodeURIComponent(state.model)}`, {
                method: "POST",
                headers: {
                    "X-API-Key": state.apiKey,
                },
            });
            const data = await resp.json();

            finishProgressUI();

            if (resp.ok) {
                showToast(`Extracted ${data.fact_count} facts and ${data.relationship_count} relationships`, "success");
                await refreshAllData();
            } else {
                throw new Error(data.detail || "Processing failed");
            }
        } catch (e) {
            finishProgressUI();
            showToast(`Processing error: ${e.message}`, "error");
        }
    }

    // --- Render: Canonicals View ---
    function renderCanonicals() {
        const subjects = state.canonicals.subjects || [];
        const attributes = state.canonicals.attributes || [];

        elements.canonicalSubjectsCount.textContent = subjects.length;
        elements.canonicalAttributesCount.textContent = attributes.length;

        elements.canonicalSubjectsList.innerHTML =
            subjects.length > 0
                ? subjects.map((s) => `<span class="canonical-chip">${escapeHtml(s)}</span>`).join("")
                : `<span class="text-muted">No canonical entities registered yet.</span>`;

        elements.canonicalAttributesList.innerHTML =
            attributes.length > 0
                ? attributes.map((a) => `<span class="canonical-chip">${escapeHtml(a)}</span>`).join("")
                : `<span class="text-muted">No canonical attributes registered yet.</span>`;
    }

    // --- Compare Sandbox ---
    function populateCompareDropdowns() {
        if (!elements.compareFact1Select) return;

        const options = state.facts
            .map(
                (f) =>
                    `<option value="${f.id}">${escapeHtml(f.subject)} • ${escapeHtml(f.attribute)} = ${escapeHtml(f.value)} (${escapeHtml(f.temporal_scope || 'unspecified')})</option>`
            )
            .join("");

        elements.compareFact1Select.innerHTML = `<option value="">-- Choose Fact 1 --</option>` + options;
        elements.compareFact2Select.innerHTML = `<option value="">-- Choose Fact 2 --</option>` + options;
    }

    function updateComparePreview(index) {
        const select = index === 1 ? elements.compareFact1Select : elements.compareFact2Select;
        const preview = index === 1 ? elements.previewFact1 : elements.previewFact2;
        const factId = select.value;

        const fact = state.facts.find((f) => f.id === factId);
        if (!fact) {
            preview.innerHTML = `<p class="text-muted">Select a fact above to preview its structured data and evidence quote.</p>`;
            return;
        }

        preview.innerHTML = `
            <div style="display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                <strong>${escapeHtml(fact.subject)}</strong>
                <span class="page-tag">p. ${fact.page}</span>
            </div>
            <p style="color:#cbd5e1; margin-bottom:0.25rem;">${escapeHtml(fact.attribute)}: <strong>${escapeHtml(fact.value)} ${escapeHtml(fact.unit || '')}</strong></p>
            <p style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.5rem;">Period: ${escapeHtml(fact.temporal_scope || 'unspecified')}</p>
            <div class="evidence-quote-snippet" style="font-size:0.75rem;">"${escapeHtml(fact.evidence_quote)}"</div>
        `;
    }

    async function runOnDemandComparison() {
        const id1 = elements.compareFact1Select.value;
        const id2 = elements.compareFact2Select.value;

        if (!id1 || !id2) {
            showToast("Please select two distinct facts to compare", "error");
            return;
        }
        if (id1 === id2) {
            showToast("Please choose two different facts", "error");
            return;
        }
        if (!state.apiKey) {
            showToast("Please enter an API key for the relation judge", "error");
            elements.apiKeyInput.focus();
            return;
        }

        elements.runCompareBtn.disabled = true;
        elements.runCompareBtn.innerHTML = `<span>⏳ Judging Relationship...</span>`;

        try {
            const resp = await fetch(`/api/relationships/compare?model=${encodeURIComponent(state.model)}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": state.apiKey,
                },
                body: JSON.stringify({ fact_id_1: id1, fact_id_2: id2 }),
            });

            const data = await resp.json();
            elements.runCompareBtn.disabled = false;
            elements.runCompareBtn.innerHTML = `<span>⚖️ Run Relation Judge</span>`;

            if (!resp.ok) {
                throw new Error(data.detail || "Comparison failed");
            }

            const rel = data.relationship;
            elements.compareResultCard.classList.remove("hidden");

            let badgeClass = "badge-corroborates";
            if (rel.relation_type === "contradicts") badgeClass = "badge-contradicts";
            if (rel.relation_type === "reconciled") badgeClass = "badge-reconciled";

            elements.compareResultBadge.className = `rel-badge ${badgeClass}`;
            elements.compareResultBadge.textContent = rel.relation_type.toUpperCase();

            elements.compareResultBody.innerHTML = `
                <p style="margin-bottom:0.5rem;"><strong>Reconciling Factor:</strong> ${escapeHtml(rel.reconciling_factor || 'none')}</p>
                <div class="relation-reasoning" style="margin-top:0.5rem;">
                    <span class="reasoning-icon">💡</span>
                    <div class="reasoning-text">${escapeHtml(rel.explanation)}</div>
                </div>
            `;

            showToast(`Judged as ${rel.relation_type}!`, "success");
            await fetchRelationships();
        } catch (e) {
            elements.runCompareBtn.disabled = false;
            elements.runCompareBtn.innerHTML = `<span>⚖️ Run Relation Judge</span>`;
            showToast(`Comparison error: ${e.message}`, "error");
        }
    }

    // --- Export View ---
    async function loadExportPreview() {
        elements.jsonExportPreview.textContent = "Loading knowledge layer snapshot...";
        try {
            const resp = await fetch("/api/export");
            const data = await resp.json();
            elements.jsonExportPreview.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
            elements.jsonExportPreview.textContent = `Error loading export: ${e.message}`;
        }
    }

    async function downloadExportJson() {
        try {
            const resp = await fetch("/api/export");
            const data = await resp.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `concord-knowledge-layer-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast("Downloaded knowledge layer JSON bundle", "success");
        } catch (e) {
            showToast(`Export error: ${e.message}`, "error");
        }
    }

    // --- Fact Detail Modal ---
    async function openFactModal(factId) {
        const fact = state.facts.find((f) => f.id === factId);
        if (!fact) return;

        elements.modalFactTitle.textContent = `${fact.subject} — ${fact.attribute}`;
        elements.modalFactScope.textContent = fact.temporal_scope || "Scope Unspecified";
        elements.modalFactQuote.textContent = fact.evidence_quote;
        elements.modalFactDoc.textContent = `📄 ${fact.source_doc}`;
        elements.modalFactPage.textContent = `📍 Page ${fact.page}`;
        elements.modalFactConfidence.textContent = `🎯 Confidence: ${(fact.confidence * 100).toFixed(0)}%`;

        elements.modalFactSubject.textContent = fact.subject;
        elements.modalFactSubjectNorm.textContent = fact.subject_normalized;
        elements.modalFactAttribute.textContent = fact.attribute;
        elements.modalFactAttrNorm.textContent = fact.attribute_normalized;
        elements.modalFactValue.textContent = `${fact.value} ${fact.unit || ""}`;
        elements.modalFactFingerprint.textContent = fact.claim_fingerprint;

        // Fetch related links
        try {
            const resp = await fetch(`/api/facts/${factId}/related`);
            if (resp.ok) {
                const data = await resp.json();
                const rels = data.relationships || [];
                elements.modalRelCount.textContent = rels.length;

                if (rels.length === 0) {
                    elements.modalRelationshipsList.innerHTML = `<p class="text-muted">No cross-document relationships recorded for this fact yet.</p>`;
                } else {
                    elements.modalRelationshipsList.innerHTML = rels
                        .map((rel) => {
                            const other = rel.target_fact || { subject: "Other", attribute: "", value: "-" };
                            return `
                            <div style="background:rgba(0,0,0,0.3); border:1px solid var(--border-subtle); padding:0.75rem; border-radius:6px; margin-bottom:0.5rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                                    <span class="rel-badge badge-${rel.relation_type}">${rel.relation_type}</span>
                                    <small class="text-muted">with ${escapeHtml(other.source_doc || 'Doc')}</small>
                                </div>
                                <p style="font-size:0.8rem; color:#cbd5e1;">${escapeHtml(other.subject)}: ${escapeHtml(other.attribute)} = <strong>${escapeHtml(other.value)} ${escapeHtml(other.unit || '')}</strong></p>
                                <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">${escapeHtml(rel.explanation)}</p>
                            </div>
                            `;
                        })
                        .join("");
                }
            }
        } catch (e) {
            console.error("Error loading related facts:", e);
        }

        elements.factModal.classList.remove("hidden");
    }

    function closeFactModal() {
        elements.factModal.classList.add("hidden");
    }

    // --- Toast Notifications ---
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        const icon = type === "success" ? "✓" : type === "error" ? "✕" : "ℹ️";
        toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // --- Utility: HTML Escape ---
    function escapeHtml(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
