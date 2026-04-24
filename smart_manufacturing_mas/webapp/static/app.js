const runForm = document.getElementById("run-form");
const datasetSelect = document.getElementById("dataset-select");
const uploadInput = document.getElementById("upload-input");
const datasetPreviewToggle = document.getElementById("dataset-preview-toggle");
const datasetPreview = document.getElementById("dataset-preview");
const runProblemTypeSelect = document.getElementById("run-problem-type");
const runTrainModeSelect = document.getElementById("run-train-mode");
const runTargetColumnSelect = document.getElementById("run-target-column");
const runFeatureColumnsSelect = document.getElementById("run-feature-columns");
const runFeatureColumnsHidden = document.getElementById("run-feature-columns-hidden");
const runPreferredModelSelect = document.getElementById("run-preferred-model");
const runSummary = document.getElementById("run-summary");
const runMetrics = document.getElementById("run-metrics");
const stageProgress = document.getElementById("stage-progress");
const runStatusCopy = document.getElementById("run-status-copy");
const pipelineStages = document.getElementById("pipeline-stages");
const logFeed = document.getElementById("log-feed");
const resultOverview = document.getElementById("result-overview");
const recommendationPreview = document.getElementById("recommendation-preview");
const artifactList = document.getElementById("artifact-list");
const runHistory = document.getElementById("run-history");
const syntheticModal = document.getElementById("synthetic-modal");
const openSyntheticModalButton = document.getElementById("open-synthetic-modal");
const openSyntheticModalInlineButton = document.getElementById("open-synthetic-modal-inline");
const closeSyntheticModalButton = document.getElementById("close-synthetic-modal");
const syntheticProblemTypeSelect = document.getElementById("synthetic-problem-type");
const syntheticPreferredModelSelect = document.getElementById("synthetic-preferred-model");
const tabButtons = [...document.querySelectorAll(".tab-button")];
const tabPanels = [...document.querySelectorAll(".tab-content")];

let activeRunId = null;
let pollHandle = null;
let isDatasetPreviewVisible = false;
let modelCatalog = {
  pretrained: { classification: [], regression: [], anomaly_detection: [] },
  live: {
    classification: ["RandomForestClassifier", "LogisticRegression", "SVC"],
    regression: ["RandomForestRegressor", "HistGradientBoostingRegressor", "LinearRegression", "Ridge", "Lasso", "SVR"],
    anomaly_detection: ["IsolationForest"],
  },
};

const STAGE_LABELS = [
  ["resolve", "Resolve"],
  ["load", "Load"],
  ["preprocess", "Preprocess"],
  ["analyze", "Analyze"],
  ["optimize", "Recommend"],
  ["summary", "Summarize"],
];

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(value) >= 1000) return value.toLocaleString();
    if (Math.abs(value) >= 1) return value.toFixed(3).replace(/\.000$/, "");
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return escapeHtml(JSON.stringify(value));
  return escapeHtml(value);
}

function formatMultiline(value) {
  return escapeHtml(value || "").replaceAll("\n", "<br>");
}

function createTable(columns, rows, maxColumns = 6) {
  const safeColumns = columns.slice(0, maxColumns);
  if (!safeColumns.length || !rows.length) {
    return "<p>No tabular preview available.</p>";
  }

  return `
    <table class="preview-table">
      <thead>
        <tr>${safeColumns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                ${safeColumns.map((column) => `<td>${formatValue(row[column])}</td>`).join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function uniqueStrings(items) {
  return [...new Set((items || []).map((item) => String(item)).filter((item) => item.length > 0))];
}

function setSingleSelectOptions(selectEl, values, emptyLabel, emptyValue = "") {
  if (!selectEl) return;
  const previousValue = selectEl.value;
  const options = uniqueStrings(values);

  selectEl.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = emptyValue;
  placeholder.textContent = emptyLabel;
  selectEl.appendChild(placeholder);

  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectEl.appendChild(option);
  });

  if ([...selectEl.options].some((option) => option.value === previousValue)) {
    selectEl.value = previousValue;
  } else {
    selectEl.value = emptyValue;
  }
}

function setMultiSelectOptions(selectEl, values, emptyLabel) {
  if (!selectEl) return;
  const previous = new Set([...selectEl.selectedOptions].map((option) => option.value));
  const options = uniqueStrings(values);

  selectEl.innerHTML = "";
  if (!options.length) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = emptyLabel;
    placeholder.disabled = true;
    selectEl.appendChild(placeholder);
    return;
  }

  options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = previous.has(value);
    selectEl.appendChild(option);
  });
}

function syncFeatureColumnsHiddenField() {
  if (!runFeatureColumnsSelect || !runFeatureColumnsHidden) return;
  const values = [...runFeatureColumnsSelect.selectedOptions]
    .map((option) => option.value)
    .filter((value) => value);
  runFeatureColumnsHidden.value = values.join(",");
}

function reconcileFeatureSelectionWithTarget() {
  if (!runFeatureColumnsSelect || !runTargetColumnSelect) return;
  const target = runTargetColumnSelect.value;
  [...runFeatureColumnsSelect.options].forEach((option) => {
    if (!option.value) return;
    const shouldDisable = Boolean(target) && option.value === target;
    option.disabled = shouldDisable;
    if (shouldDisable) {
      option.selected = false;
    }
  });
  syncFeatureColumnsHiddenField();
}

function populateRunColumnSelectors(columns) {
  const safeColumns = uniqueStrings(columns);
  setSingleSelectOptions(runTargetColumnSelect, safeColumns, "Auto-detect from dataset");
  setMultiSelectOptions(runFeatureColumnsSelect, safeColumns, "Select dataset in Review CSV tab");
  reconcileFeatureSelectionWithTarget();
}

function getModelOptions(mode, problemType) {
  const catalogForMode = modelCatalog?.[mode] || {};
  if (problemType && catalogForMode[problemType]) {
    return uniqueStrings(catalogForMode[problemType]).sort((a, b) => a.localeCompare(b));
  }

  return uniqueStrings(Object.values(catalogForMode).flat()).sort((a, b) => a.localeCompare(b));
}

function populateModelDropdown(selectEl, options, emptyLabel) {
  if (!selectEl) return;
  setSingleSelectOptions(selectEl, options, emptyLabel);
}

function populateRunModelOptions() {
  const mode = runTrainModeSelect?.value || "pretrained";
  const problemType = runProblemTypeSelect?.value || "";
  const options = getModelOptions(mode, problemType);
  const label = mode === "pretrained" ? "Auto-select best pretrained model" : "Auto-select model";
  populateModelDropdown(runPreferredModelSelect, options, label);
}

function populateSyntheticModelOptions() {
  const problemType = syntheticProblemTypeSelect?.value || syntheticForm?.problem_type?.value || "classification";
  const options = getModelOptions("pretrained", problemType);
  populateModelDropdown(syntheticPreferredModelSelect, options, "Auto-select best model");
}

async function loadModelCatalog() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) return;
    const payload = await response.json();
    if (payload && typeof payload === "object") {
      modelCatalog = {
        ...modelCatalog,
        pretrained: payload.pretrained || modelCatalog.pretrained,
        live: payload.live || modelCatalog.live,
      };
    }
  } catch (error) {
    console.warn("Could not load model catalog from backend.", error);
  } finally {
    populateRunModelOptions();
    populateSyntheticModelOptions();
  }
}

function openSyntheticModal() {
  if (!syntheticModal) return;
  syntheticModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  startSyntheticPolling();
}

function closeSyntheticModal() {
  if (!syntheticModal) return;
  syntheticModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
  stopSyntheticPolling();
}

function setupModalEventListeners() {
  if (openSyntheticModalButton) {
    openSyntheticModalButton.addEventListener("click", openSyntheticModal);
  }

  if (openSyntheticModalInlineButton) {
    openSyntheticModalInlineButton.addEventListener("click", openSyntheticModal);
  }

  if (closeSyntheticModalButton) {
    closeSyntheticModalButton.addEventListener("click", closeSyntheticModal);
  }

  if (syntheticModal) {
    syntheticModal.addEventListener("click", (event) => {
      if (event.target === syntheticModal) {
        closeSyntheticModal();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && syntheticModal && !syntheticModal.classList.contains("hidden")) {
      closeSyntheticModal();
    }
  });
}

function activateControlTab(targetId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabTarget === targetId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });

  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetId);
  });
}

function setupControlTabListeners() {
  if (!tabButtons.length || !tabPanels.length) return;
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activateControlTab(button.dataset.tabTarget);
    });
  });
}

function renderDatasetPreview(payload) {
  if (!payload || !payload.columns || payload.columns.length === 0) {
    populateRunColumnSelectors([]);
    datasetPreview.classList.add("empty");
    datasetPreview.innerHTML = "<p>Preview unavailable for this dataset.</p>";
    return;
  }

  populateRunColumnSelectors(payload.columns || []);
  datasetPreview.classList.remove("empty");
  datasetPreview.innerHTML = `
    <div class="kv-list">
      <div><dt>Dataset</dt><dd>${escapeHtml(payload.path)}</dd></div>
      <div><dt>Columns</dt><dd>${payload.columns.length}</dd></div>
      <div><dt>Preview rows</dt><dd>${(payload.preview_rows || []).length}</dd></div>
    </div>
    ${createTable(payload.columns, payload.preview_rows || [], payload.columns.length)}
  `;
}

function setDatasetPreviewVisibility(visible) {
  isDatasetPreviewVisible = visible;
  if (datasetPreview) {
    datasetPreview.classList.toggle("hidden", !visible);
  }
  if (datasetPreviewToggle) {
    datasetPreviewToggle.textContent = visible ? "Hide Dataset Preview" : "Show Dataset Preview";
  }
}

function refreshDatasetPreviewForCurrentSelection() {
  if (uploadInput?.files?.length > 0) {
    previewUploadedCsv(uploadInput.files[0]);
    return;
  }
  loadDatasetPreview(datasetSelect?.value || "");
}

setDatasetPreviewVisibility(false);
populateRunColumnSelectors([]);

async function loadDatasetPreview(path) {
  if (!path) {
    populateRunColumnSelectors([]);
    datasetPreview.classList.add("empty");
    datasetPreview.innerHTML = "<p>Select or upload a dataset to inspect the first few columns.</p>";
    return;
  }

  const res = await fetch(`/api/datasets/preview?path=${encodeURIComponent(path)}`);
  if (!res.ok) {
    populateRunColumnSelectors([]);
    datasetPreview.classList.add("empty");
    datasetPreview.innerHTML = "<p>Could not load dataset preview.</p>";
    return;
  }

  const payload = await res.json();
  renderDatasetPreview(payload);
}

function previewUploadedCsv(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result || "");
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0).slice(0, 6);
    if (!lines.length) {
      populateRunColumnSelectors([]);
      datasetPreview.classList.add("empty");
      datasetPreview.innerHTML = `<p>Upload selected: <strong>${escapeHtml(file.name)}</strong>.</p>`;
      return;
    }

    const headers = lines[0].split(",").map((item) => item.trim());
    const rows = lines.slice(1).map((line) => {
      const cells = line.split(",");
      const row = {};
      headers.forEach((header, index) => {
        row[header] = cells[index] ?? "";
      });
      return row;
    });

    populateRunColumnSelectors(headers);
    datasetPreview.classList.remove("empty");
    datasetPreview.innerHTML = `
      <div class="kv-list">
        <div><dt>Upload</dt><dd>${escapeHtml(file.name)}</dd></div>
        <div><dt>Columns</dt><dd>${headers.length}</dd></div>
      </div>
      ${createTable(headers, rows, 8)}
    `;
  };
  reader.readAsText(file.slice(0, 4096));
}

function renderSummary(run) {
  const config = run.config || {};
  const result = run.result || {};
  const items = [
    ["Status", run.status],
    ["Dataset", config.dataset_label || "Unknown"],
    ["Problem", result.problem_type || config.problem_type || "Auto"],
    ["Target", result.target_column || config.target_column || "Auto"],
  ];

  runSummary.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="summary-item">
          <span>${escapeHtml(label)}</span>
          <strong>${formatValue(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderMetrics(run) {
  const result = run.result || {};
  const metrics = result.analysis_metrics || {};
  const cards = [
    ["Model", metrics.model_name],
    ["Accuracy", metrics.accuracy],
    ["R²", metrics.r2],
    ["MSE", metrics.mse],
    ["Recommendations", result.recommendation_count],
    ["Duration (s)", result.total_duration_seconds],
    ["Pretrained", metrics.from_pretrained],
    ["Cached", metrics.from_cache],
  ];

  runMetrics.innerHTML = cards
    .filter(([, value]) => value !== undefined)
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <span>${escapeHtml(label)}</span>
          <strong>${formatValue(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderStageProgress(run) {
  const stages = new Map((run.stages || []).map((stage) => [stage.key, stage]));
  stageProgress.innerHTML = STAGE_LABELS.map(([key, label], index) => {
    const stage = stages.get(key);
    const status = stage ? stage.status : "queued";
    const helper = stage?.duration_seconds
      ? `${formatValue(stage.duration_seconds)}s`
      : status === "queued"
        ? "Waiting"
        : status === "running"
          ? "In progress"
          : status === "failed"
            ? "Needs attention"
            : "Done";
    return `
      <div class="progress-step ${status}">
        <span class="step-index">${index + 1}</span>
        <strong>${escapeHtml(label)}</strong>
        <p>${escapeHtml(helper)}</p>
      </div>
    `;
  }).join("");
}

function renderKvBlock(obj) {
  const entries = Object.entries(obj || {});
  if (entries.length === 0) {
    return "<p>No details recorded.</p>";
  }

  return `
    <dl class="kv-list">
      ${entries
        .map(
          ([key, value]) => `
            <div>
              <dt>${escapeHtml(key.replaceAll("_", " "))}</dt>
              <dd>${formatValue(value)}</dd>
            </div>
          `,
        )
        .join("")}
    </dl>
  `;
}

function renderPreview(preview) {
  if (!preview || Object.keys(preview).length === 0) {
    return "<p>No preview available.</p>";
  }

  const chunks = [];
  if (preview.shape) {
    chunks.push(`<p><strong>Shape:</strong> ${preview.shape[0]} × ${preview.shape[1]}</p>`);
  }
  const predictionSample = preview.predictions_preview || preview.prediction_preview;
  if (predictionSample && predictionSample.length) {
    chunks.push(`<p><strong>Prediction sample:</strong> ${predictionSample.map(formatValue).join(", ")}</p>`);
  }
  if (preview.top_feature_importance && preview.top_feature_importance.length) {
    chunks.push(`
      <p><strong>Top feature importance</strong></p>
      <ul>
        ${preview.top_feature_importance
          .map((item) => `<li>${escapeHtml(item.feature)}: ${formatValue(item.importance)}</li>`)
          .join("")}
      </ul>
    `);
  }
  if (preview.columns && preview.rows) {
    chunks.push(createTable(preview.columns, preview.rows, 6));
  }

  return chunks.join("");
}

function renderStages(run) {
  const stages = (run.stages || []).filter((stage) => stage.key !== "synthetic");
  if (stages.length === 0) {
    pipelineStages.innerHTML = '<div class="stage-card"><p>The pipeline trace will appear here once the run starts.</p></div>';
    return;
  }

  pipelineStages.innerHTML = stages
    .map(
      (stage) => `
        <article class="stage-card">
          <div class="stage-top">
            <div>
              <h3>${escapeHtml(stage.title)}</h3>
              <p>${stage.duration_seconds ? `${formatValue(stage.duration_seconds)}s` : "Running..."}</p>
            </div>
            <span class="status-pill ${escapeHtml(stage.status)}">${escapeHtml(stage.status)}</span>
          </div>
          <div class="stage-grid">
            <div class="subcard">
              <h4>Input</h4>
              ${renderKvBlock(stage.input_summary)}
            </div>
            <div class="subcard">
              <h4>Output</h4>
              ${renderKvBlock(stage.output_summary)}
            </div>
            <div class="subcard">
              <h4>Metrics</h4>
              ${renderKvBlock(stage.metrics)}
            </div>
            <div class="subcard">
              <h4>Preview</h4>
              ${renderPreview(stage.preview)}
            </div>
          </div>
          ${
            stage.notes && stage.notes.length
              ? `<div class="subcard" style="margin-top: 12px;"><h4>Notes</h4>${stage.notes
                  .map((note) => `<p>${escapeHtml(note)}</p>`)
                  .join("")}</div>`
              : ""
          }
        </article>
      `,
    )
    .join("");
}

function renderLogs(run) {
  const lines = (run.logs || [])
    .map((entry) => `<div class="log-line">${escapeHtml(entry.timestamp)}  ${escapeHtml(entry.message)}</div>`)
    .join("");
  logFeed.innerHTML = lines || "<div class='log-line'>No logs yet.</div>";
  logFeed.scrollTop = logFeed.scrollHeight;
}

function renderResults(run) {
  const result = run.result;
  if (!result) {
    resultOverview.classList.add("empty");
    resultOverview.innerHTML = "<p>Run the pipeline to see model metrics, recommendations, and downloadable outputs.</p>";
    recommendationPreview.classList.add("empty");
    recommendationPreview.innerHTML = "<p>Recommendation rows will appear here after a successful run.</p>";
  } else {
    resultOverview.classList.remove("empty");
    resultOverview.innerHTML = `
      <div class="kv-list">
        <div><dt>Total duration</dt><dd>${formatValue(result.total_duration_seconds)}s</dd></div>
        <div><dt>Problem type</dt><dd>${formatValue(result.problem_type)}</dd></div>
        <div><dt>Target column</dt><dd>${formatValue(result.target_column)}</dd></div>
        <div><dt>Recommendations</dt><dd>${formatValue(result.recommendation_count)}</dd></div>
      </div>
      <div class="subcard" style="margin-top: 12px;">
        <h4>Executive Summary</h4>
        <p>${formatMultiline(result.summary || "No summary available.")}</p>
      </div>
    `;

    if (result.recommendations_preview && result.recommendations_preview.columns?.length) {
      recommendationPreview.classList.remove("empty");
      recommendationPreview.innerHTML = `
        <h4>Recommendation Preview</h4>
        ${createTable(result.recommendations_preview.columns, result.recommendations_preview.rows || [], 6)}
      `;
    } else {
      recommendationPreview.classList.add("empty");
      recommendationPreview.innerHTML = "<p>No recommendation rows were generated for this run.</p>";
    }
  }

  const artifacts = run.artifacts || [];
  artifactList.innerHTML = artifacts
    .map(
      (artifact) => `
        <li>
          <a href="/api/files?path=${encodeURIComponent(artifact.path)}" target="_blank" rel="noreferrer">
            <span>${escapeHtml(artifact.label)}</span>
            <span class="artifact-kind">${escapeHtml(artifact.kind)}</span>
          </a>
        </li>
      `,
    )
    .join("");
}

function renderHistory(runs) {
  if (!runs.length) {
    runHistory.innerHTML = '<div class="run-history-empty">No runs yet. Start one from the runner.</div>';
    return;
  }

  runHistory.innerHTML = runs
    .map((run) => {
      const isActive = run.id === activeRunId;
      return `
        <button class="run-history-item ${isActive ? "active" : ""}" data-run-id="${escapeHtml(run.id)}">
          <strong>${escapeHtml(run.config?.dataset_label || run.id)}</strong>
          <small>${escapeHtml(run.status)} · ${escapeHtml(run.created_at)}</small>
          <small>${escapeHtml(run.result?.problem_type || run.config?.problem_type || "auto")}</small>
        </button>
      `;
    })
    .join("");

  [...runHistory.querySelectorAll(".run-history-item")].forEach((button) => {
    button.addEventListener("click", () => {
      activeRunId = button.dataset.runId;
      if (pollHandle) clearInterval(pollHandle);
      pollHandle = setInterval(pollRun, 1200);
      pollRun();
    });
  });
}

async function refreshHistory() {
  const res = await fetch("/api/runs");
  if (!res.ok) return;
  const runs = await res.json();
  renderHistory(runs);
}

function updateRun(run) {
  runStatusCopy.textContent = `Run ${run.id} is currently ${run.status}.`;
  renderSummary(run);
  renderMetrics(run);
  renderStageProgress(run);
  renderStages(run);
  renderLogs(run);
  renderResults(run);
}

async function pollRun() {
  if (!activeRunId) {
    await refreshHistory();
    return;
  }
  const res = await fetch(`/api/runs/${activeRunId}`);
  if (!res.ok) return;
  const run = await res.json();
  updateRun(run);
  await refreshHistory();
  if (run.status === "completed" || run.status === "failed") {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

datasetSelect.addEventListener("change", (event) => {
  if (uploadInput.value) uploadInput.value = "";
  loadDatasetPreview(event.target.value);
});

uploadInput.addEventListener("change", () => {
  if (uploadInput.files.length > 0) {
    datasetSelect.value = "";
    previewUploadedCsv(uploadInput.files[0]);
  } else {
    loadDatasetPreview(datasetSelect.value);
  }
});

if (runTargetColumnSelect) {
  runTargetColumnSelect.addEventListener("change", () => {
    reconcileFeatureSelectionWithTarget();
  });
}

if (runFeatureColumnsSelect) {
  runFeatureColumnsSelect.addEventListener("change", () => {
    syncFeatureColumnsHiddenField();
  });
}

if (runProblemTypeSelect) {
  runProblemTypeSelect.addEventListener("change", () => {
    populateRunModelOptions();
  });
}

if (runTrainModeSelect) {
  runTrainModeSelect.addEventListener("change", () => {
    populateRunModelOptions();
  });
}

if (datasetPreviewToggle) {
  datasetPreviewToggle.addEventListener("click", () => {
    const nextVisible = !isDatasetPreviewVisible;
    setDatasetPreviewVisibility(nextVisible);
    if (nextVisible) {
      refreshDatasetPreviewForCurrentSelection();
    }
  });
}

runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  syncFeatureColumnsHiddenField();
  const submitButton = runForm.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.textContent = "Starting...";

  try {
    const formData = new FormData(runForm);
    const response = await fetch("/api/runs", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      window.alert(err.detail || "Could not start the run.");
      return;
    }

    const payload = await response.json();
    activeRunId = payload.run_id;
    if (pollHandle) clearInterval(pollHandle);
    runStatusCopy.textContent = `Run ${activeRunId} is starting...`;
    pollHandle = setInterval(pollRun, 1200);
  } catch (error) {
    window.alert(error?.message || "Could not start the run.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Launch Run";
  }
});

function renderQualityScoreBadge(score) {
  let color, label;
  if (score >= 85) {
    color = "#4caf50";
    label = "Excellent";
  } else if (score >= 70) {
    color = "#8bc34a";
    label = "Good";
  } else if (score >= 50) {
    color = "#ff9800";
    label = "Fair";
  } else {
    color = "#f44336";
    label = "Poor";
  }
  return `
    <div style="display: flex; align-items: center; gap: 12px;">
      <div style="
        width: 60px; 
        height: 60px; 
        border-radius: 50%; 
        background: ${color}; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        color: white; 
        font-weight: bold; 
        font-size: 20px;
      ">${score.toFixed(1)}</div>
      <div>
        <div style="font-size: 14px; font-weight: bold; color: ${color};">${label}</div>
        <div style="font-size: 12px; color: #666;">Synthetic Data Quality</div>
      </div>
    </div>
  `;
}

function renderDataQualitySection(qualityData) {
  if (!qualityData) return "";

  const healthIcons = {
    "✓ Healthy": "✓",
    "⚠ Attention needed": "⚠",
    "✓ Good": "✓",
    "⚠ Check differences": "⚠",
  };

  const healthColors = {
    "✓ Healthy": "#4caf50",
    "⚠ Attention needed": "#ff9800",
    "✓ Good": "#4caf50",
    "⚠ Check differences": "#ff9800",
  };

  const healthIndicators = qualityData.health_indicators || {};

  return `
    <article class="stage-card">
      <div class="stage-top">
        <div>
          <h3>Data Quality Analysis</h3>
          <p>Original vs Synthetic Dataset Comparison</p>
        </div>
      </div>
      <div style="margin: 20px 0;">
        ${renderQualityScoreBadge(qualityData.quality_score)}
      </div>
      <div style="margin: 20px 0; padding: 16px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #2196F3;">
        <p style="margin: 0; font-size: 14px; color: #333;">${escapeHtml(qualityData.recommendation)}</p>
      </div>
      <div class="stage-grid">
        <div class="subcard">
          <h4>Similarity Metrics</h4>
          <dl class="kv-list">
            <div>
              <dt>Numeric Similarity</dt>
              <dd>${formatValue(qualityData.summary_metrics.numeric_similarity)}</dd>
            </div>
            <div>
              <dt>Categorical Similarity</dt>
              <dd>${formatValue(qualityData.summary_metrics.categorical_similarity)}</dd>
            </div>
          </dl>
        </div>
        <div class="subcard">
          <h4>Health Status</h4>
          <dl class="kv-list">
            ${Object.entries(healthIndicators)
              .map(
                ([key, value]) => `
              <div>
                <dt>${escapeHtml(key)}</dt>
                <dd style="color: ${healthColors[value]}; font-weight: bold;">
                  ${escapeHtml(value)}
                </dd>
              </div>
            `,
              )
              .join("")}
          </dl>
        </div>
        <div class="subcard">
          <h4>Dataset Info</h4>
          <dl class="kv-list">
            <div>
              <dt>Original Rows</dt>
              <dd>${formatValue(qualityData.summary_metrics.total_original_rows)}</dd>
            </div>
            <div>
              <dt>Synthetic Rows</dt>
              <dd>${formatValue(qualityData.summary_metrics.total_synthetic_rows)}</dd>
            </div>
          </dl>
        </div>
      </div>
      ${
        qualityData.top_column_differences && qualityData.top_column_differences.length > 0
          ? `
        <div class="subcard" style="margin-top: 16px;">
          <h4>Top Column Differences</h4>
          <ul style="list-style: none; padding: 0; margin: 0;">
            ${qualityData.top_column_differences
              .map(
                (item) => `
              <li style="padding: 8px 0; border-bottom: 1px solid #eee;">
                <strong>${escapeHtml(item.column)}</strong>: 
                <span style="color: #ff9800; font-weight: bold;">
                  ${formatValue(item.mean_difference_pct)}% difference
                </span>
              </li>
            `,
              )
              .join("")}
          </ul>
        </div>
      `
          : ""
      }
    </article>
  `;
}

function renderPredictionAnalysisSection(predictionData) {
  if (!predictionData) return "";

  const analysis = predictionData.analysis || {};
  const recommendations = predictionData.recommendations || [];

  let performanceMetrics = [];
  if (predictionData.problem_type === "classification") {
    performanceMetrics = [
      ["Total Predictions", analysis.total_predictions],
      ["Unique Classes", analysis.unique_classes],
      ["Accuracy", analysis.accuracy],
      ["Precision", analysis.precision],
      ["Recall", analysis.recall],
      ["F1 Score", analysis.f1_score],
    ];
  } else if (predictionData.problem_type === "regression") {
    performanceMetrics = [
      ["Total Predictions", analysis.total_predictions],
      ["Mean Prediction", analysis.mean_prediction],
      ["Median Prediction", analysis.median_prediction],
      ["Std Dev", analysis.std_prediction],
      ["RMSE", analysis.rmse],
      ["MAE", analysis.mae],
      ["R² Score", analysis.r2_score],
    ];
  }

  return `
    <article class="stage-card">
      <div class="stage-top">
        <div>
          <h3>Prediction Analysis</h3>
          <p>Model Performance on Synthetic Data</p>
        </div>
        <span class="status-pill" style="background: #2196F3; color: white;">
          ${escapeHtml(predictionData.problem_type.toUpperCase())}
        </span>
      </div>
      <div class="stage-grid">
        <div class="subcard">
          <h4>Performance Metrics</h4>
          <dl class="kv-list">
            ${performanceMetrics
              .filter(([, value]) => value !== undefined && value !== null)
              .map(
                ([label, value]) => `
              <div>
                <dt>${escapeHtml(label)}</dt>
                <dd>${formatValue(value)}</dd>
              </div>
            `,
              )
              .join("")}
          </dl>
        </div>
        ${
          predictionData.problem_type === "classification" && analysis.class_distribution
            ? `
          <div class="subcard">
            <h4>Class Distribution</h4>
            <ul style="list-style: none; padding: 0; margin: 0;">
              ${Object.entries(analysis.class_distribution)
                .map(
                  ([cls, count]) => `
                <li style="padding: 8px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #eee;">
                  <strong>${escapeHtml(cls)}</strong>
                  <span style="color: #2196F3;">${formatValue(count)} (${formatValue(analysis.class_percentages[cls])}%)</span>
                </li>
              `,
                )
                .join("")}
            </ul>
          </div>
        `
            : ""
        }
      </div>
      <div class="subcard" style="margin-top: 16px;">
        <h4>Recommendations</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
          ${recommendations
            .map(
              (rec) => {
                let bgColor, borderColor, textColor;
                if (rec.severity === "success") {
                  bgColor = "#e8f5e9";
                  borderColor = "#4caf50";
                  textColor = "#2e7d32";
                } else if (rec.severity === "error") {
                  bgColor = "#ffebee";
                  borderColor = "#f44336";
                  textColor = "#c62828";
                } else if (rec.severity === "warning") {
                  bgColor = "#fff3e0";
                  borderColor = "#ff9800";
                  textColor = "#e65100";
                } else {
                  bgColor = "#e3f2fd";
                  borderColor = "#2196F3";
                  textColor = "#1565c0";
                }
                return `
                <li style="
                  padding: 12px;
                  margin: 8px 0;
                  background: ${bgColor};
                  border-left: 4px solid ${borderColor};
                  border-radius: 4px;
                ">
                  <div style="color: ${textColor}; font-weight: bold; font-size: 12px; margin-bottom: 4px;">
                    ${escapeHtml(rec.category)}
                  </div>
                  <div style="color: ${textColor}; font-size: 13px;">
                    ${escapeHtml(rec.message)}
                  </div>
                </li>
              `;
              }
            )
            .join("")}
        </ul>
      </div>
    </article>
  `;
}

function renderSyntheticDashboard(stage) {
  if (!stage || stage.key !== "synthetic") return "";

  const output = stage.output_summary || {};
  const preview = stage.preview || {};

  let dashboard = "";

  // Data Quality Section
  if (preview.data_quality) {
    dashboard += renderDataQualitySection(preview.data_quality);
  }

  // Prediction Analysis Section
  if (preview.predictions_analysis) {
    dashboard += renderPredictionAnalysisSection(preview.predictions_analysis);
  }

  return dashboard;
}


// ===== SYNTHETIC DATA WORKFLOW =====

let syntheticForm = null;
let syntheticList = null;
let syntheticDatasetSelect = null;
let syntheticTargetColumnSelect = null;
let synthRefreshHandle = null;
let syntheticListenersBound = false;

function startSyntheticPolling() {
  refreshSyntheticList();
  if (synthRefreshHandle) return;
  synthRefreshHandle = setInterval(refreshSyntheticList, 5000);
}

function stopSyntheticPolling() {
  if (!synthRefreshHandle) return;
  clearInterval(synthRefreshHandle);
  synthRefreshHandle = null;
}

function initSyntheticElements() {
  syntheticForm = document.getElementById("synthetic-form");
  syntheticList = document.getElementById("synthetic-list");
  syntheticDatasetSelect = document.getElementById("synthetic-dataset-select");
  syntheticTargetColumnSelect = document.getElementById("synthetic-target-column");
  
  if (!syntheticDatasetSelect || !syntheticTargetColumnSelect) {
    return false;
  }
  
  return true;
}

async function populateSyntheticTargetColumns(datasetPath) {
  if (!syntheticTargetColumnSelect) {
    if (!initSyntheticElements()) return;
  }

  if (!datasetPath) {
    syntheticTargetColumnSelect.innerHTML = '<option value="">Select a source dataset first</option>';
    return;
  }

  try {
    const encodedPath = encodeURIComponent(datasetPath);
    const url = `/api/datasets/preview?path=${encodedPath}`;
    const res = await fetch(url);
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const payload = await res.json();
    const columns = Array.isArray(payload.columns) ? payload.columns : [];
    const selectedValue = syntheticTargetColumnSelect.value;

    if (!columns.length) {
      syntheticTargetColumnSelect.innerHTML = '<option value="">No columns found in dataset</option>';
      return;
    }

    syntheticTargetColumnSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select target column";
    syntheticTargetColumnSelect.appendChild(placeholder);

    columns.forEach((col) => {
      const opt = document.createElement("option");
      opt.value = String(col);
      opt.textContent = String(col);
      syntheticTargetColumnSelect.appendChild(opt);
    });

    if ([...syntheticTargetColumnSelect.options].some((opt) => opt.value === selectedValue)) {
      syntheticTargetColumnSelect.value = selectedValue;
    }
  } catch (err) {
    console.error("[Synthetic] Error loading columns:", err);
    syntheticTargetColumnSelect.innerHTML = '<option value="">Error: ' + err.message + '</option>';
  }
}

function setupSyntheticEventListeners() {
  if (!initSyntheticElements()) return;
  if (syntheticListenersBound) return;

  if (syntheticDatasetSelect) {
    const onSyntheticDatasetSelection = (e) => {
      const selectedPath = syntheticDatasetSelect.value || "";
      populateSyntheticTargetColumns(selectedPath);
    };
    
    syntheticDatasetSelect.addEventListener("change", onSyntheticDatasetSelection);
  }

  if (syntheticProblemTypeSelect) {
    syntheticProblemTypeSelect.addEventListener("change", () => {
      populateSyntheticModelOptions();
    });
  }

  if (syntheticForm) {
    syntheticForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const datasetPath = syntheticDatasetSelect?.value?.trim();
      const problemType = syntheticForm.problem_type?.value?.trim() || "classification";
      const preferredModel = syntheticForm.preferred_model?.value?.trim() || "";
      const nRows = parseInt(syntheticForm.n_rows.value, 10);
      const targetColumn = syntheticTargetColumnSelect?.value?.trim() || "";
      const seed = parseInt(syntheticForm.seed.value, 10);

      if (!datasetPath) {
        alert("Please select a dataset.");
        return;
      }
      
      if (!targetColumn) {
        alert("Please select a target column.");
        return;
      }

      if (!nRows || nRows < 10 || nRows > 10000) {
        alert("Invalid row count (10-10000).");
        return;
      }

      syntheticForm.querySelectorAll("button").forEach((btn) => (btn.disabled = true));

      try {
        const formData = new FormData();
        formData.append("dataset_path", datasetPath);
        formData.append("n_rows", String(nRows));
        formData.append("problem_type", problemType);
        formData.append("target_column", targetColumn);
        if (preferredModel) formData.append("preferred_model", preferredModel);
        formData.append("seed", String(Number.isFinite(seed) ? seed : 42));

        const resp = await fetch("/api/synthetic/generate", {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          throw new Error(`Generate failed: ${resp.statusText}`);
        }

        const data = await resp.json();
        alert(`✓ Synthetic generation started (ID: ${data.synthetic_id})`);
        await refreshDatasetDropdowns();
        refreshSyntheticList();
      } catch (err) {
        alert(`Error: ${err.message}`);
      } finally {
        syntheticForm.querySelectorAll("button").forEach((btn) => (btn.disabled = false));
      }
    });
  }

  syntheticListenersBound = true;
}

// Initialize synthetic workflow when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    setupModalEventListeners();
    setupControlTabListeners();
    initSyntheticElements();
    setupSyntheticEventListeners();
  });
} else {
  setupModalEventListeners();
  setupControlTabListeners();
  initSyntheticElements();
  setupSyntheticEventListeners();
}

async function refreshSyntheticList() {
  if (!syntheticList) return;

  try {
    const resp = await fetch("/api/synthetic/list");
    if (!resp.ok) throw new Error(resp.statusText);
    const datasets = await resp.json();

    if (!datasets || datasets.length === 0) {
      syntheticList.innerHTML = "<p>No synthetic datasets generated yet.</p>";
      return;
    }

    syntheticList.innerHTML = datasets
      .map((ds) => {
        const status = ds.status || "unknown";
        const statusColor =
          status === "complete" ? "#4CAF50" : status === "ready_for_inference" ? "#FFC107" : status === "failed" ? "#F44336" : "#2196F3";
        const genResult = ds.generation_result || {};
        const infResult = ds.inference_result || {};
        const config = ds.config || {};
        const syntheticLabel = ds.name || ds.id;
        const quality = genResult.data_quality || null;
        const safeId = String(ds.id || "").replaceAll("'", "\\'");

        let controls = `<button class="small-button danger" onclick="deleteDataset('${safeId}')">Delete</button>`;
        if (status === "ready_for_inference") {
          controls += ` <button class="small-button primary" onclick="runInference('${safeId}')">Run Inference</button>`;
        }
        if (status === "running_inference" || status === "inference_failed") {
          controls += ` <button class="small-button" onclick="getInferenceResult('${safeId}')">Check Result</button>`;
        }

        return `
          <article class="synthetic-item">
            <div class="synthetic-item-top">
              <strong class="synthetic-item-title">${escapeHtml(syntheticLabel)}</strong>
              <span class="synthetic-status" style="background: ${statusColor};">${escapeHtml(status)}</span>
            </div>
            <div class="synthetic-meta">
              ID: ${escapeHtml(ds.id)}<br />
              Created: ${new Date(ds.created_at).toLocaleString()}<br />
              Signature: ${escapeHtml(ds.signature || "pending")}
            </div>
            <div class="synthetic-source">
              Source: ${escapeHtml(config.source_dataset || "unknown")}<br />
              Problem type: ${escapeHtml(config.problem_type || "unknown")}<br />
              Target column: ${escapeHtml(config.target_column || "all columns")}
            </div>
            ${genResult.n_rows ? `<div><strong>Generated rows:</strong> ${genResult.n_rows} | <strong>Columns:</strong> ${genResult.n_columns}</div>` : ""}
            ${quality ? `
              <div class="synthetic-inference">
                <strong>Original vs Synthetic Quality</strong><br />
                Score: ${formatValue(quality.quality_score)}<br />
                Numeric similarity: ${formatValue(quality.summary_metrics?.numeric_similarity)}<br />
                Categorical similarity: ${formatValue(quality.summary_metrics?.categorical_similarity)}
              </div>
            ` : ""}
            ${genResult.preview ? `<div class="synthetic-preview">${createTable(genResult.columns || [], genResult.preview, 5)}</div>` : ""}
            ${infResult.model_name ? `<div class="synthetic-inference"><strong>Inference:</strong> ${infResult.model_name} | Predictions: ${infResult.n_predictions}</div>` : ""}
            ${ds.error ? `<div class="synthetic-error"><strong>Error:</strong> ${escapeHtml(ds.error)}</div>` : ""}
            <div class="synthetic-controls">${controls}</div>
          </article>
        `;
      })
      .join("");
  } catch (err) {
    syntheticList.innerHTML = `<p style="color: red;">Error loading datasets: ${err.message}</p>`;
  }
}

async function refreshDatasetDropdowns() {
  try {
    const resp = await fetch("/api/datasets");
    if (!resp.ok) return;

    const datasets = await resp.json();
    const dropdowns = [datasetSelect, syntheticDatasetSelect].filter(Boolean);
    dropdowns.forEach((dropdown) => {
      const currentValue = dropdown.value;
      const currentLabel = dropdown.id === "dataset-select" ? "Choose a dataset from /data" : "Choose a dataset to generate synthetic data from";
      dropdown.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = currentLabel;
      dropdown.appendChild(placeholder);

      datasets.forEach((dataset) => {
        const option = document.createElement("option");
        option.value = String(dataset.path || "");
        const category = dataset.category ? ` [${dataset.category}]` : "";
        option.textContent = `${dataset.directory} / ${dataset.label}${category}`;
        dropdown.appendChild(option);
      });

      if ([...dropdown.options].some((option) => option.value === currentValue)) {
        dropdown.value = currentValue;
      }
    });

    // Populate target columns if a dataset is already selected
    if (syntheticDatasetSelect?.value) {
      await populateSyntheticTargetColumns(syntheticDatasetSelect.value);
    }
    if (datasetSelect?.value) {
      await loadDatasetPreview(datasetSelect.value);
    } else if (!uploadInput?.files?.length) {
      populateRunColumnSelectors([]);
    }
  } catch (err) {
    console.warn("Failed to refresh dataset dropdowns", err);
  }
}

async function runInference(syntheticId) {
  try {
    const detailsResp = await fetch(`/api/synthetic/${syntheticId}`);
    if (!detailsResp.ok) {
      throw new Error("Could not fetch synthetic dataset details before inference.");
    }
    const details = await detailsResp.json();
    const config = details.config || {};

    if (!config.problem_type || !config.target_column) {
      throw new Error("Synthetic dataset is missing problem_type or target_column. Regenerate dataset with a selected target.");
    }

    const formData = new FormData();
    formData.append("problem_type", config.problem_type);
    formData.append("target_column", config.target_column);
    if (config.preferred_model) {
      formData.append("preferred_model", config.preferred_model);
    }

    const resp = await fetch(`/api/synthetic/${syntheticId}/infer`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const payload = await resp.json();
        detail = payload.detail || detail;
      } catch {
        // Ignore parsing failures and keep status text
      }
      throw new Error(detail);
    }
    alert("✓ Inference started. Checking results...");
    pollInferenceResult(syntheticId);
  } catch (err) {
    alert(`Inference error: ${err.message}`);
  }
}

function pollInferenceResult(syntheticId, maxAttempts = 30) {
  let attempts = 0;

  const checkResult = async () => {
    try {
      const resp = await fetch(`/api/synthetic/${syntheticId}/infer-status`);
      if (!resp.ok) throw new Error(resp.statusText);

      const data = await resp.json();
      const status = data.status || "unknown";

      if (status === "complete" || status === "inference_failed") {
        clearInterval(pollInterval);
        alert(`Inference ${status === "complete" ? "completed" : "failed"}!`);
        refreshSyntheticList();
        return;
      }

      attempts++;
      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        alert("Inference still running. Check back later.");
        refreshSyntheticList();
      }
    } catch (err) {
      clearInterval(pollInterval);
      alert(`Poll error: ${err.message}`);
    }
  };

  const pollInterval = setInterval(checkResult, 1000);
}

async function getInferenceResult(syntheticId) {
  try {
    const resp = await fetch(`/api/synthetic/${syntheticId}/infer-status`);
    if (!resp.ok) throw new Error(resp.statusText);

    const data = await resp.json();
    alert(`Status: ${data.status}\nInference: ${data.inference_result ? JSON.stringify(data.inference_result, null, 2) : "Pending"}`);
    refreshSyntheticList();
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

async function deleteDataset(syntheticId) {
  if (!confirm("Delete this synthetic dataset?")) return;

  // Currently no delete endpoint, but can be added later
  alert("Delete functionality not yet implemented.");
}

// Refresh synthetic list on page load and periodically
refreshDatasetDropdowns();
loadModelCatalog();

refreshHistory();
