const runForm = document.getElementById("run-form");
const datasetSelect = document.getElementById("dataset-select");
const uploadInput = document.getElementById("upload-input");
const datasetPreview = document.getElementById("dataset-preview");
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

let activeRunId = null;
let pollHandle = null;

const STAGE_LABELS = [
  ["resolve", "Resolve"],
  ["load", "Load"],
  ["preprocess", "Preprocess"],
  ["analyze", "Analyze"],
  ["optimize", "Recommend"],
  ["summary", "Summarize"],
  ["synthetic", "Synthetic"],
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

function renderDatasetPreview(payload) {
  if (!payload || !payload.columns || payload.columns.length === 0) {
    datasetPreview.classList.add("empty");
    datasetPreview.innerHTML = "<p>Preview unavailable for this dataset.</p>";
    return;
  }

  datasetPreview.classList.remove("empty");
  datasetPreview.innerHTML = `
    <div class="kv-list">
      <div><dt>Dataset</dt><dd>${escapeHtml(payload.path)}</dd></div>
      <div><dt>Columns</dt><dd>${payload.columns.length}</dd></div>
      <div><dt>Preview rows</dt><dd>${(payload.preview_rows || []).length}</dd></div>
    </div>
    ${createTable(payload.columns, payload.preview_rows || [], 8)}
  `;
}

async function loadDatasetPreview(path) {
  if (!path) {
    datasetPreview.classList.add("empty");
    datasetPreview.innerHTML = "<p>Select or upload a dataset to inspect the first few columns.</p>";
    return;
  }

  const res = await fetch(`/api/datasets/preview?path=${encodeURIComponent(path)}`);
  if (!res.ok) {
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
  const stages = run.stages || [];
  if (stages.length === 0) {
    pipelineStages.innerHTML = '<div class="stage-card"><p>The pipeline trace will appear here once the run starts.</p></div>';
    return;
  }

  let html = stages
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

  // Add synthetic data dashboard if synthetic stage exists and has completed
  const syntheticStage = stages.find(s => s.key === "synthetic");
  if (syntheticStage && syntheticStage.status === "completed") {
    const syntheticDashboard = renderSyntheticDashboard(syntheticStage);
    if (syntheticDashboard) {
      html += syntheticDashboard;
    }
  }

  pipelineStages.innerHTML = html;
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
  }
});

runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
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
              `,
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


refreshHistory();
