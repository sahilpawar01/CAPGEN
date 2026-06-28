const API = "";
const SESSION_KEY = "capgen_caption_batch";

const elErr = document.getElementById("analyzeError");
const elLoad = document.getElementById("analyzeLoading");
const elMain = document.getElementById("analyzeMain");
const elContext = document.getElementById("analyzeContextLine");
const elInsights = document.getElementById("insightList");
const elMetrics = document.getElementById("metricGrid");
const elOutWrap = document.getElementById("outlierDupWrap");
const elOutPanel = document.getElementById("outlierPanel");
const elOutCards = document.getElementById("outlierCards");
const elDupPanel = document.getElementById("dupPanel");
const elDupSummary = document.getElementById("dupSummary");
const elDupList = document.getElementById("dupList");
const elDetail = document.getElementById("analyzeDetailBody");
const btnExport = document.getElementById("btnExportJson");

/** @type {any} */
let ch1 = null;
/** @type {any} */
let ch2 = null;
/** @type {any} */
let ch3 = null;
/** @type {any} */
let ch4 = null;

/** @type {any} */
let lastApiPayload = null;

if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#8b9cb3";
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.1)";
  Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
}

function destroyCharts() {
  for (const c of [ch1, ch2, ch3, ch4]) {
    if (c) c.destroy();
  }
  ch1 = ch2 = ch3 = ch4 = null;
}

function addMetric(k, v) {
  const d = document.createElement("div");
  d.className = "metric-card";
  d.innerHTML = `<span class="metric-k"></span><span class="metric-v"></span>`;
  d.querySelector(".metric-k").textContent = k;
  d.querySelector(".metric-v").textContent = v;
  elMetrics.append(d);
}

function run() {
  elErr.hidden = true;
  elLoad.hidden = false;
  elMain.hidden = true;
  let raw = null;
  try {
    raw = sessionStorage.getItem(SESSION_KEY);
  } catch (_) {
    raw = null;
  }
  if (!raw) {
    elLoad.hidden = true;
    elErr.hidden = false;
    elErr.textContent =
      "No caption batch found. Go to the home page, run Get captions, then click Analyze again.";
    return;
  }
  let rows;
  try {
    rows = JSON.parse(raw);
  } catch {
    elLoad.hidden = true;
    elErr.hidden = false;
    elErr.textContent = "Could not read stored results. Return home and run captions again.";
    return;
  }
  if (!Array.isArray(rows) || rows.length === 0) {
    elLoad.hidden = true;
    elErr.hidden = false;
    elErr.textContent = "No rows to analyze. Add images and get captions on the home page first.";
    return;
  }
  const ok = rows.some(
    (r) => String(r.caption || "").trim() && !String(r.error || "").trim()
  );
  if (!ok) {
    elLoad.hidden = true;
    elErr.hidden = false;
    elErr.textContent =
      "All rows have errors. Fix uploads and re-run Get captions, then return here to analyze.";
    return;
  }

  void (async () => {
    try {
      const res = await fetch(`${API}/api/caption/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = data.detail || data.message || res.statusText || "Analysis failed";
        const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
        throw new Error(detail);
      }
      lastApiPayload = data;
      if (!data.n_captions) {
        elLoad.hidden = true;
        elErr.hidden = false;
        elErr.textContent = data.message || "No successful captions to analyze.";
        return;
      }
      elLoad.hidden = true;
      elErr.hidden = true;
      elMain.hidden = false;
      renderAll(data, rows);
    } catch (e) {
      elLoad.hidden = true;
      elErr.hidden = false;
      elErr.textContent = e instanceof Error ? e.message : "Request failed";
    }
  })();
}

function renderAll(data) {
  elMetrics.replaceChildren();
  elInsights.replaceChildren();
  elDetail.replaceChildren();
  destroyCharts();

  const n = data.n_captions;
  const t = data.n_rows_total ?? n;
  const bq = data.batch_quality || {};
  const voc = data.vocabulary || {};
  const wcs = data.word_count_stats || {};
  const cl = data.caption_length || {};
  const ss = data.sentiment_summary || {};

  elContext.textContent = `Batch: ${t} file(s) · ${n} successful caption(s) for text stats · ${(bq.success_rate_pct ?? 0) + "%"
    } of rows ingested without error. VADER = word-level writing tone, not an image content label.`;

  for (const it of data.insights || []) {
    const li = document.createElement("li");
    const t0 = (it.title || "Insight").replace(/[<>]/g, "");
    const t1 = (it.text || "").replace(/[<>]/g, "");
    li.innerHTML = `<strong></strong> `;
    li.querySelector("strong").textContent = t0;
    li.append(document.createTextNode(t1));
    elInsights.append(li);
  }

  addMetric("Ingestion success", `${bq.success_rate_pct ?? "—"}%`);
  addMetric("Failed rows", String(bq.failed_rows ?? "—"));
  addMetric("Analyzed lines", String(n));
  addMetric("Unique content words", String(voc.unique_content_words ?? "—"));
  addMetric("Type–token ratio", String(voc.type_token_ratio ?? "—"));
  addMetric(
    "Words (mean · median)",
    `${cl.words_mean ?? "—"} · ${cl.words_median == null ? "—" : Number(cl.words_median).toFixed(1)}`
  );
  addMetric("Word count p25 – p75", `${wcs.p25 ?? "—"} – ${wcs.p75 ?? "—"}`);
  addMetric("Word count (min – max · σ)", `${wcs.min ?? "—"} – ${wcs.max ?? "—"} · ${wcs.stdev ?? "—"}`);
  addMetric("Chars (mean · σ range)", `${cl.chars_mean ?? "—"}`);
  addMetric("VADER (mean compound)", String(ss.mean_compound ?? "—"));
  addMetric("VADER (pos / neu / neg)", [ss.mean_pos, ss.mean_neu, ss.mean_neg].map((x) => (x == null ? "—" : x)).join(" / "));

  const dup = data.duplicates || {};
  if (data.outliers) {
    elOutPanel.hidden = false;
    elOutCards.replaceChildren();
    const o = data.outliers;
    for (const kind of ["longest", "shortest"]) {
      const x = o[kind];
      if (!x) continue;
      const d = document.createElement("div");
      d.className = "outlier-card";
      d.innerHTML = `<span class="tag"></span><div class="name"></div><p class="cap"></p>`;
      d.querySelector(".tag").textContent = kind;
      d.querySelector(".name").textContent = `${x.filename} · ${x.word_count} words`;
      d.querySelector(".cap").textContent = x.caption || "—";
      elOutCards.append(d);
    }
  } else {
    elOutPanel.hidden = true;
  }

  if (dup.n_duplicate_lines) {
    elDupPanel.hidden = false;
    elDupSummary.textContent = `${dup.n_duplicate_lines} distinct caption(s) appear more than once; ${dup.n_files_in_duplicate_captions ?? 0} file rows are part of a duplicate set.`;
    elDupList.replaceChildren();
    for (const g of (dup.groups || []).slice(0, 12)) {
      const li = document.createElement("li");
      const c = (g.caption || "").slice(0, 200);
      const rest = (g.caption || "").length > 200 ? "…" : "";
      li.append(`${g.count}×: “${c}${rest}” — `);
      li.append(String((g.files || []).join(", ")));
      elDupList.append(li);
    }
  } else {
    elDupPanel.hidden = true;
  }
  elOutWrap.hidden = elOutPanel.hidden && elDupPanel.hidden;

  for (const p of data.per_caption || []) {
    const tr = document.createElement("tr");
    const a = document.createElement("td");
    a.textContent = p.filename || "";
    const b = document.createElement("td");
    b.textContent = p.caption || "";
    const c = document.createElement("td");
    c.textContent = String(p.word_count ?? "");
    const d = document.createElement("td");
    const s = document.createElement("span");
    s.className = "vader-sub";
    s.textContent = `p ${p.pos} · n ${p.neu} · ng ${p.neg}`;
    d.append(String(p.compound ?? ""));
    d.append(s);
    const e = document.createElement("td");
    e.textContent = String(p.label || "");
    e.className =
      p.label === "negative" ? "cell-error" : p.label === "positive" ? "cell-ok" : "";
    tr.append(a, b, c, d, e);
    elDetail.append(tr);
  }

  if (typeof Chart === "undefined") {
    return;
  }

  const topW = (data.top_words || []).slice(0, 16);
  const bgs = (data.top_bigrams || []).slice(0, 14);
  const bins = data.word_count_bins || {};
  const byLabel = ss.by_label || {};
  const elTW = document.getElementById("chTopWords");
  const elBG = document.getElementById("chBigrams");
  const elWB = document.getElementById("chWordBins");
  const elST = document.getElementById("chSent");
  if (topW.length && elTW) {
    ch1 = new Chart(elTW, {
      type: "bar",
      data: {
        labels: topW.map((x) => x.word),
        datasets: [
          {
            label: "Count",
            data: topW.map((x) => x.count),
            backgroundColor: "rgba(61, 139, 253, 0.65)",
            borderColor: "rgba(61, 139, 253, 0.9)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.08)" } },
          y: { grid: { display: false } },
        },
      },
    });
  }
  if (bgs.length && elBG) {
    ch2 = new Chart(elBG, {
      type: "bar",
      data: {
        labels: bgs.map((x) => `${x.w1} ${x.w2}`),
        datasets: [
          {
            label: "Count",
            data: bgs.map((x) => x.count),
            backgroundColor: "rgba(167, 139, 250, 0.55)",
            borderColor: "rgba(167, 139, 250, 0.9)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.08)" } },
          y: { grid: { display: false } },
        },
      },
    });
  }
  if (Object.keys(bins).length && elWB) {
    const labels = Object.keys(bins);
    ch3 = new Chart(elWB, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Captions",
            data: labels.map((k) => bins[k]),
            backgroundColor: "rgba(52, 211, 153, 0.55)",
            borderColor: "rgba(52, 211, 153, 0.85)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: "rgba(255,255,255,0.08)" } },
          x: { grid: { display: false } },
        },
      },
    });
  }
  if (Object.keys(byLabel).length && elST) {
    const lab = ["positive", "neutral", "negative"];
    const colors = ["rgba(52, 211, 153, 0.75)", "rgba(148, 163, 184, 0.6)", "rgba(248, 113, 113, 0.7)"];
    const borders = ["rgb(34, 197, 94)", "rgb(148, 163, 184)", "rgb(248, 113, 113)"];
    ch4 = new Chart(elST, {
      type: "bar",
      data: {
        labels: lab.map((k) => k[0].toUpperCase() + k.slice(1)),
        datasets: [
          {
            label: "Captions",
            data: lab.map((k) => byLabel[k] ?? 0),
            backgroundColor: colors,
            borderColor: borders,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: "rgba(255,255,255,0.08)" } },
          x: { grid: { display: false } },
        },
      },
    });
  }
}

run();

btnExport?.addEventListener("click", () => {
  if (!lastApiPayload) return;
  const text = JSON.stringify(lastApiPayload, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `caption-analysis-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});
