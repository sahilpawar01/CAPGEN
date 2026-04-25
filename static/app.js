const form = document.getElementById("form");
const fileInput = document.getElementById("file");
const submitBtn = document.getElementById("submit");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const saveRow = document.getElementById("saveRow");
const saveCsvBtn = document.getElementById("saveCsv");
const saveXlsxBtn = document.getElementById("saveXlsx");
const tableWrap = document.getElementById("tableWrap");
const resultsBody = document.getElementById("resultsBody");

const API = "";

/** @type {{ filename: string, caption: string, error: string }[]} */
let lastRows = [];

function setSaveEnabled(on) {
  saveCsvBtn.disabled = !on;
  saveXlsxBtn.disabled = !on;
  saveRow.hidden = !on;
}

function renderTable(rows) {
  resultsBody.replaceChildren();
  for (const r of rows) {
    const tr = document.createElement("tr");
    const fd = document.createElement("td");
    fd.textContent = r.filename;
    const cd = document.createElement("td");
    cd.textContent = r.caption;
    const ed = document.createElement("td");
    ed.textContent = r.error || "";
    ed.className = r.error ? "cell-error" : "";
    tr.append(fd, cd, ed);
    resultsBody.append(tr);
  }
  tableWrap.hidden = rows.length === 0;
}

function escapeCsvField(s) {
  const t = String(s);
  if (/[",\r\n]/.test(t)) return `"${t.replace(/"/g, '""')}"`;
  return t;
}

function downloadText(filename, text, mime) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function saveAsCsv() {
  if (lastRows.length === 0) return;
  const header = "Filename,Caption,Error";
  const lines = [header];
  for (const r of lastRows) {
    lines.push(
      [escapeCsvField(r.filename), escapeCsvField(r.caption), escapeCsvField(r.error || "")].join(
        ","
      )
    );
  }
  const bom = "\uFEFF";
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  downloadText(`captions-${stamp}.csv`, bom + lines.join("\r\n"), "text/csv;charset=utf-8");
}

async function saveAsXlsx() {
  if (lastRows.length === 0) return;
  const res = await fetch(`${API}/api/export/xlsx`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      rows: lastRows.map((r) => ({
        filename: r.filename,
        caption: r.caption,
        error: r.error || "",
      })),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err.detail || res.statusText || "Export failed";
    const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
    throw new Error(detail);
  }
  const blob = await res.blob();
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `captions-${stamp}.xlsx`;
  a.click();
  URL.revokeObjectURL(a.href);
}

fileInput.addEventListener("change", () => {
  const files = fileInput.files;
  submitBtn.disabled = !files || files.length === 0;
  lastRows = [];
  setSaveEnabled(false);
  tableWrap.hidden = true;
  resultsBody.replaceChildren();
  preview.replaceChildren();
  preview.hidden = !files || files.length === 0;
  if (!files || files.length === 0) return;
  for (const f of files) {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.alt = f.name;
    img.src = URL.createObjectURL(f);
    wrap.append(img);
    preview.append(wrap);
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = fileInput.files;
  if (!files || files.length === 0) return;

  statusEl.className = "status loading";
  statusEl.textContent = `Running BLIP on ${files.length} image(s)…`;
  setSaveEnabled(false);
  tableWrap.hidden = true;
  submitBtn.disabled = true;

  const body = new FormData();
  for (const f of files) {
    body.append("files", f);
  }

  try {
    const res = await fetch(`${API}/api/caption/batch`, {
      method: "POST",
      body,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail || data.message || res.statusText || "Request failed";
      const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
      throw new Error(detail);
    }
    const items = data.items || [];
    lastRows = items.map((it) => ({
      filename: it.filename || "",
      caption: it.caption || "",
      error: it.error || "",
    }));
    statusEl.className = "status";
    statusEl.textContent =
      lastRows.length > 0
        ? `Done: ${lastRows.length} row(s).`
        : "";
    renderTable(lastRows);
    setSaveEnabled(lastRows.length > 0);
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = err instanceof Error ? err.message : "Something went wrong";
    lastRows = [];
    setSaveEnabled(false);
  } finally {
    submitBtn.disabled = !fileInput.files?.length;
  }
});

saveCsvBtn.addEventListener("click", () => {
  try {
    saveAsCsv();
  } catch (e) {
    statusEl.className = "status error";
    statusEl.textContent = e instanceof Error ? e.message : "CSV save failed";
  }
});

saveXlsxBtn.addEventListener("click", async () => {
  statusEl.className = "status loading";
  statusEl.textContent = "Building Excel file…";
  try {
    await saveAsXlsx();
    statusEl.className = "status";
    statusEl.textContent = lastRows.length ? `Done: ${lastRows.length} row(s).` : "";
  } catch (e) {
    statusEl.className = "status error";
    statusEl.textContent = e instanceof Error ? e.message : "Excel save failed";
  }
});
