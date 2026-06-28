const form = document.getElementById("form");
const fileInput = document.getElementById("file");
const submitBtn = document.getElementById("submit");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const captionTimerEl = document.getElementById("captionTimer");
const saveRow = document.getElementById("saveRow");
const saveCsvBtn = document.getElementById("saveCsv");
const saveXlsxBtn = document.getElementById("saveXlsx");
const tableWrap = document.getElementById("tableWrap");
const resultsBody = document.getElementById("resultsBody");
const camToggle = document.getElementById("camToggle");
const camPanel = document.getElementById("camPanel");
const camVideo = document.getElementById("camVideo");
const camCanvas = document.getElementById("camCanvas");
const camLiveText = document.getElementById("camLiveText");
const camSpeak = document.getElementById("camSpeak");
const camSubStatus = document.getElementById("camSubStatus");
const camMeta = document.getElementById("camMeta");
const camSaveServer = document.getElementById("camSaveServer");
const analyzeBtn = document.getElementById("analyzeBtn");

const API = "";
const ANALYZE_SESSION_KEY = "capgen_caption_batch";

/** Rough server-side time per image for batch captions (~2.5s each). */
const CAPGEN_EST_SEC_PER_IMAGE = 2.5;

const CAPTURE_MS = 2500;
const JPEG_Q = 0.88;
const camCtx = camCanvas.getContext("2d");

/** @type {MediaStream | null} */
let camStream = null;
/** @type {ReturnType<typeof setInterval> | null} */
let camTimer = null;
let camBusy = false;

/** @type {{ filename: string, caption: string, error: string }[]} */
let lastRows = [];

/** @type {ReturnType<typeof setInterval> | null} */
let captionCountdownInterval = null;

function hasAnalyzableCaptions() {
  return lastRows.some((r) => String(r.caption || "").trim() && !String(r.error || "").trim());
}

function setAnalyzeButtonState() {
  if (analyzeBtn) analyzeBtn.disabled = !hasAnalyzableCaptions();
}

function setSaveEnabled(on) {
  saveCsvBtn.disabled = !on;
  saveXlsxBtn.disabled = !on;
  saveRow.hidden = !on;
}

function formatRemainingTime(sec) {
  const s = Math.max(0, sec);
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, "0")}`;
  }
  if (s >= 10) return `${Math.floor(s)} s`;
  return `${s.toFixed(1)} s`;
}

function stopCaptionCountdown() {
  if (captionCountdownInterval !== null) {
    clearInterval(captionCountdownInterval);
    captionCountdownInterval = null;
  }
  if (captionTimerEl) {
    captionTimerEl.hidden = true;
    captionTimerEl.textContent = "";
  }
}

function startCaptionCountdown(nImages) {
  stopCaptionCountdown();
  if (!captionTimerEl) return;
  const totalSec = nImages * CAPGEN_EST_SEC_PER_IMAGE;
  const end = performance.now() + totalSec * 1000;
  captionTimerEl.hidden = false;
  captionTimerEl.className = "status-timer status-timer--active";
  const tick = () => {
    const left = Math.max(0, (end - performance.now()) / 1000);
    if (left <= 0) {
      captionTimerEl.textContent = "Almost done (past estimate)…";
      if (captionCountdownInterval !== null) {
        clearInterval(captionCountdownInterval);
        captionCountdownInterval = null;
      }
      return;
    }
    captionTimerEl.textContent = `Est. time left: ${formatRemainingTime(left)}`;
  };
  tick();
  captionCountdownInterval = setInterval(tick, 100);
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
  setAnalyzeButtonState();
  stopCaptionCountdown();
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
  statusEl.textContent = `Running CAPGEN on ${files.length} image(s)…`;
  setSaveEnabled(false);
  tableWrap.hidden = true;
  submitBtn.disabled = true;
  startCaptionCountdown(files.length);

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
    setAnalyzeButtonState();
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = err instanceof Error ? err.message : "Something went wrong";
    lastRows = [];
    setSaveEnabled(false);
    setAnalyzeButtonState();
  } finally {
    stopCaptionCountdown();
    submitBtn.disabled = !fileInput.files?.length;
  }
});

analyzeBtn?.addEventListener("click", () => {
  if (!hasAnalyzableCaptions()) return;
  try {
    sessionStorage.setItem(
      ANALYZE_SESSION_KEY,
      JSON.stringify(
        lastRows.map((r) => ({
          filename: r.filename,
          caption: r.caption,
          error: r.error || "",
        }))
      )
    );
  } catch {
    statusEl.className = "status error";
    statusEl.textContent = "Could not store results (session storage blocked or full).";
    return;
  }
  window.location.assign("/analyze");
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

function stopCameraSpeech() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  if (camSpeak) {
    camSpeak.classList.remove("btn-cam-speak--active");
  }
}

function getCamSpeakableText() {
  const t = (camLiveText.textContent || "").trim();
  if (!t) return null;
  if (t === "—" || t === "-") return null;
  if (/^starting/i.test(t) || /^warming up/i.test(t)) return null;
  return t;
}

function updateCamSpeakButton() {
  if (!camSpeak) return;
  if (!window.speechSynthesis) {
    camSpeak.disabled = true;
    camSpeak.title = "Text-to-speech is not available in this browser.";
    return;
  }
  const t = getCamSpeakableText();
  const hasText = Boolean(t);
  camSpeak.disabled = !hasText;
  camSpeak.title = hasText
    ? "Read the current caption aloud. Click again to stop."
    : "Caption will appear here; then you can have it read aloud.";
}

function speakCamCaption() {
  if (!window.speechSynthesis) {
    return;
  }
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    camSpeak.classList.remove("btn-cam-speak--active");
    return;
  }
  const text = getCamSpeakableText();
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-GB";
  u.rate = 0.95;
  u.onstart = () => {
    if (camSpeak) camSpeak.classList.add("btn-cam-speak--active");
  };
  u.onend = () => {
    if (camSpeak) camSpeak.classList.remove("btn-cam-speak--active");
  };
  u.onerror = () => {
    if (camSpeak) camSpeak.classList.remove("btn-cam-speak--active");
  };
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

function stopCamera() {
  stopCameraSpeech();
  if (camTimer) {
    clearInterval(camTimer);
    camTimer = null;
  }
  if (camStream) {
    for (const t of camStream.getTracks()) t.stop();
    camStream = null;
  }
  camVideo.srcObject = null;
  camPanel.hidden = true;
  camMeta.hidden = true;
  camToggle.textContent = "Open camera";
  camBusy = false;
  camLiveText.textContent = "—";
  camSubStatus.textContent = "";
  if (camSpeak) {
    camSpeak.disabled = true;
    camSpeak.classList.remove("btn-cam-speak--active");
  }
}

async function sendFrameToCapgen() {
  if (camBusy || !camStream) return;
  const w = camVideo.videoWidth;
  const h = camVideo.videoHeight;
  if (!w || !h) return;
  camCanvas.width = w;
  camCanvas.height = h;
  camCtx.drawImage(camVideo, 0, 0, w, h);
  const blob = await new Promise((resolve) =>
    camCanvas.toBlob((b) => resolve(b), "image/jpeg", JPEG_Q)
  );
  if (!blob) {
    camSubStatus.textContent = "Could not capture frame.";
    return;
  }
  camBusy = true;
  camSubStatus.className = "cam-sub";
  camSubStatus.textContent = "Asking CAPGEN…";
  const body = new FormData();
  body.append("file", blob, "webcam-frame.jpg");
  if (camSaveServer.checked) {
    body.append("save", "true");
  }
  try {
    const res = await fetch(`${API}/api/caption`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail || data.message || res.statusText || "Request failed";
      const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
      throw new Error(detail);
    }
    stopCameraSpeech();
    camLiveText.textContent = data.caption || "(no caption)";
    if (data.save_error) {
      camSubStatus.className = "cam-sub cam-sub-error";
      camSubStatus.textContent = data.save_error;
    } else if (camSaveServer.checked && data.saved) {
      camSubStatus.className = "cam-sub";
      camSubStatus.textContent = `Saved: ${data.saved.image || ""}`;
    } else {
      camSubStatus.className = "cam-sub";
      camSubStatus.textContent = "";
    }
  } catch (e) {
    camSubStatus.className = "cam-sub cam-sub-error";
    camSubStatus.textContent = e instanceof Error ? e.message : "Caption failed";
  } finally {
    camBusy = false;
    updateCamSpeakButton();
  }
}

async function startCamera() {
  camSubStatus.className = "cam-sub";
  camSubStatus.textContent = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    camSubStatus.className = "cam-sub cam-sub-error";
    camSubStatus.textContent = "Camera not available (use HTTPS or localhost, or a modern browser).";
    return;
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (e) {
    camSubStatus.className = "cam-sub cam-sub-error";
    camSubStatus.textContent =
      e instanceof Error ? e.message : "Could not open camera (check permission).";
    return;
  }
  camVideo.srcObject = camStream;
  try {
    await camVideo.play();
  } catch (_) {
    /* play() may need gesture on some devices */
  }
  camPanel.hidden = false;
  camMeta.hidden = false;
  camToggle.textContent = "Stop camera";
  camLiveText.textContent = "Warming up…";
  updateCamSpeakButton();
  setTimeout(() => {
    void sendFrameToCapgen();
  }, 300);
  camTimer = setInterval(() => {
    void sendFrameToCapgen();
  }, CAPTURE_MS);
}

camToggle.addEventListener("click", () => {
  if (camStream) {
    stopCamera();
  } else {
    void startCamera();
  }
});

camSpeak?.addEventListener("click", () => {
  speakCamCaption();
});

window.addEventListener("beforeunload", () => {
  stopCamera();
});
