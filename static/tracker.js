/* Activity tracker page — /tracker */
const trkName = document.getElementById("trkName");
const trkRef = document.getElementById("trkRef");
const trkEnroll = document.getElementById("trkEnroll");
const trkSessionInfo = document.getElementById("trkSessionInfo");
const trkSessionIdEl = document.getElementById("trkSessionId");
const trkSave = document.getElementById("trkSave");
const trkToggle = document.getElementById("trkToggle");
const trkMeta = document.getElementById("trkMeta");
const trkPanel = document.getElementById("trkPanel");
const trkVideo = document.getElementById("trkVideo");
const trkCanvas = document.getElementById("trkCanvas");
const trkLine = document.getElementById("trkLine");
const trkSubStatus = document.getElementById("trkSubStatus");
const trkDets = document.getElementById("trkDets");
const trkCtx = trkCanvas.getContext("2d");

const API = "";
const TRK_MS = 3300;
const JPEG_Q = 0.88;

/** @type {MediaStream | null} */
let trkStream = null;
/** @type {ReturnType<typeof setInterval> | null} */
let trkTimer = null;
let trkBusy = false;
/** @type {string | null} */
let trkSession = null;

function updateTrkEnrollButton() {
  const n = (trkName.value || "").trim();
  trkEnroll.disabled = !n || !(trkRef.files && trkRef.files.length);
}

trkName.addEventListener("input", updateTrkEnrollButton);
trkRef.addEventListener("change", updateTrkEnrollButton);

trkEnroll.addEventListener("click", async () => {
  const n = (trkName.value || "").trim();
  const list = trkRef.files;
  if (!n || !list || !list.length) return;
  trkSubStatus.className = "cam-sub";
  trkSubStatus.textContent = "Enrolling (FaceNet)…";
  const body = new FormData();
  body.append("name", n);
  for (let i = 0; i < list.length; i += 1) {
    const f = list[i];
    body.append("files", f, f.name);
  }
  try {
    const res = await fetch(`${API}/api/tracker/enroll`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail || res.statusText || "Enroll failed";
      const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
      throw new Error(detail);
    }
    trkSession = data.session_id || null;
    if (trkSession) {
      trkSessionIdEl.textContent = trkSession;
      trkSessionInfo.hidden = false;
    }
    trkToggle.disabled = !trkSession;
    trkSubStatus.className = "cam-sub";
    const msg = data.message || "Enrolled. Start the activity camera when ready.";
    const w = data.photo_warnings;
    trkSubStatus.textContent = Array.isArray(w) && w.length ? `${msg} | ${w.join(" · ")}` : msg;
  } catch (e) {
    trkSubStatus.className = "cam-sub cam-sub-error";
    trkSubStatus.textContent = e instanceof Error ? e.message : "Enroll failed";
    trkSession = null;
    trkToggle.disabled = true;
  }
});

function stopTrackerCamera() {
  if (trkTimer) {
    clearInterval(trkTimer);
    trkTimer = null;
  }
  if (trkStream) {
    for (const t of trkStream.getTracks()) t.stop();
    trkStream = null;
  }
  trkVideo.srcObject = null;
  trkPanel.hidden = true;
  trkMeta.hidden = true;
  trkToggle.textContent = "Start activity camera";
  trkBusy = false;
  trkLine.textContent = "—";
  trkSubStatus.textContent = "";
  trkDets.hidden = true;
}

async function sendTrackerFrame() {
  if (trkBusy || !trkStream || !trkSession) return;
  const w = trkVideo.videoWidth;
  const h = trkVideo.videoHeight;
  if (!w || !h) return;
  trkCanvas.width = w;
  trkCanvas.height = h;
  trkCtx.drawImage(trkVideo, 0, 0, w, h);
  const blob = await new Promise((resolve) =>
    trkCtx.canvas.toBlob((b) => resolve(b), "image/jpeg", JPEG_Q)
  );
  if (!blob) {
    trkSubStatus.textContent = "Could not capture frame.";
    return;
  }
  trkBusy = true;
  trkSubStatus.className = "cam-sub";
  trkSubStatus.textContent = "Running YOLO + face match…";
  const body = new FormData();
  body.append("file", blob, "trk-frame.jpg");
  body.append("session_id", trkSession);
  if (trkSave.checked) {
    body.append("save", "true");
  }
  try {
    const res = await fetch(`${API}/api/tracker/analyze`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail || res.statusText || "Request failed";
      const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
      throw new Error(detail);
    }
    trkLine.textContent = data.line || "—";
    const dup = Boolean(data.skipped_duplicate_log);
    if (data.save_error) {
      trkSubStatus.className = "cam-sub cam-sub-error";
      trkSubStatus.textContent = data.save_error;
    } else if (dup) {
      trkSubStatus.className = "cam-sub";
      trkSubStatus.textContent = "No change to activity (same as last log) — file & log skipped";
    } else if (trkSave.checked && data.saved) {
      trkSubStatus.className = "cam-sub";
      trkSubStatus.textContent = `Activity change · saved ${data.saved.image || ""} · line appended`;
    } else {
      trkSubStatus.className = "cam-sub";
      trkSubStatus.textContent = "Activity change · line appended to log (snapshot save off or unchecked).";
    }
    if (data.detections && data.detections.length) {
      trkDets.textContent = `Detections: ${data.detections.join(", ")}`;
      trkDets.hidden = false;
    } else {
      trkDets.hidden = true;
    }
  } catch (e) {
    trkSubStatus.className = "cam-sub cam-sub-error";
    trkSubStatus.textContent = e instanceof Error ? e.message : "Analysis failed";
  } finally {
    trkBusy = false;
  }
}

async function startTrackerCamera() {
  trkSubStatus.className = "cam-sub";
  trkSubStatus.textContent = "";
  if (!trkSession) {
    trkSubStatus.className = "cam-sub cam-sub-error";
    trkSubStatus.textContent = "Enroll with name and a face photo first.";
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    trkSubStatus.className = "cam-sub cam-sub-error";
    trkSubStatus.textContent = "Camera not available (HTTPS or localhost).";
    return;
  }
  try {
    trkStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (e) {
    trkSubStatus.className = "cam-sub cam-sub-error";
    trkSubStatus.textContent = e instanceof Error ? e.message : "Could not open camera.";
    return;
  }
  trkVideo.srcObject = trkStream;
  try {
    await trkVideo.play();
  } catch (_) {
    /* need gesture on some mobile browsers */
  }
  trkPanel.hidden = false;
  trkMeta.hidden = false;
  trkToggle.textContent = "Stop activity camera";
  trkLine.textContent = "Warming up…";
  setTimeout(() => {
    void sendTrackerFrame();
  }, 400);
  trkTimer = setInterval(() => {
    void sendTrackerFrame();
  }, TRK_MS);
}

trkToggle.addEventListener("click", () => {
  if (trkStream) {
    stopTrackerCamera();
  } else {
    void startTrackerCamera();
  }
});

window.addEventListener("beforeunload", () => {
  stopTrackerCamera();
});
