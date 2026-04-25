const form = document.getElementById("form");
const fileInput = document.getElementById("file");
const submitBtn = document.getElementById("submit");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

const API = ""; // same origin

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  submitBtn.disabled = !f;
  resultEl.hidden = true;
  if (!f) {
    preview.hidden = true;
    return;
  }
  previewImg.src = URL.createObjectURL(f);
  preview.hidden = false;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = fileInput.files && fileInput.files[0];
  if (!f) return;

  statusEl.className = "status loading";
  statusEl.textContent = "Running BLIP (first request may be slow)…";
  resultEl.hidden = true;
  submitBtn.disabled = true;

  const body = new FormData();
  body.append("file", f);

  try {
    const res = await fetch(`${API}/api/caption`, {
      method: "POST",
      body,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail || data.message || res.statusText || "Request failed";
      const detail = typeof msg === "string" ? msg : JSON.stringify(msg);
      throw new Error(detail);
    }
    statusEl.className = "status";
    statusEl.textContent = "";
    resultEl.textContent = data.caption || "(empty)";
    resultEl.hidden = false;
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = err instanceof Error ? err.message : "Something went wrong";
  } finally {
    submitBtn.disabled = !fileInput.files?.[0];
  }
});
