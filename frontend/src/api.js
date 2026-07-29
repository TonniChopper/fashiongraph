// Thin client for the FashionGraph backend. In dev, Vite proxies /api → :8000.
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

// The one router. `selection` are the canvas cards the user is acting on (the
// referent for "this" / refine).
export function agent({ message, selection = [], learn = true, maxSteps = 4 }) {
  return post("/agent", { message, selection, learn, max_steps: maxSteps });
}

export async function analyze(file, occasion = "") {
  const form = new FormData();
  form.append("file", file, file.name || "upload.png");
  form.append("occasion", occasion);
  const res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`/analyze → ${res.status}`);
  return res.json();
}

// Analyze a data-URL (used for canvas sketches / pasted images).
export function dataUrlToBlob(dataUrl) {
  const [meta, b64] = dataUrl.split(",");
  const mime = (meta.match(/:(.*?);/) || [])[1] || "image/png";
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

export function analyzeDataUrl(dataUrl, occasion = "") {
  const blob = dataUrlToBlob(dataUrl);
  blob.name = "sketch.png";
  return analyze(blob, occasion);
}

// Compose/review several images (+ optional note) together in one vision call.
export async function compose(dataUrls = [], note = "") {
  const form = new FormData();
  dataUrls.forEach((u, i) => form.append("files", dataUrlToBlob(u), `piece${i}.jpg`));
  form.append("note", note);
  const res = await fetch(`${BASE}/compose`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`/compose → ${res.status}`);
  return res.json();
}

export async function health() {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}
