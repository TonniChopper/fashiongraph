// Thin client for the FashionGraph backend. Same-origin by default (the backend
// serves this app in production); in dev, Vite proxies the API paths → :8000.
const BASE = import.meta.env.VITE_API_BASE ?? "";

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

// Streamed agent run (SSE over POST): live step/card events, then final.
export async function agentStream(body, h) {
  const res = await fetch(`${BASE}/agent/stream`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new Error(`/agent/stream → ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 2);
      const data = line.split("\n").find((l) => l.startsWith("data:"));
      if (!data) continue;
      let ev; try { ev = JSON.parse(data.slice(5).trim()); } catch { continue; }
      if (ev.type === "step") h.onStep?.(ev.text);
      else if (ev.type === "card") h.onCard?.(ev.card);
      else if (ev.type === "final") h.onFinal?.(ev);
      else if (ev.type === "error") h.onError?.(ev.message);
    }
  }
}

// Boards — server-side save/load.
export const listBoards = () => fetch(`${BASE}/boards`).then((r) => (r.ok ? r.json() : []));
export const getBoard = (id) => fetch(`${BASE}/boards/${id}`).then((r) => r.json());
export const saveBoard = (name, state) => post("/boards", { name, state });
export const deleteBoard = (id) => fetch(`${BASE}/boards/${id}`, { method: "DELETE" }).then((r) => r.json());

export async function health() {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}
