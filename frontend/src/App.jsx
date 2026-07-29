import React, { useEffect, useRef, useState } from "react";
import Canvas from "./Canvas.jsx";
import ChatRail from "./ChatRail.jsx";
import { CARD_WIDTH } from "./cards.jsx";
import * as api from "./api.js";

const MODES = [
  { id: "stylist", label: "Stylist" },
  { id: "designer", label: "Designer" },
  { id: "buyer", label: "Buyer" },
];

const STARTERS = {
  stylist: [
    { eyebrow: "Styling", prompt: "Style me for a gallery opening — I like The Row and Lemaire." },
    { eyebrow: "Lineage", prompt: "What is Bottega Veneta known for?" },
    { eyebrow: "Trend", prompt: "Is quiet luxury still trending in 2026?" },
  ],
  designer: [
    { eyebrow: "Brand DNA", prompt: "Define the brand DNA for a minimalist Scandinavian menswear label." },
    { eyebrow: "Lineage", prompt: "Trace the design lineage of Jil Sander." },
    { eyebrow: "Trend", prompt: "Rate the momentum of utilitarian tailoring for next season." },
  ],
  buyer: [
    { eyebrow: "Trend", prompt: "Which silhouettes are rising for Fall 2026?" },
    { eyebrow: "Trend", prompt: "Rate the durability of the 'boho revival' trend." },
    { eyebrow: "Lineage", prompt: "How are Celine and Dior connected?" },
  ],
};

export const EST_H = { style: 340, brand_dna: 320, trend: 280, lineage: 240, look: 260, answer: 150, source: 96, image: 300, text: 150, sketch: 400, starter: 150 };
const W = { image: 300, text: 280, sketch: 300 };
const COL_W = 372, GAP = 26, START_X = 24, START_Y = 24;
const STORE = "fg_canvas_v3";

let _id = Date.now();
const uid = () => `n${++_id}`;
const loadStore = () => { try { return JSON.parse(localStorage.getItem(STORE) || "null"); } catch { return null; } };
const estH = (n) => n.h || EST_H[n.type] || EST_H[n.kind] || 180;

function fileToDataUrl(file, max = 560) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const s = Math.min(1, max / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        resolve({ src: c.toDataURL("image/jpeg", 0.86), ratio: img.height / img.width });
      };
      img.onerror = reject; img.src = reader.result;
    };
    reader.onerror = reject; reader.readAsDataURL(file);
  });
}

function rasterizeSketch(node) {
  const w = node.w, h = node.h || EST_H.sketch;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.fillStyle = "#ffffff"; g.fillRect(0, 0, w, h);
  g.strokeStyle = "#141414"; g.lineWidth = 2.6; g.lineJoin = "round"; g.lineCap = "round";
  for (const stroke of node.strokes || []) {
    g.beginPath();
    stroke.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.stroke();
  }
  return c.toDataURL("image/png");
}

export default function App() {
  const saved = loadStore();
  const [mode, setMode] = useState(saved?.mode || "stylist");
  const [tool, setTool] = useState("select");
  const [nodes, setNodes] = useState(saved?.nodes || []);
  const [edges, setEdges] = useState(saved?.edges || []);
  const [messages, setMessages] = useState(saved?.messages || []);
  const [selectedId, setSelectedId] = useState(null);
  const [sending, setSending] = useState(false);
  const [ok, setOk] = useState(false);
  const [focus, setFocus] = useState(null);
  const cols = useRef(saved?.cols || [START_Y, START_Y, START_Y]);
  const seeded = useRef((saved?.nodes || []).some((n) => n.kind !== "starter"));

  useEffect(() => { api.health().then((h) => setOk(!!h)); }, []);
  useEffect(() => {
    try { localStorage.setItem(STORE, JSON.stringify({ mode, nodes, edges, messages, cols: cols.current })); } catch {}
  }, [mode, nodes, edges, messages]);

  useEffect(() => {
    if (seeded.current || nodes.length) return;
    setNodes((STARTERS[mode] || []).map((s, i) => ({
      id: uid(), kind: "starter", type: "answer", w: COL_W, x: START_X + i * (COL_W + GAP), y: START_Y, ...s,
    })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  const nextSlot = (type) => {
    if (!seeded.current) { cols.current = [START_Y, START_Y, START_Y]; seeded.current = true; }
    const c = cols.current.indexOf(Math.min(...cols.current));
    const x = START_X + c * (COL_W + GAP), y = cols.current[c];
    cols.current[c] += (EST_H[type] || 200) + GAP;
    return { x, y };
  };
  const clearStarters = (prev) => (seeded.current ? prev : prev.filter((n) => n.kind !== "starter"));

  const focusOn = (list) => {
    if (!list?.length) return;
    const x = Math.min(...list.map((n) => n.x));
    const y = Math.min(...list.map((n) => n.y));
    const right = Math.max(...list.map((n) => n.x + (n.w || 320)));
    setFocus({ box: { x, y, w: right - x }, nonce: Date.now() });
  };

  const placeCards = (cards) => {
    const added = (cards || []).map((card) => {
      const { x, y } = nextSlot(card.type);
      return { id: uid(), kind: "card", type: card.type, card, w: CARD_WIDTH[card.type] || 320, x, y, enter: true };
    });
    setNodes((prev) => [...clearStarters(prev), ...added]);
    return added;
  };

  const addEdge = (a, b) => {
    if (a === b) return;
    setEdges((prev) => (prev.some((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a)) ? prev
      : [...prev, { id: uid(), a, b }]));
  };
  const linkTo = (src, list) => list.forEach((n) => addEdge(src, n.id));
  const removeEdge = (id) => setEdges((prev) => prev.filter((e) => e.id !== id));

  const groupOf = (id) => {
    const seen = new Set([id]); const stack = [id];
    while (stack.length) {
      const cur = stack.pop();
      for (const e of edges) {
        const nxt = e.a === cur ? e.b : e.b === cur ? e.a : null;
        if (nxt && !seen.has(nxt)) { seen.add(nxt); stack.push(nxt); }
      }
    }
    return [...seen];
  };

  const updateNode = (id, patch) => setNodes((prev) => prev.map((n) => (n.id === id ? { ...n, ...patch } : n)));

  const runAgent = async (message, selection = [], opts = {}) => {
    if (message) setMessages((m) => [...m, { role: "user", text: message }]);
    setSending(true);
    try {
      const res = await api.agent({ message, selection });
      if (opts.replaceId && res.cards?.length) {
        const fresh = res.cards.find((c) => c.type === opts.expectType) || res.cards[0];
        updateNode(opts.replaceId, { card: fresh, enter: false });
      } else {
        const added = placeCards(res.cards);
        if (opts.sourceId) linkTo(opts.sourceId, added);
        focusOn(added);
      }
      setMessages((m) => [...m, { role: "ai", text: res.answer, steps: res.trace, learned: res.learned }]);
    } catch {
      setMessages((m) => [...m, { role: "ai", text: "The atelier is unreachable — is the backend running on :8000?" }]);
    } finally { setSending(false); }
  };

  const summarise = (card) => card.text
    ? card.text.slice(0, 220)
    : Object.entries(card).filter(([k]) => !["type", "raw"].includes(k))
        .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.slice(0, 4).join(", ") : typeof v === "object" ? JSON.stringify(v).slice(0, 60) : v}`).join("; ").slice(0, 300);

  const placeUser = (node) => setNodes((prev) => [...clearStarters(prev), node]);

  const addImage = async (file) => {
    try {
      const { src, ratio } = await fileToDataUrl(file);
      const w = W.image, h = Math.round(w * (ratio || 1.25));
      const pos = nextSlot("image"); const id = uid();
      placeUser({ id, kind: "image", src, w, h, x: pos.x, y: pos.y, enter: true });
      focusOn([{ x: pos.x, y: pos.y, w }]);
      reviewImage(id, src);
    } catch {}
  };

  const reviewImage = async (id, src) => {
    setSending(true); setMessages((m) => [...m, { role: "user", text: "Review this look ↑" }]);
    try {
      const res = await api.analyzeDataUrl(src);
      const added = placeCards([res.card]); linkTo(id, added); focusOn(added);
      setMessages((m) => [...m, { role: "ai", text: res.review }]);
    } catch { setMessages((m) => [...m, { role: "ai", text: "Couldn't review that image — check the backend." }]); }
    finally { setSending(false); }
  };

  const analyzeSketch = async (id) => {
    const n = nodes.find((x) => x.id === id);
    if (!n || !(n.strokes || []).length) return;
    setSending(true); setMessages((m) => [...m, { role: "user", text: "Analyze this silhouette ↑" }]);
    try {
      const res = await api.analyzeDataUrl(rasterizeSketch(n), "hand-drawn silhouette sketch");
      const added = placeCards([res.card]); linkTo(id, added); focusOn(added);
      setMessages((m) => [...m, { role: "ai", text: res.review }]);
    } catch { setMessages((m) => [...m, { role: "ai", text: "Couldn't read that sketch — check the backend." }]); }
    finally { setSending(false); }
  };

  const analyzeGroup = async (id) => {
    const ids = groupOf(id);
    const members = nodes.filter((n) => ids.includes(n.id));
    const images = members.filter((n) => n.kind === "image").map((n) => n.src);
    const note = members.filter((n) => n.kind === "text").map((n) => n.text).filter(Boolean).join(" · ");
    if (images.length) {
      setSending(true); setMessages((m) => [...m, { role: "user", text: `Analyze this group of ${members.length} ↑` }]);
      try {
        const res = await api.compose(images, note);
        const added = placeCards([res.card]); linkTo(id, added); focusOn(added);
        setMessages((m) => [...m, { role: "ai", text: res.answer }]);
      } catch { setMessages((m) => [...m, { role: "ai", text: "Couldn't compose the group — check the backend." }]); }
      finally { setSending(false); }
    } else {
      const selection = members.map((n) => n.kind === "text" ? { type: "note", text: n.text }
        : n.kind === "card" ? { type: n.type, text: summarise(n.card), data: n.card }
        : { type: n.kind, text: n.kind });
      runAgent("Analyze these together as one set — how they relate and work as a whole.", selection, { sourceId: id });
    }
  };

  const onPlace = (t, x, y) => {
    const id = uid();
    if (t === "text") { placeUser({ id, kind: "text", text: "", w: W.text, x, y, editing: true }); setSelectedId(id); }
    else if (t === "sketch") { placeUser({ id, kind: "sketch", strokes: [], w: W.sketch, h: EST_H.sketch, x, y }); setSelectedId(id); }
    setTool("select");
  };

  const onAction = (id, action) => {
    const n = nodes.find((x) => x.id === id); if (!n) return;
    if (action === "dismiss") {
      setNodes((p) => p.filter((x) => x.id !== id));
      setEdges((p) => p.filter((e) => e.a !== id && e.b !== id));
      setSelectedId(null); return;
    }
    if (action === "group") return analyzeGroup(id);
    if (n.kind === "card" && action === "explore") runAgent("Expand on this and suggest a next step.", [{ type: n.type, text: summarise(n.card), data: n.card }], { sourceId: id });
    if (n.kind === "image" && action === "review") reviewImage(id, n.src);
    if (n.kind === "sketch" && action === "analyze") analyzeSketch(id);
    if (n.kind === "sketch" && action === "clear") updateNode(id, { strokes: [] });
    if (n.kind === "text" && action === "style") runAgent(`Style advice around this: ${(n.text || "").trim()}`, [{ type: "note", text: (n.text || "").trim() }], { sourceId: id });
    if (n.kind === "text" && action === "explore") runAgent(`Tell me more about: ${(n.text || "").trim()}`, [{ type: "note", text: (n.text || "").trim() }], { sourceId: id });
  };

  const onRefine = (id, instruction) => {
    const n = nodes.find((x) => x.id === id); if (!n || n.kind !== "card") return;
    runAgent(instruction, [{ type: n.type, text: summarise(n.card), data: n.card }], { replaceId: id, expectType: n.type });
  };

  const clearBoard = () => {
    if (!confirm("Clear the whole board?")) return;
    cols.current = [START_Y, START_Y, START_Y]; seeded.current = false;
    setNodes([]); setEdges([]); setSelectedId(null);
  };

  return (
    <div className="app">
      <div>
        <header className="topbar">
          <div className="brand"><span className="mark" /><h1>Fashion<em>Graph</em></h1></div>
          <div className="modes" role="tablist">
            {MODES.map((m) => (
              <button key={m.id} className="mode-pill" data-on={mode === m.id} onClick={() => setMode(m.id)}>{m.label}</button>
            ))}
          </div>
          <div className="status"><span className="dot" data-ok={ok} /><span className="label">{ok ? "Atelier live" : "Offline"}</span></div>
        </header>

        <Canvas
          nodes={nodes} edges={edges} focus={focus}
          tool={tool} setTool={setTool}
          selectedId={selectedId} onSelect={setSelectedId} onDeselect={() => setSelectedId(null)}
          onStarter={(n) => runAgent(n.prompt, [])}
          onPlace={onPlace} onUpload={addImage} onDropImage={addImage}
          onUpdateNode={updateNode} onAction={onAction} onRefine={onRefine}
          onAddEdge={addEdge} onDeleteEdge={removeEdge} onClear={clearBoard}
        />
      </div>

      <ChatRail messages={messages} sending={sending} onSend={(t) => runAgent(t, [])} />
    </div>
  );
}
