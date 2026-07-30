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
  const [liveSteps, setLiveSteps] = useState([]);
  const [boards, setBoards] = useState([]);
  const [agentAt, setAgentAt] = useState(null);   // world pos the sprite runs to
  const [agentStep, setAgentStep] = useState("");
  const cols = useRef(saved?.cols || [START_Y, START_Y, START_Y]);
  const seeded = useRef((saved?.nodes || []).some((n) => n.kind !== "starter"));

  useEffect(() => { api.health().then((h) => setOk(!!h)); refreshBoards(); }, []);
  const refreshBoards = () => api.listBoards().then(setBoards).catch(() => {});
  useEffect(() => {
    try { localStorage.setItem(STORE, JSON.stringify({ mode, nodes, edges, messages, cols: cols.current })); } catch {}
  }, [mode, nodes, edges, messages]);

  useEffect(() => {
    if (seeded.current) return;   // real work exists → keep it; only reseed a fresh board
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

  // Place a single card at an explicit spot (used to cluster a response next to its source).
  const placeAtPos = (card, x, y) => {
    const node = { id: uid(), kind: "card", type: card.type, card, w: CARD_WIDTH[card.type] || 320, x, y, enter: true };
    setNodes((prev) => [...clearStarters(prev), node]);
    return node;
  };

  // Drop cards clustered beside their source and chained, so lines are short + connected.
  const clusterFrom = (sourceId, cards) => {
    const src = sourceId ? nodes.find((n) => n.id === sourceId) : null;
    let ax = src ? src.x + (src.w || 320) + 72 : null;
    let ay = src ? src.y : null;
    let prev = sourceId || null;
    const out = [];
    for (const card of cards) {
      const node = ax != null ? placeAtPos(card, ax, ay) : placeCards([card])[0];
      if (ax != null) ay += (EST_H[card.type] || 180) + 24;
      if (prev) addEdge(prev, node.id);
      prev = node.id; out.push(node);
    }
    focusOn(out);
    return out;
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
    setSending(true); setLiveSteps([]); setAgentStep("thinking");
    const steps = []; const added = []; let replaced = false;
    // Cluster the response next to its source; chain cards so every line is short + connected.
    const src = opts.sourceId ? nodes.find((n) => n.id === opts.sourceId) : null;
    let ax = src ? src.x + (src.w || 320) + 72 : null;
    let ay = src ? src.y : null;
    let prevId = opts.sourceId || null;
    const drop = (card) => {
      const node = ax != null ? placeAtPos(card, ax, ay) : placeCards([card])[0];
      if (ax != null) ay += (EST_H[card.type] || 180) + 24;
      added.push(node);
      if (prevId) addEdge(prevId, node.id);   // source → c1 → c2 → …
      prevId = node.id;
      return node;
    };
    try {
      await api.agentStream({ message, selection, learn: true, max_steps: 4 }, {
        onStep: (s) => { steps.push(s); setLiveSteps([...steps]); setAgentStep(s); },
        onCard: (card) => {
          if (opts.replaceId && !replaced) { replaced = true; updateNode(opts.replaceId, { card, enter: false }); prevId = opts.replaceId; }
          else { const node = drop(card); setAgentAt({ x: node.x + node.w / 2, y: node.y + (EST_H[card.type] || 180) / 2 }); }
        },
        onFinal: (ev) => {
          (ev.sources || []).forEach((u) => drop({ type: "source", url: u }));
          focusOn(added);
          setMessages((m) => [...m, { role: "ai", text: ev.answer, steps, learned: ev.learned }]);
        },
        onError: () => setMessages((m) => [...m, { role: "ai", text: "The atelier hit an error mid-thought." }]),
      });
    } catch {
      setMessages((m) => [...m, { role: "ai", text: "The atelier is unreachable — is the backend running on :8000?" }]);
    } finally { setSending(false); setLiveSteps([]); setAgentStep(""); }
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
      clusterFrom(id, [res.card]);
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
      clusterFrom(id, [res.card]);
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
        clusterFrom(id, [res.card]);
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

  const saveCurrent = async () => {
    const name = prompt("Name this board", "Board " + new Date().toLocaleDateString());
    if (!name) return;
    try { await api.saveBoard(name, { nodes, edges, mode }); refreshBoards(); }
    catch { alert("Couldn't save — is the backend running?"); }
  };
  const loadBoard = async (id) => {
    if (!id) return;
    try {
      const d = await api.getBoard(id); const st = d.state || {};
      setNodes(st.nodes || []); setEdges(st.edges || []); if (st.mode) setMode(st.mode);
      seeded.current = (st.nodes || []).some((n) => n.kind !== "starter");
      const maxY = Math.max(START_Y, ...(st.nodes || []).map((n) => n.y + estH(n)));
      cols.current = [maxY + GAP, maxY + GAP, maxY + GAP];
      setSelectedId(null);
    } catch {}
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
          <div className="boards">
            <button className="board-btn" onClick={saveCurrent}>Save</button>
            <select className="board-sel" value="" onChange={(e) => { loadBoard(e.target.value); e.target.value = ""; }}>
              <option value="">Load…</option>
              {boards.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </div>
          <div className="status"><span className="dot" data-ok={ok} /><span className="label">{ok ? "Atelier live" : "Offline"}</span></div>
        </header>

        <Canvas
          nodes={nodes} edges={edges} focus={focus}
          sending={sending} agentAt={agentAt} agentStep={agentStep}
          tool={tool} setTool={setTool}
          selectedId={selectedId} onSelect={setSelectedId} onDeselect={() => setSelectedId(null)}
          onStarter={(n) => runAgent(n.prompt, [])}
          onPlace={onPlace} onUpload={addImage} onDropImage={addImage}
          onUpdateNode={updateNode} onAction={onAction} onRefine={onRefine}
          onAddEdge={addEdge} onDeleteEdge={removeEdge} onClear={clearBoard}
        />
      </div>

      <ChatRail messages={messages} sending={sending} liveSteps={liveSteps} onSend={(t) => runAgent(t, [])} />
    </div>
  );
}
