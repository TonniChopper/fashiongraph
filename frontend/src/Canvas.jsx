import React, { useCallback, useEffect, useRef, useState } from "react";
import { CardView, CARD_WIDTH } from "./cards.jsx";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const VIEW = "fg_view_v2";
const loadView = () => { try { return JSON.parse(localStorage.getItem(VIEW)) || null; } catch { return null; } };

const NH = { style: 340, brand_dna: 320, trend: 280, lineage: 240, look: 260, answer: 150, source: 96, text: 150, sketch: 400, starter: 150 };
const nodeH = (n) => n.h || NH[n.type] || NH[n.kind] || 180;
const centerOf = (n) => [n.x + (n.w || CARD_WIDTH[n.type] || 320) / 2, n.y + nodeH(n) / 2];

export default function Canvas({
  nodes, edges = [], focus, tool, setTool, selectedId, onSelect, onDeselect,
  onStarter, onPlace, onDropImage, onUpload, onUpdateNode, onAction, onRefine, onAddEdge, onDeleteEdge, onClear,
}) {
  const ref = useRef(null);
  const fileRef = useRef(null);
  const [t, setT] = useState(loadView() || { x: 120, y: 150, scale: 1 });
  const pan = useRef(null);
  const drag = useRef(null);
  const [grabbing, setGrabbing] = useState(false);
  const [refineFor, setRefineFor] = useState(null);
  const [refineText, setRefineText] = useState("");
  const [linkStart, setLinkStart] = useState(null);

  // Bring freshly-placed cards into view — no more hunting for the answer.
  useEffect(() => {
    if (!focus?.nonce || !focus.box || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setT((p) => ({ scale: p.scale,
      x: rect.width / 2 - (focus.box.x + focus.box.w / 2) * p.scale,
      y: rect.height * 0.32 - focus.box.y * p.scale }));
  }, [focus?.nonce]);

  useEffect(() => { if (tool !== "link") setLinkStart(null); }, [tool]);

  useEffect(() => {
    const id = setTimeout(() => { try { localStorage.setItem(VIEW, JSON.stringify(t)); } catch {} }, 200);
    return () => clearTimeout(id);
  }, [t]);

  const onWheel = useCallback((e) => {
    e.preventDefault();
    const rect = ref.current.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    setT((prev) => {
      const scale = clamp(prev.scale * Math.exp(-e.deltaY * 0.0016), 0.32, 2.6);
      const k = scale / prev.scale;
      return { scale, x: px - (px - prev.x) * k, y: py - (py - prev.y) * k };
    });
  }, []);
  useEffect(() => {
    const el = ref.current;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const toWorld = (clientX, clientY) => {
    const rect = ref.current.getBoundingClientRect();
    return { x: (clientX - rect.left - t.x) / t.scale, y: (clientY - rect.top - t.y) / t.scale };
  };

  const onPointerDown = (e) => {
    if (e.target.closest(".node") || e.target.closest(".actionbar") || e.target.closest(".zoombar") || e.target.closest(".toolbar") || e.target.closest(".edge-g")) return;
    pan.current = { sx: e.clientX, sy: e.clientY, ox: t.x, oy: t.y, moved: 0 };
    if (tool === "select") setGrabbing(true);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e) => {
    const p0 = pan.current;
    if (!p0) return;
    const dx = e.clientX - p0.sx, dy = e.clientY - p0.sy;
    p0.moved += Math.abs(dx) + Math.abs(dy);
    if (tool === "select") setT((p) => ({ ...p, x: p0.ox + dx, y: p0.oy + dy }));
  };
  const onPointerUp = (e) => {
    if (pan.current && pan.current.moved < 5) {
      if (tool === "link") setLinkStart(null);
      else if (tool !== "select") { const w = toWorld(e.clientX, e.clientY); onPlace(tool, w.x - 20, w.y - 20); }
      else { onDeselect(); setRefineFor(null); }
    }
    pan.current = null; setGrabbing(false);
  };

  const zoom = (dir) => setT((p) => {
    const scale = clamp(p.scale * (dir > 0 ? 1.2 : 1 / 1.2), 0.32, 2.6);
    const rect = ref.current.getBoundingClientRect();
    const cx = rect.width / 2, cy = rect.height / 2, k = scale / p.scale;
    return { scale, x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k };
  });

  const sel = nodes.find((n) => n.id === selectedId);
  const barPos = sel && (() => {
    const w = sel.w || CARD_WIDTH[sel.type] || 320;
    return { left: sel.x * t.scale + t.x + (w * t.scale) / 2, top: sel.y * t.scale + t.y - 14 };
  })();

  const submitRefine = (e) => {
    e.preventDefault();
    if (refineText.trim()) { onRefine(selectedId, refineText.trim()); setRefineText(""); setRefineFor(null); }
  };

  const onDrop = (e) => {
    e.preventDefault();
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) onDropImage(f);
  };

  return (
    <div
      ref={ref}
      className={"canvas" + (grabbing ? " grabbing" : "") + (tool !== "select" ? " placing" : "")}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
      onDragOver={(e) => e.preventDefault()} onDrop={onDrop}
    >
      <svg className="edges" width="100%" height="100%">
        {edges.map((e) => {
          const a = nodes.find((n) => n.id === e.a), b = nodes.find((n) => n.id === e.b);
          if (!a || !b) return null;
          const [ax, ay] = centerOf(a), [bx, by] = centerOf(b);
          const x1 = ax * t.scale + t.x, y1 = ay * t.scale + t.y;
          const x2 = bx * t.scale + t.x, y2 = by * t.scale + t.y;
          const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - 26;
          const d = `M${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
          const px = 0.25 * x1 + 0.5 * mx + 0.25 * x2;
          const py = 0.25 * y1 + 0.5 * my + 0.25 * y2;
          return (
            <g key={e.id} className="edge-g"
              onPointerDown={(ev) => ev.stopPropagation()}
              onClick={() => onDeleteEdge && onDeleteEdge(e.id)}>
              <path d={d} className="edge-hit" />
              <path d={d} className="edge" />
              <circle cx={x1} cy={y1} r={4} className="edge-dot" />
              <circle cx={x2} cy={y2} r={4} className="edge-dot" />
              <g className="edge-del" transform={`translate(${px} ${py})`}>
                <circle r="9" className="edge-del-bg" />
                <path d="M-3.2 -3.2 L3.2 3.2 M3.2 -3.2 L-3.2 3.2" className="edge-del-x" />
              </g>
            </g>
          );
        })}
      </svg>

      <div className="world" style={{ transform: `translate(${t.x}px, ${t.y}px) scale(${t.scale})` }}>
        <div className="grid" />
        {nodes.map((n) => (
          <div
            key={n.id}
            className={"node " + (n.kind || "card") + (n.enter ? " enter" : "")}
            data-selected={n.id === selectedId}
            style={{ left: n.x, top: n.y, width: n.w || CARD_WIDTH[n.type] || 320 }}
            data-linkstart={n.id === linkStart}
            onPointerDown={(e) => {
              e.stopPropagation();
              drag.current = { id: n.id, sx: e.clientX, sy: e.clientY, ox: n.x, oy: n.y, moved: 0 };
              e.currentTarget.setPointerCapture?.(e.pointerId);
            }}
            onPointerMove={(e) => {
              const d = drag.current;
              if (!d || d.id !== n.id) return;
              d.moved += Math.abs(e.clientX - d.sx) + Math.abs(e.clientY - d.sy);
              if (tool === "select") onUpdateNode(n.id, { x: d.ox + (e.clientX - d.sx) / t.scale, y: d.oy + (e.clientY - d.sy) / t.scale, enter: false });
            }}
            onPointerUp={() => {
              const d = drag.current; drag.current = null;
              if (!d || d.moved >= 5) return;
              if (n.kind === "starter") onStarter(n);
              else if (tool === "link") setLinkStart((s) => { if (!s) return n.id; if (s !== n.id) onAddEdge(s, n.id); return n.id; });
              else onSelect(n.id);
            }}
          >
            <NodeBody node={n} onUpdateNode={onUpdateNode} scaleRef={ref} />
          </div>
        ))}
      </div>

      {sel && barPos && sel.kind !== "starter" && (
        <div className="actionbar" style={{ left: barPos.left, top: barPos.top }}>
          {refineFor === sel.id ? (
            <form className="refine-in" onSubmit={submitRefine}>
              <input autoFocus placeholder="make it less formal, add colour…" value={refineText}
                onChange={(e) => setRefineText(e.target.value)} onPointerDown={(e) => e.stopPropagation()} />
              <button type="submit">Apply</button>
            </form>
          ) : (
            <ActionButtons node={sel} grouped={edges.some((e) => e.a === sel.id || e.b === sel.id)}
              onAction={onAction} openRefine={() => { setRefineFor(sel.id); setRefineText(""); }} />
          )}
        </div>
      )}

      <div className="toolbar">
        <ToolBtn t="select" tool={tool} setTool={setTool} title="Move" glyph={<ArrowCursor />} />
        <ToolBtn t="text" tool={tool} setTool={setTool} title="Add text" glyph={<span style={{ fontFamily: "Bodoni Moda, serif", fontSize: 17 }}>T</span>} />
        <ToolBtn t="sketch" tool={tool} setTool={setTool} title="Sketch a silhouette" glyph={<Pencil />} />
        <ToolBtn t="link" tool={tool} setTool={setTool} title="Connect objects into a group" glyph={<LinkIcon />} />
        <button className="tool" title="Upload image" onClick={() => fileRef.current?.click()}><ImageIcon /></button>
        <input ref={fileRef} type="file" accept="image/*" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ""; }} />
        <span className="tool-sep" />
        <button className="tool danger" title="Clear board" onClick={onClear}><Trash /></button>
      </div>

      <div className="zoombar">
        <button onClick={() => zoom(-1)} aria-label="Zoom out">–</button>
        <span className="pct">{Math.round(t.scale * 100)}%</span>
        <button onClick={() => zoom(1)} aria-label="Zoom in">+</button>
      </div>
    </div>
  );
}

function NodeBody({ node, onUpdateNode }) {
  if (node.kind === "starter") return <StarterCard node={node} />;
  if (node.kind === "image") return <ImageNode node={node} />;
  if (node.kind === "text") return <TextNode node={node} onUpdateNode={onUpdateNode} />;
  if (node.kind === "sketch") return <SketchNode node={node} onUpdateNode={onUpdateNode} />;
  return <CardView card={node.card} />;
}

function ActionButtons({ node, grouped, onAction, openRefine }) {
  const A = (label, action, danger) => (
    <button className={danger ? "danger" : ""} onClick={() => onAction(node.id, action)}>{label}</button>
  );
  const kind = node.kind || "card";
  return (
    <>
      {grouped && A("Analyze group", "group")}
      {kind === "card" && <button onClick={openRefine}>Refine</button>}
      {kind === "card" && A("Explore", "explore")}
      {kind === "image" && A("Review look", "review")}
      {kind === "text" && A("Style around", "style")}
      {kind === "text" && A("Explore", "explore")}
      {kind === "sketch" && A("Analyze", "analyze")}
      {kind === "sketch" && A("Clear", "clear")}
      <span className="sep" />
      {A("Dismiss", "dismiss", true)}
    </>
  );
}

/* ---------- node bodies ---------- */
function StarterCard({ node }) {
  return (
    <div className="card">
      <div className="head"><span className="kicker"><span className="tick" /><span className="label">{node.eyebrow}</span></span></div>
      <div className="body"><div className="prompt">{node.prompt}</div><div className="go">Begin <span aria-hidden>→</span></div></div>
    </div>
  );
}

function ImageNode({ node }) {
  return (
    <div className="card image-card">
      <div className="head"><span className="kicker"><span className="tick" /><span className="label">Your look</span></span></div>
      <img src={node.src} alt="dropped look" draggable={false} style={{ width: "100%", display: "block" }} />
    </div>
  );
}

function TextNode({ node, onUpdateNode }) {
  const ref = useRef(null);
  useEffect(() => { if (node.editing && ref.current) ref.current.focus(); }, [node.editing]);
  const grow = (e) => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; };
  return (
    <div className="card text-card">
      <div className="head"><span className="kicker"><span className="tick" /><span className="label">Note</span></span></div>
      <div className="body">
        <textarea
          ref={ref} rows={2} value={node.text} placeholder="Type a note, a brief, an idea…"
          onChange={(e) => { onUpdateNode(node.id, { text: e.target.value }); grow(e); }}
          onBlur={() => onUpdateNode(node.id, { editing: false })}
          onPointerDown={(e) => e.stopPropagation()}
        />
      </div>
    </div>
  );
}

function SketchNode({ node, onUpdateNode }) {
  const svgRef = useRef(null);
  const [draft, setDraft] = useState(null);
  const W = node.w, H = node.h || 400;
  const pt = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  };
  const down = (e) => { e.stopPropagation(); svgRef.current.setPointerCapture?.(e.pointerId); setDraft([pt(e)]); };
  const move = (e) => { if (draft) { e.stopPropagation(); setDraft((d) => [...d, pt(e)]); } };
  const up = () => { if (draft && draft.length > 1) onUpdateNode(node.id, { strokes: [...(node.strokes || []), draft] }); setDraft(null); };
  const toPath = (s) => s.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const empty = !(node.strokes || []).length && !draft;
  return (
    <div className="card sketch-card">
      <div className="head"><span className="kicker"><span className="tick" /><span className="label">Silhouette</span></span></div>
      <svg ref={svgRef} className="sketch" viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}
        onPointerDown={down} onPointerMove={move} onPointerUp={up}>
        {(node.strokes || []).map((s, i) => <path key={i} d={toPath(s)} className="ink" />)}
        {draft && <path d={toPath(draft)} className="ink" />}
        {empty && <text x={W / 2} y={H / 2} className="sketch-hint">draw a silhouette</text>}
      </svg>
    </div>
  );
}

/* ---------- toolbar bits ---------- */
function ToolBtn({ t, tool, setTool, title, glyph }) {
  return (
    <button className={"tool" + (tool === t ? " on" : "")} title={title} aria-label={title} onClick={() => setTool(t)}>{glyph}</button>
  );
}
const S = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
const ArrowCursor = () => <svg {...S}><path d="M5 3l6 15 2-6 6-2z" /></svg>;
const Pencil = () => <svg {...S}><path d="M4 20h4L20 8l-4-4L4 16z" /><path d="M14 6l4 4" /></svg>;
const ImageIcon = () => <svg {...S}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="8.5" cy="9.5" r="1.6" /><path d="M4 17l5-5 4 4 3-3 4 4" /></svg>;
const Trash = () => <svg {...S}><path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" /></svg>;
const LinkIcon = () => <svg {...S}><path d="M9 15l6-6" /><path d="M11 6l1-1a3.5 3.5 0 015 5l-1 1" /><path d="M13 18l-1 1a3.5 3.5 0 01-5-5l1-1" /></svg>;
