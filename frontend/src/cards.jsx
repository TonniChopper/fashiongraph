// Typed-card renderers — each backend card type becomes an editorial tearsheet.
import React from "react";
import { mdToHtml } from "./format.js";

const Prose = ({ text }) => <div className="prose" dangerouslySetInnerHTML={{ __html: mdToHtml(text || "") }} />;

const KIND = {
  style: "Styling",
  brand_dna: "Brand DNA",
  trend: "Trend Report",
  lineage: "Lineage",
  look: "Look Review",
  answer: "Note",
  source: "Source",
};

function Head({ kind, right }) {
  return (
    <div className="head">
      <span className="kicker">
        <span className="tick" />
        <span className="label">{KIND[kind] || kind}</span>
      </span>
      {right}
    </div>
  );
}

function Palette({ palette }) {
  if (!palette?.length) return null;
  return (
    <div className="palette">
      {palette.map((c, i) => {
        const hex = typeof c === "string" ? null : c.hex;
        const name = typeof c === "string" ? c : c.name;
        return (
          <div className="swatch" key={i} style={{ background: hex || "#ccc" }} title={name}>
            <span>{name}</span>
          </div>
        );
      })}
    </div>
  );
}

function StyleCard({ card }) {
  return (
    <div className="card">
      <Head kind="style" />
      <div className="body">
        {card.title && <h3>{card.title}</h3>}
        {(card.outfits || []).map((o, i) => (
          <div className="outfit" key={i}>
            <div className="oname">{o.name || `Look ${i + 1}`}</div>
            <ul className="pieces">
              {(o.pieces || []).map((p, j) => (
                <li key={j}>
                  <span className="slot">{p.slot}</span>
                  <span className="item">{p.item}</span>
                </li>
              ))}
            </ul>
            <Palette palette={o.palette} />
            {o.why && <div className="why" style={{ marginTop: 8 }}>{o.why}</div>}
          </div>
        ))}
        {card.tip && (
          <>
            <div className="rule" />
            <p><span className="label" style={{ marginRight: 8 }}>Tip</span>{card.tip}</p>
          </>
        )}
        {card.raw && <p>{card.text}</p>}
      </div>
    </div>
  );
}

function TrendCard({ card }) {
  const score = Number.isFinite(card.score) ? card.score : null;
  return (
    <div className="card">
      <Head kind="trend" right={card.trajectory ? <span className="label">{card.trajectory}</span> : null} />
      <div className="body">
        {card.topic && <h3 style={{ marginBottom: 10 }}>{card.topic}</h3>}
        <div className="gauge">
          {score !== null && (
            <div className="score">{score}<small>/100</small></div>
          )}
          <div className="verdict">
            {card.verdict && <div className="v">“{card.verdict}”</div>}
          </div>
        </div>
        {score !== null && <div className="meter"><i style={{ width: `${Math.max(0, Math.min(100, score))}%` }} /></div>}
        <div className="evidence">
          <div className="for">
            <span className="label">For</span>
            <ul>{(card.evidence_for || []).map((e, i) => <li key={i}>{e}</li>)}</ul>
          </div>
          <div className="against">
            <span className="label">Against</span>
            <ul>{(card.evidence_against || []).map((e, i) => <li key={i}>{e}</li>)}</ul>
          </div>
        </div>
        {card.raw && <p>{card.text}</p>}
      </div>
    </div>
  );
}

function LineageCard({ card }) {
  const groups = card.by_relation || {};
  return (
    <div className="card">
      <Head kind="lineage" />
      <div className="body">
        <div className="chips" style={{ marginBottom: 12 }}>
          <span className="chip node-self">{card.entity}</span>
        </div>
        {Object.entries(groups).map(([rel, targets]) => (
          <div className="lineage-group" key={rel}>
            <span className="label lineage-rel">{rel}</span>
            <div className="chips">
              {targets.map((t, i) => <span className="chip" key={i}>{t}</span>)}
            </div>
          </div>
        ))}
        {!Object.keys(groups).length && card.text && <p>{card.text}</p>}
      </div>
    </div>
  );
}

function BrandCard({ card }) {
  const Field = ({ label, children, full }) =>
    children ? (
      <div className={full ? "full" : ""}>
        <dt className="label">{label}</dt>
        <dd>{children}</dd>
      </div>
    ) : null;
  const list = (a) => (Array.isArray(a) ? a.join(" · ") : a);
  return (
    <div className="card">
      <Head kind="brand_dna" />
      <div className="body">
        {card.name && <h3>{card.name}</h3>}
        {card.aesthetic && <p style={{ marginBottom: 10 }}>{card.aesthetic}</p>}
        <Palette palette={card.palette} />
        <dl className="dna-grid">
          <Field label="Values">{list(card.values)}</Field>
          <Field label="Silhouettes">{list(card.silhouettes)}</Field>
          <Field label="Materials">{list(card.signature_materials)}</Field>
          <Field label="References">{list(card.reference_points)}</Field>
          <Field label="Positioning" full>{card.positioning}</Field>
        </dl>
        {card.tagline && <div className="tagline">“{card.tagline}”</div>}
        {card.raw && <p>{card.text}</p>}
      </div>
    </div>
  );
}

function LookCard({ card }) {
  return (
    <div className="card">
      <Head kind="look" />
      <div className="body">
        {card.garments?.length > 0 && (
          <div className="chips" style={{ marginBottom: 12 }}>
            {card.garments.map((g, i) => <span className="chip" key={i}>{g}</span>)}
          </div>
        )}
        <Prose text={card.review} />
      </div>
    </div>
  );
}

function SourceCard({ card }) {
  let host = card.url;
  try { host = new URL(card.url).hostname.replace(/^www\./, ""); } catch {}
  return (
    <div className="card source-card">
      <Head kind="source" />
      <div className="body">
        <span className="label" style={{ display: "block", marginBottom: 6 }}>{host}</span>
        <a href={card.url} target="_blank" rel="noreferrer">{card.title || card.url}</a>
      </div>
    </div>
  );
}

function AnswerCard({ card }) {
  return (
    <div className="card answer-card">
      <Head kind="answer" />
      <div className="body"><Prose text={card.text} /></div>
    </div>
  );
}

export const CARD_WIDTH = {
  style: 340, brand_dna: 360, trend: 360, lineage: 320,
  look: 340, answer: 320, source: 260,
};

export function CardView({ card }) {
  switch (card.type) {
    case "style": return <StyleCard card={card} />;
    case "trend": return <TrendCard card={card} />;
    case "lineage": return <LineageCard card={card} />;
    case "brand_dna": return <BrandCard card={card} />;
    case "look": return <LookCard card={card} />;
    case "source": return <SourceCard card={card} />;
    default: return <AnswerCard card={card} />;
  }
}
