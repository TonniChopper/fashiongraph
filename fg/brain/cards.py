"""Typed cards — the structured artifacts the agent places on the canvas.

A plain chat answer is a paragraph; a *card* is a structured object the front end
renders richly and the user can select, refine, and act on. Each capability emits
its own shape:

* ``style``     — outfit combinations with slotted pieces + a colour palette.
* ``brand_dna`` — a brand's signature: aesthetic, values, materials, palette, positioning.
* ``trend``     — a verdict + 0–100 score + evidence for/against (for "rate this trend").
* ``lineage``   — the *knowledge graph's own* relations for an entity (deterministic,
                  no LLM — this is the legible "where does this come from" card).
* ``look``      — a reviewed outfit photo (garments, designers, palette, score).

The LLM-backed cards ask for strict JSON and fall back to a text card if the model
misbehaves, so the canvas never breaks. Palettes come back as human colour names
*and* hex, so the front end can drop real swatches.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fg.llm.base import Message

logger: logging.Logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict | None:
    """Best-effort JSON object extraction from an LLM reply."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    blob = m.group(0)
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):  # strip trailing commas
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _grounding(context_builder: Any | None, query: str) -> str:
    if context_builder is None:
        return ""
    try:
        return context_builder.build(query, n_rag=4).knowledge_block()
    except Exception as exc:  # noqa: BLE001
        logger.warning("card grounding failed (%s).", exc)
        return ""


def _json_card(llm: Any, system: str, request: str, ctype: str,
               context_builder: Any | None, *, temperature: float = 0.5,
               max_tokens: int = 700) -> dict:
    """Runs a JSON-mode capability call; falls back to a text card on parse failure."""
    know = _grounding(context_builder, request)
    user = request + (f"\n\nGrounding (use if relevant):\n{know}" if know else "")
    raw = llm.chat([Message("system", system), Message("user", user)],
                   temperature=temperature, max_tokens=max_tokens)
    data = _extract_json(raw)
    if not data:
        return {"type": ctype, "text": (raw or "").strip(), "raw": True}
    data["type"] = ctype
    return data


# ---------------------------------------------------------------------------
# LLM-backed cards
# ---------------------------------------------------------------------------

_STYLE_SYS = (
    'You are an expert personal stylist. Return ONLY JSON, no prose:\n'
    '{"title": str, "outfits": [{"name": str, '
    '"pieces": [{"slot": str, "item": str}], '
    '"palette": [{"name": str, "hex": str}], "why": str}], "tip": str}\n'
    'Give 2–3 concrete outfits with real garments, colours (with hex), and a short why.'
)
_BRAND_SYS = (
    'You are a fashion brand strategist. Return ONLY JSON, no prose:\n'
    '{"name": str, "aesthetic": str, "values": [str], '
    '"signature_materials": [str], "silhouettes": [str], '
    '"palette": [{"name": str, "hex": str}], "reference_points": [str], '
    '"positioning": str, "tagline": str}'
)
_TREND_SYS = (
    'You are a fashion trend analyst. Return ONLY JSON, no prose:\n'
    '{"topic": str, "verdict": str, "score": int, "trajectory": str, '
    '"evidence_for": [str], "evidence_against": [str]}\n'
    '"score" is 0–100 plausibility/momentum. "verdict" is one crisp line. '
    'When asked to *rate* a trend, weigh evidence honestly on both sides.'
)


def build_style_card(llm: Any, request: str, context_builder: Any | None = None) -> dict:
    """Styling advice → an outfits card."""
    return _json_card(llm, _STYLE_SYS, request, "style", context_builder)


def build_brand_card(llm: Any, request: str, context_builder: Any | None = None) -> dict:
    """Brand brief → a Brand-DNA card."""
    return _json_card(llm, _BRAND_SYS, request, "brand_dna", context_builder, temperature=0.5)


def build_trend_card(llm: Any, request: str, context_builder: Any | None = None) -> dict:
    """Trend topic → an analysed/scored trend card."""
    return _json_card(llm, _TREND_SYS, request, "trend", context_builder, temperature=0.3)


# ---------------------------------------------------------------------------
# Deterministic card (no LLM) — the graph speaks for itself
# ---------------------------------------------------------------------------

def build_lineage_card(kg: Any, entity: str) -> dict | None:
    """Designer/house lineage straight from the knowledge graph (no LLM).

    Reliable and legible — it shows the *actual* relations the KG holds, which is
    exactly the "where does this come from" evidence the thesis wants on screen.
    Returns ``None`` if the entity isn't in the graph.
    """
    if kg is None:
        return None
    try:
        outgoing = kg.outgoing(entity)
    except Exception:  # noqa: BLE001
        return None
    if not outgoing:
        return None
    conns = [
        {"relation": f["relation"].replace("_", " "), "target": f["object"]}
        for f in outgoing
    ]
    # Group by relation for a tidy card.
    by_rel: dict[str, list[str]] = {}
    for c in conns:
        by_rel.setdefault(c["relation"], []).append(c["target"])
    summary = "; ".join(f"{rel}: {', '.join(t[:4])}" for rel, t in list(by_rel.items())[:5])
    return {
        "type": "lineage",
        "entity": entity,
        "connections": conns[:16],
        "by_relation": {r: t[:6] for r, t in by_rel.items()},
        "text": f"{entity} — {summary}",
    }


def build_look_card(review_text: str, garments: list[str], sources: list[str]) -> dict:
    """Packages a look review (from /analyze) as a card."""
    return {
        "type": "look",
        "garments": garments,
        "review": review_text,
        "sources": sources,
        "text": review_text[:200],
    }
