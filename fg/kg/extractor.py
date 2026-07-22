"""LLM-assisted triple extraction (the Farfetch extract→normalize→link idea).

Given a text chunk, the LLM fills a *fixed* schema (subject, relation, object)
using only the allowed relations. A constrained schema + JSON output makes a
small local model reliable enough for KG construction. The JSON parser is pure
and unit-tested; the LLM call is a thin wrapper around it.
"""

from __future__ import annotations

import json
import logging
import re

from fg.kg.schema import ENTITY_TYPES, RELATION_TYPES, Triple, canonical_relation
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)


def build_extraction_prompt(text: str, source: str = "") -> list[Message]:
    """Builds the extraction chat prompt for a text chunk.

    Args:
        text: Source text (a fashion article/chunk).
        source: Provenance label passed through for context.

    Returns:
        Chat messages instructing strict JSON-triple extraction.
    """
    relations = ", ".join(sorted(RELATION_TYPES))
    types = ", ".join(sorted(ENTITY_TYPES))
    system = (
        "You extract a fashion knowledge graph from text. Output ONLY a JSON "
        "array of triples. Each triple has keys: subject, subject_type, "
        "relation, object, object_type. "
        f"Allowed relations: [{relations}]. "
        f"Allowed entity types: [{types}]. "
        "Extract EVERY specific fact the text supports — founders, creative "
        "directors, cities/HQ, materials, silhouettes, eras, collaborations, "
        "influences, sub-brands, successions. Prefer precise relations "
        "(founded_by, based_in, creative_director, uses_material, "
        "influenced_by) over the vague 'known_for'; use 'known_for' only for a "
        "genuine signature. Aim for many diverse triples, not a few. Do not "
        "invent facts. Output [] if nothing fits. JSON only, no prose."
    )
    example = (
        'Example: text "Bottega Veneta, founded in Vicenza in 1966, is known '
        'for its woven leather intrecciato; Matthieu Blazy is creative '
        'director." →\n'
        '[{"subject":"Bottega Veneta","subject_type":"brand","relation":'
        '"based_in","object":"Vicenza","object_type":"city"},'
        '{"subject":"Bottega Veneta","subject_type":"brand","relation":'
        '"uses_material","object":"woven leather","object_type":"material"},'
        '{"subject":"Bottega Veneta","subject_type":"brand","relation":'
        '"creative_director","object":"Matthieu Blazy","object_type":"designer"}]'
    )
    user = f"{example}\n\nSource: {source}\n\nText:\n{text[:3500]}\n\nJSON triples:"
    return [Message("system", system), Message("user", user)]


def parse_triples(raw: str, source: str = "") -> list[Triple]:
    """Parses an LLM JSON response into valid :class:`Triple` objects.

    Robust to extra prose around the JSON and to missing fields. Invalid or
    out-of-schema triples are dropped.

    Args:
        raw: The model's raw text response.
        source: Provenance to stamp on each triple.

    Returns:
        Valid, in-schema triples.
    """
    if not raw:
        return []
    match = _JSON_BLOCK.search(raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    triples: list[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        relation = canonical_relation(str(item.get("relation", "")))
        if not relation:
            continue  # unmappable → drop
        t = Triple(
            subject=str(item.get("subject", "")).strip(),
            relation=relation,
            object=str(item.get("object", "")).strip(),
            subject_type=str(item.get("subject_type", "")).strip().lower(),
            object_type=str(item.get("object_type", "")).strip().lower(),
            source=source,
        )
        if t.is_valid():
            triples.append(t)
    return triples


def extract_triples(text: str, llm: LLM, source: str = "") -> list[Triple]:
    """Runs LLM extraction on *text* and returns parsed triples.

    Args:
        text: Source text.
        llm: LLM backend.
        source: Provenance label.

    Returns:
        Valid triples (empty on any failure — extraction is best-effort).
    """
    try:
        raw = llm.chat(build_extraction_prompt(text, source), temperature=0.0, max_tokens=800)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Extraction LLM call failed (%s).", exc)
        return []
    triples = parse_triples(raw, source=source)
    logger.info("Extracted %d triples from source '%s'.", len(triples), source)
    return triples
