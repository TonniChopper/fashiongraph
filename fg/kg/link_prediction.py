"""LLM link prediction via one-shot in-context learning.

Predicts *plausible missing* edges for a KG entity — the completion side of
multi-hop reasoning. Given an entity's known neighbourhood, a one-shot ICL
prompt shows the model one worked example, then asks for likely-true triples the
graph is missing (constrained to the schema relations). Cheap, local, and no
training — the pragmatic alternative to embedding models (TransE/RotatE).

Predictions are returned tagged ``source="llm_predicted"`` and are NOT added
automatically — they are hypotheses to review or gate, not ground truth.
"""

from __future__ import annotations

import logging

from fg.kg.extractor import parse_triples
from fg.kg.schema import RELATION_TYPES, Triple, canonical_entity
from fg.llm.base import LLM, Message

logger: logging.Logger = logging.getLogger(__name__)

PREDICTED_SOURCE = "llm_predicted"

#: One worked example (the ICL shot) — teaches the format + the "missing, not
#: repeated" behaviour.
_ONE_SHOT = (
    "Example.\n"
    "Entity: Helmut Lang\n"
    "Known facts:\n"
    "- Helmut Lang known for minimalism\n"
    "- Helmut Lang based in Vienna\n"
    "Plausible MISSING facts (JSON):\n"
    '[{"subject":"Helmut Lang","subject_type":"designer","relation":"from_era",'
    '"object":"1990s","object_type":"era"},'
    '{"subject":"Helmut Lang","subject_type":"designer","relation":"associated_with",'
    '"object":"deconstruction","object_type":"aesthetic"}]'
)


def build_prediction_messages(entity: str, known_facts: list[str], k: int = 5) -> list[Message]:
    """Builds the one-shot ICL prompt for link prediction.

    Args:
        entity: The target entity (predictions have it as subject).
        known_facts: Its existing facts (so the model doesn't repeat them).
        k: How many predictions to request.

    Returns:
        Chat messages.
    """
    relations = ", ".join(sorted(RELATION_TYPES))
    system = (
        "You predict PLAUSIBLE, likely-true facts that are MISSING from a fashion "
        f"knowledge graph. Use ONLY these relations: [{relations}]. Ground each "
        "prediction in the known facts plus well-established fashion knowledge. Do "
        "NOT repeat known facts, and do not invent obscure specifics you are unsure "
        "of. The subject of every predicted triple must be the target entity. "
        "Output ONLY a JSON array of triples (subject, subject_type, relation, "
        "object, object_type)."
    )
    known = "\n".join(f"- {f}" for f in known_facts) or "- (none)"
    user = (
        f"{_ONE_SHOT}\n\n"
        f"Now do the same.\nEntity: {entity}\nKnown facts:\n{known}\n\n"
        f"Give up to {k} plausible MISSING facts as a JSON array:"
    )
    return [Message("system", system), Message("user", user)]


def predict_links(entity: str, kg, llm: LLM, k: int = 5) -> list[Triple]:
    """Predicts up to *k* plausible missing triples for *entity*.

    Args:
        entity: The target entity.
        kg: A ``KnowledgeGraph`` (for the entity's known facts + dedup).
        llm: LLM backend.
        k: Max predictions.

    Returns:
        Predicted, schema-valid, novel triples (tagged ``llm_predicted``).
    """
    known = kg.facts_as_text(entity, limit=30)
    messages = build_prediction_messages(entity, known, k)
    try:
        raw = llm.chat(messages, temperature=0.3, max_tokens=500)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Link-prediction call failed (%s).", exc)
        return []

    candidates = parse_triples(raw, source=PREDICTED_SOURCE)
    ent_key = canonical_entity(entity)
    existing = {(f["relation"], canonical_entity(f["object"])) for f in kg.outgoing(entity)}

    out: list[Triple] = []
    seen: set[tuple[str, str]] = set()
    for t in candidates:
        if t.subject_key != ent_key:            # keep predictions about the entity
            continue
        sig = (t.relation, t.object_key)
        if sig in existing or sig in seen:       # skip already-known / dup
            continue
        seen.add(sig)
        out.append(t)
        if len(out) >= k:
            break
    logger.info("Predicted %d new links for '%s'.", len(out), entity)
    return out
