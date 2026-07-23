"""Track B supervision — KG-concept alignment pairs for the embedder.

The Phase-5 council chose Track B: fine-tune the fashion embedder so an outfit
photo lands nearer the *concepts* its designer is known for. The defensible
novelty is **the knowledge graph supplying the supervision signal** — not the
bare designer label (that would be FashionKLIP re-run and a "house-signature
detector"). So this module joins the built runway ``VisualIndex`` (image
embeddings + designer/show metadata) to the KG's per-designer concepts, and
exposes exactly the pieces the training + ablation need:

* :func:`load_supervision` — one record per runway image: its embedding row,
  designer (the *label-only* ablation target), collection group (for an honest
  by-collection split), and the KG **concept set** for that designer.
* :func:`build_concept_vocab` — the concept vocabulary, keeping concepts shared
  by ``>= min_designers`` houses. Concepts that recur across designers are what
  make this concept alignment rather than a relabelled designer id — the direct
  answer to the "is the KG ornamental?" risk the council flagged.
* :func:`by_collection_split` — holds out whole shows (mirrors
  ``runway_eval.split_by='collection'``) so no test look shares a collection
  with train.
* :func:`signal_report` — the "one thing to do first": counts and cross-designer
  concept overlap, so we know there is signal *before* spending a GPU-hour.

Pure ``numpy`` + stdlib (no torch) so it is cheap to import and unit-testable.
The training loop and its ablation consume these records; they live separately.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from fg.kg.schema import canonical_entity

logger: logging.Logger = logging.getLogger(__name__)

#: KG relations whose objects are look-relevant *concepts* (visually grounded or
#: aesthetic), as opposed to biographical facts (based_in, founded_by, …).
CONCEPT_RELATIONS: frozenset[str] = frozenset({
    "uses_material", "has_silhouette", "known_for", "from_era",
    "associated_with", "influenced_by",
})

_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"^(19|20)\d{2}s?$")


def clean_concept(text: str, max_words: int = 3) -> str | None:
    """Normalises a KG object into a compact concept token, or drops it.

    Keeps short, reusable descriptors (materials, silhouettes, one-to-three-word
    aesthetics) and rejects the long idiosyncratic ``known_for`` phrases
    ("British Designer of the Year award") that could only ever belong to one
    designer — those would leak the label back in and inflate the ablation.

    Args:
        text: Raw KG object string.
        max_words: Longest phrase (in words) to keep.

    Returns:
        A lowercased concept token, or ``None`` if it should be dropped.
    """
    t = text.strip().lower().replace("_", " ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or _YEAR_RE.match(t):
        return None
    words = t.split()
    if not (1 <= len(words) <= max_words):
        return None
    # Drop hedge/noise tokens that carry no visual meaning.
    if t in {"not specified", "not specified in text", "likely", "high fashion brand"}:
        return None
    return t


@dataclass
class LookRecord:
    """One runway image as a supervision unit.

    Attributes:
        row: Index into the ``VisualIndex.embeddings`` matrix.
        designer: Canonical designer key (label-only ablation target).
        group: ``designer|show`` — the by-collection split unit.
        concepts: Canonical KG concepts for this designer (may be empty).
    """

    row: int
    designer: str
    group: str
    concepts: list[str] = field(default_factory=list)


def designer_concepts(kg, designer_key: str) -> list[str]:
    """Returns cleaned concept tokens for a canonical *designer_key* from the KG.

    Args:
        kg: A :class:`fg.kg.store.KnowledgeGraph`.
        designer_key: Canonical designer/brand key.

    Returns:
        Sorted unique concept tokens (possibly empty).
    """
    out: set[str] = set()
    for f in kg.outgoing(designer_key):
        if f["relation"] not in CONCEPT_RELATIONS:
            continue
        c = clean_concept(f["object"])
        if c:
            out.add(c)
    return sorted(out)


def load_supervision(index, kg) -> list[LookRecord]:
    """Builds one :class:`LookRecord` per image in the runway *index*.

    Joins each image's designer to the KG's concepts once per designer (cached),
    so this is cheap even though the index has thousands of rows.

    Args:
        index: A runway ``VisualIndex`` (metadata needs ``designer``/``show``).
        kg: A :class:`fg.kg.store.KnowledgeGraph`.

    Returns:
        A list of records aligned to ``index.embeddings`` rows.
    """
    concept_cache: dict[str, list[str]] = {}
    records: list[LookRecord] = []
    for row, m in enumerate(index.metadata):
        dk = canonical_entity(m.get("designer", "") or "unknown")
        if dk not in concept_cache:
            concept_cache[dk] = designer_concepts(kg, dk)
        show = m.get("show") or m.get("season") or ""
        records.append(LookRecord(
            row=row, designer=dk, group=f"{dk}|{show}",
            concepts=concept_cache[dk],
        ))
    return records


def build_concept_vocab(records: list[LookRecord], min_designers: int = 2) -> list[str]:
    """Concepts shared by at least *min_designers* distinct designers.

    Restricting to shared concepts is deliberate: a concept that belongs to only
    one house is indistinguishable from that house's label, so it cannot show
    that the *graph* (not the label) drove any lift. The surviving vocabulary is
    the cross-designer structure the alignment actually learns from.

    Args:
        records: Output of :func:`load_supervision`.
        min_designers: Minimum distinct designers a concept must appear for.

    Returns:
        Sorted concept vocabulary.
    """
    by_concept: dict[str, set[str]] = defaultdict(set)
    for r in records:
        for c in r.concepts:
            by_concept[c].add(r.designer)
    return sorted(c for c, ds in by_concept.items() if len(ds) >= min_designers)


def by_collection_split(
    records: list[LookRecord], holdout_frac: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Splits row indices by whole collection (no show spans train and test).

    Mirrors ``runway_eval.evaluate_designer_topk(split_by="collection")`` so the
    training split and the honest eval split use the same definition of leakage.

    Args:
        records: Output of :func:`load_supervision`.
        holdout_frac: Fraction of *collections* to hold out.
        seed: RNG seed.

    Returns:
        ``(train_rows, test_rows)`` as int arrays.
    """
    groups = np.array([r.group for r in records])
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_hold = max(1, int(len(uniq) * holdout_frac))
    held = set(uniq[:n_hold].tolist())
    mask = np.array([g in held for g in groups])
    rows = np.array([r.row for r in records])
    return rows[~mask], rows[mask]


def signal_report(records: list[LookRecord], min_designers: int = 2) -> dict:
    """Quantifies whether the KG supplies usable cross-designer concept signal.

    The council's "one thing to do first": before any training, confirm the KG
    concepts (a) exist per designer and (b) overlap across designers. Mean
    pairwise Jaccard over designer concept-sets is the headline — near 0 means
    concepts are just labels in disguise (Track B would collapse to FashionKLIP);
    clearly positive means there is shared structure to align to.

    Args:
        records: Output of :func:`load_supervision`.
        min_designers: Threshold passed to :func:`build_concept_vocab`.

    Returns:
        A metrics dict (also logged).
    """
    designers = sorted({r.designer for r in records})
    concepts_by_designer = {
        d: set(next(r.concepts for r in records if r.designer == d))
        for d in designers
    }
    vocab = build_concept_vocab(records, min_designers=min_designers)
    all_concepts = {c for s in concepts_by_designer.values() for c in s}

    # Mean pairwise Jaccard overlap between designers' concept sets.
    jac: list[float] = []
    for i in range(len(designers)):
        for j in range(i + 1, len(designers)):
            a, b = concepts_by_designer[designers[i]], concepts_by_designer[designers[j]]
            if a or b:
                jac.append(len(a & b) / len(a | b))
    mean_jaccard = round(float(np.mean(jac)), 3) if jac else 0.0

    groups = {r.group for r in records}
    report = {
        "n_images": len(records),
        "n_designers": len(designers),
        "n_collections": len(groups),
        "concepts_per_designer": {
            d: len(concepts_by_designer[d]) for d in designers
        },
        "total_unique_concepts": len(all_concepts),
        "shared_concepts": len(vocab),
        "shared_fraction": round(len(vocab) / max(1, len(all_concepts)), 3),
        "mean_pairwise_jaccard": mean_jaccard,
        "images_missing_concepts": sum(1 for r in records if not r.concepts),
    }
    logger.info("Track B signal report: %s", report)
    return report
