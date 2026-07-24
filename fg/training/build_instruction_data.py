"""Phase 5 Track A — build the fashion instruction dataset (the 'behaviour' half).

Fine-tuning the LLM has two ingredients: *domain text* (your books/PDFs → voice,
see ``build_corpus.py``) and *instruction data* (task behaviour → this module).
The instructions are manufactured from assets we already own, so no corpus hunting
and nothing copyrighted:

* **KG → Q/A** — template questions over the knowledge graph (creative director,
  known-for, materials, silhouettes, influences, collaborations, era) plus
  multi-hop "how are X and Y connected" from the graph reasoner. Deterministic and
  grounded — no LLM in the loop, so it can't hallucinate. This also demonstrates
  the KG is *generative*, not just queryable (a nice thesis point).
* **Runway captions → styling description** — the VLM captions
  (``data/processed/runway_captions.jsonl``) become "describe a look from …" tasks.
* **Style-instruct seed** — ``neuralwork/fashion-style-instruct`` (3.2k styling
  dialogues) reshaped to chat.

Output is MLX-LM / Ollama-ready **chat JSONL** (``{"messages": [...]}``), split into
``train.jsonl`` / ``valid.jsonl``. Facts still come from RAG/KG at inference — this
teaches the model to *speak and reason* like a stylist, not to memorise facts.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

from fg.config import settings
from fg.kg.reasoning import GraphReasoner, format_path
from fg.kg.store import KnowledgeGraph

logger: logging.Logger = logging.getLogger(__name__)

SYSTEM = "You are a knowledgeable, tasteful fashion stylist and fashion historian."

#: relation → (question template, answer template). {s}=subject, {o}=object phrase.
REL_TEMPLATES: dict[str, tuple[str, str]] = {
    "creative_director": ("Who is the creative director of {s}?",
                          "The creative director of {s} is {o}."),
    "founded_by": ("Who founded {s}?", "{s} was founded by {o}."),
    "based_in": ("Where is {s} based?", "{s} is based in {o}."),
    "known_for": ("What is {s} known for?", "{s} is known for {o}."),
    "uses_material": ("What materials is {s} known for using?",
                      "{s} frequently works with {o}."),
    "has_silhouette": ("What silhouettes are characteristic of {s}?",
                       "{s} is characterised by {o} silhouettes."),
    "influenced_by": ("What has influenced {s}?", "{s} has been influenced by {o}."),
    "collaborated_with": ("Who has {s} collaborated with?",
                          "{s} has collaborated with {o}."),
}
_SINGLE = {"creative_director", "founded_by", "based_in"}

#: Relations meaningful enough for a 'how are X and Y connected' answer — a shared
#: material/silhouette is trivially true and reads awkwardly, so exclude those.
_PATH_RELS: frozenset[str] = frozenset({
    "successor_of", "influenced_by", "collaborated_with",
    "creative_director", "founded_by", "part_of", "associated_with",
})
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_SEASON_RE = re.compile(r"\b(spring|fall|autumn|winter|resort|pre-fall|couture)\b", re.I)


def _clean_obj(text: str, drop_temporal: bool = False) -> str | None:
    """Keeps readable object phrases; drops noise/hedges/over-long strings.

    Args:
        text: Raw KG object.
        drop_temporal: If set, reject year/season strings (they are noise for
            factual relations like ``based_in`` — a city, not a date).
    """
    t = " ".join(text.replace("_", " ").split()).strip(" .,")
    if not t or len(t) > 60:
        return None
    low = t.lower()
    if low in {"not specified", "not specified in text", "likely", "n/a", "unknown",
               "high fashion brand"}:
        return None
    if drop_temporal and (_YEAR_RE.search(t) or _SEASON_RE.search(t)):
        return None
    return t


def _join(objs: list[str]) -> str:
    """Natural-language join: 'a', 'a and b', 'a, b, and c'."""
    objs = objs[:5]
    if len(objs) == 1:
        return objs[0]
    if len(objs) == 2:
        return f"{objs[0]} and {objs[1]}"
    return ", ".join(objs[:-1]) + f", and {objs[-1]}"


def _chat(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def kg_qa(kg: KnowledgeGraph, subjects: list[str]) -> list[dict]:
    """Template Q/A over a *curated* subject list (deterministic, grounded).

    Subjects are the known houses only — never materials/silhouettes that would
    otherwise surface as subjects (e.g. "What materials is *satin* known for…").
    Year/season objects are dropped so factual answers stay clean.
    """
    out: list[dict] = []
    for disp in subjects:
        by_rel: dict[str, list[str]] = {}
        for f in kg.outgoing(disp):
            rel = f["relation"]
            if rel not in REL_TEMPLATES:
                continue
            o = _clean_obj(f["object"], drop_temporal=True)
            if o and o.lower() != disp.lower():
                by_rel.setdefault(rel, [])
                if o not in by_rel[rel]:
                    by_rel[rel].append(o)
        for rel, objs in by_rel.items():
            if not objs:
                continue
            q_t, a_t = REL_TEMPLATES[rel]
            phrase = _join(objs[:2] if rel in _SINGLE else objs)
            out.append(_chat(q_t.format(s=disp), a_t.format(s=disp, o=phrase)))
    return out


def kg_paths(kg: KnowledgeGraph, houses: list[str], max_pairs: int = 60,
             seed: int = 42) -> list[dict]:
    """Multi-hop 'how are X and Y connected' Q/A from the graph reasoner."""
    r = GraphReasoner(kg)
    rng = random.Random(seed)
    pairs = [(a, b) for i, a in enumerate(houses) for b in houses[i + 1:]]
    rng.shuffle(pairs)
    out: list[dict] = []
    for a, b in pairs:
        if len(out) >= max_pairs:
            break
        # Keep only a path whose every edge is a *meaningful* relation (not a
        # trivial shared material/silhouette hop) and whose nodes aren't noise.
        good = next(
            (p for p in r.paths(a, b, max_hops=3)
             if p and all(e["relation"] in _PATH_RELS for e in p)
             and "not specified" not in format_path(p).lower()),
            None,
        )
        if good is None:
            continue
        out.append(_chat(
            f"How are {a} and {b} connected in the world of fashion?",
            f"They connect through the graph: {format_path(good)}.",
        ))
    return out


def caption_tasks(captions_path: Path, limit: int | None = None) -> list[dict]:
    """Runway captions → 'describe a look from …' styling tasks."""
    if not captions_path.exists():
        return []
    tmpl = [
        "Describe a runway look from {d}'s {s} collection.",
        "What might a {d} look from {s} feature?",
        "Give a short styling description of a {d} {s} runway look.",
    ]
    out: list[dict] = []
    for i, line in enumerate(captions_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cap = (rec.get("caption") or "").strip()
        d, s = rec.get("designer", ""), rec.get("show", "")
        if not cap or not d:
            continue
        out.append(_chat(tmpl[i % len(tmpl)].format(d=d, s=s or "recent"), cap))
        if limit and len(out) >= limit:
            break
    return out


def seed_examples(seed_dir: Path, limit: int | None = None) -> list[dict]:
    """neuralwork/fashion-style-instruct parquet → chat examples."""
    files = sorted(seed_dir.rglob("*.parquet"))
    if not files:
        return []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas/pyarrow missing — skipping style-instruct seed.")
        return []
    out: list[dict] = []
    for fp in files:
        df = pd.read_parquet(fp)
        for row in df.itertuples(index=False):
            r = row._asdict()
            inp, comp = str(r.get("input", "")).strip(), str(r.get("completion", "")).strip()
            ctx = str(r.get("context", "")).strip()
            if not inp or not comp:
                continue
            user = f"{inp}\n\nOccasion/context: {ctx}" if ctx and ctx.lower() != "nan" else inp
            out.append(_chat(user, comp))
            if limit and len(out) >= limit:
                return out
    return out


def build(
    out_dir: str | Path | None = None,
    kg_path: str | Path | None = None,
    captions_path: str | Path | None = None,
    seed_dir: str | Path | None = None,
    val_frac: float = 0.05,
    caption_limit: int | None = 600,
    seed: int = 42,
) -> dict:
    """Assembles all sources → train.jsonl / valid.jsonl (chat format).

    Returns a stats dict (also logged).
    """
    kg = KnowledgeGraph(str(kg_path) if kg_path else None)
    caps = Path(captions_path) if captions_path else settings.data_dir / "processed" / "runway_captions.jsonl"
    seed_d = Path(seed_dir) if seed_dir else settings.data_dir / "raw" / "fashion-style-instruct"
    houses = ["Celine", "Dior", "Prada", "Gucci", "Balenciaga", "Bottega Veneta",
              "Loewe", "Marni", "Jacquemus", "Rick Owens", "Alexander McQueen",
              "Acne Studios"]

    parts = {
        "kg_qa": kg_qa(kg, houses),
        "kg_paths": kg_paths(kg, houses),
        "captions": caption_tasks(caps, limit=caption_limit),
        "seed": seed_examples(seed_d),
    }
    # Dedup on the (user) turn.
    seen: set[str] = set()
    examples: list[dict] = []
    for src, items in parts.items():
        for ex in items:
            u = ex["messages"][1]["content"]
            if u in seen:
                continue
            seen.add(u)
            examples.append(ex)

    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    valid, train = examples[:n_val], examples[n_val:]

    out = Path(out_dir) if out_dir else settings.data_dir / "processed" / "sft"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "by_source": {k: len(v) for k, v in parts.items()},
        "total_unique": len(examples),
        "train": len(train), "valid": len(valid),
        "out_dir": str(out),
    }
    logger.info("Instruction dataset: %s", stats)
    return stats


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build the fashion instruction dataset.")
    p.add_argument("--out", default=None)
    p.add_argument("--kg", default=None)
    p.add_argument("--captions", default=None)
    p.add_argument("--seed-dir", default=None)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--caption-limit", type=int, default=600)
    args = p.parse_args()
    stats = build(out_dir=args.out, kg_path=args.kg, captions_path=args.captions,
                  seed_dir=args.seed_dir, val_frac=args.val_frac,
                  caption_limit=args.caption_limit)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
