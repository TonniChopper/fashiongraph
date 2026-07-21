"""Preference-pair sources for the aesthetic trainer.

Each source yields a ``PairData``: a pool of ``id -> PIL image`` items and a list
of ``(winner_id, loser_id)`` preference pairs. The trainer pools sources (with
source-namespaced ids), embeds unique items once, and trains a Bradley-Terry
head. This makes it trivial to *ablate* which data helps — useful science, and
useful for the thesis.

Sources implemented:
- ``surrey``: worn-look human pairwise judgments (on-domain, small).
- ``ava``: general photo aesthetics; pairs sampled by mean-score gap
  (a general "compositional beauty" prior — expect transfer, not a fashion fix).

Heavy libs (pandas/PIL) are imported lazily inside ``load``.
"""

from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fg.config import settings

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PairData:
    """A pool of items and preference pairs over them.

    Attributes:
        items: ``id -> PIL image``.
        pairs: ``(winner_id, loser_id)`` tuples referencing ``items``.
    """

    items: dict[str, Any] = field(default_factory=dict)
    pairs: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without any heavy deps)
# ---------------------------------------------------------------------------

def parse_surrey_pair_lines(lines: list[str]) -> list[tuple[str, str, int]]:
    """Parses Surrey ``aesthetic_*.txt`` lines into ``(imgA, imgB, pref)``.

    Args:
        lines: Raw lines, each ``"a.jpg b.jpg 1|2"``.

    Returns:
        Valid ``(imgA, imgB, pref)`` triples (pref 1 = left, 2 = right).
    """
    out: list[tuple[str, str, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[2] in {"1", "2"}:
            out.append((parts[0], parts[1], int(parts[2])))
    return out


def sample_pairs_from_scores(
    scored: list[tuple[str, float]],
    *,
    margin: float = 1.0,
    max_pairs: int = 20000,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Samples preference pairs from scored items by score gap.

    Only keeps pairs whose scores differ by at least *margin* (so the
    preference is unambiguous), winner = higher score.

    Args:
        scored: ``(id, score)`` pairs.
        margin: Minimum score gap to accept a pair.
        max_pairs: Cap on generated pairs.
        seed: RNG seed.

    Returns:
        ``(winner_id, loser_id)`` pairs.
    """
    if len(scored) < 2:
        return []
    rng = random.Random(seed)
    ids = [i for i, _ in scored]
    score = {i: s for i, s in scored}
    pairs: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = max_pairs * 30
    while len(pairs) < max_pairs and attempts < max_attempts:
        a, b = rng.choice(ids), rng.choice(ids)
        attempts += 1
        if a == b:
            continue
        diff = score[a] - score[b]
        if abs(diff) >= margin:
            pairs.append((a, b) if diff > 0 else (b, a))
    return pairs


def _decode_image(cell: Any):
    """Decodes an HF ``Image`` parquet cell into a PIL image (or ``None``)."""
    from PIL import Image

    if isinstance(cell, dict) and cell.get("bytes"):
        return Image.open(io.BytesIO(cell["bytes"]))
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    return None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SurreySource:
    """Worn-look pairwise judgments (Surrey aesthetics)."""

    name = "surrey"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.data_dir / "raw" / "surrey-aesthetics")

    def load(self, limit_items: int | None = None, max_pairs: int | None = None) -> PairData:
        """Loads Surrey items + preference pairs.

        Raises:
            FileNotFoundError: If no aesthetic_*.txt files are found.
        """
        from PIL import Image

        triples: list[tuple[str, str, int]] = []
        for fp in sorted(self.root.rglob("aesthetic_*.txt")):
            triples += parse_surrey_pair_lines(fp.read_text(errors="ignore").splitlines())
        if not triples:
            raise FileNotFoundError(f"No aesthetic_*.txt under {self.root}.")

        index = {p.name: p for p in self.root.rglob("*.jpg")}
        names = {a for a, _, _ in triples} | {b for _, b, _ in triples}
        items: dict[str, Any] = {}
        for n in names:
            if n in index:
                try:
                    items[n] = Image.open(index[n]).convert("RGB")
                except Exception:  # noqa: BLE001
                    pass

        pairs = [
            (a, b) if pref == 1 else (b, a)
            for a, b, pref in triples
            if a in items and b in items
        ]
        if max_pairs:
            pairs = pairs[:max_pairs]
        logger.info("surrey: %d items, %d pairs", len(items), len(pairs))
        return PairData(items, pairs)


class AVASource:
    """General photo aesthetics (AVA subset), pairs by mean-score gap."""

    name = "ava"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.data_dir / "raw" / "ava-aesthetics")

    def load(self, limit_items: int | None = 8000, max_pairs: int | None = 20000) -> PairData:
        """Loads AVA items + score-sampled pairs.

        Args:
            limit_items: Cap images embedded (AVA is large; keeps builds sane).
            max_pairs: Cap generated pairs.

        Raises:
            FileNotFoundError: If no parquet files are found.
        """
        import pandas as pd

        files = sorted(self.root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet under {self.root}.")

        items: dict[str, Any] = {}
        scored: list[tuple[str, float]] = []
        count = 0
        for fp in files:
            df = pd.read_parquet(fp)
            if "mean_score" not in df.columns or "image" not in df.columns:
                continue
            for row in df.itertuples(index=False):
                r = row._asdict()
                img = _decode_image(r.get("image"))
                if img is None:
                    continue
                iid = str(r.get("image_id", f"ava_{count}"))
                items[iid] = img.convert("RGB")
                scored.append((iid, float(r["mean_score"])))
                count += 1
                if limit_items and count >= limit_items:
                    break
            if limit_items and count >= limit_items:
                break

        pairs = sample_pairs_from_scores(scored, max_pairs=max_pairs or 20000)
        logger.info("ava: %d items, %d pairs", len(items), len(pairs))
        return PairData(items, pairs)


#: Registry of available pair sources.
SOURCES: dict[str, type] = {"surrey": SurreySource, "ava": AVASource}


def load_sources(
    names: list[str], *, limit_items: int | None = None, max_pairs: int | None = None
) -> PairData:
    """Loads and pools multiple sources with source-namespaced ids.

    Args:
        names: Source names (keys of :data:`SOURCES`).
        limit_items: Per-source item cap.
        max_pairs: Per-source pair cap.

    Returns:
        A merged :class:`PairData`.

    Raises:
        KeyError: If a source name is unknown.
    """
    merged = PairData()
    for name in names:
        if name not in SOURCES:
            raise KeyError(f"Unknown source {name!r}. Known: {list(SOURCES)}")
        data = SOURCES[name]().load(limit_items=limit_items, max_pairs=max_pairs)
        for iid, img in data.items.items():
            merged.items[f"{name}:{iid}"] = img
        for w, l in data.pairs:
            merged.pairs.append((f"{name}:{w}", f"{name}:{l}"))
    logger.info("Pooled sources %s → %d items, %d pairs",
                names, len(merged.items), len(merged.pairs))
    return merged
