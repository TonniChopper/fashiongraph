"""VLM visual extraction — the MM4KG direction (vision enriches the graph).

Runs the vision-language model over labeled runway images to produce, per look:
(a) a natural-language **caption** (the per-look descriptions the dataset lacked),
and (b) structured **image-grounded triples** — designer → silhouette / material /
aesthetic — added to the KG (``source="vlm_runway"``). This is *annotation-free*
MMKG construction: the VLM is the labeler.

Sampled per collection (a few looks each) so a full pass is minutes, not hours.
The JSON parser is pure and unit-tested; the VLM calls are thin wrappers.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fg.config import settings
from fg.kg.schema import Triple
from fg.llm.base import Message, encode_image

logger: logging.Logger = logging.getLogger(__name__)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
VLM_SOURCE = "vlm_runway"


def build_extraction_prompt(designer: str, show: str, image_b64: str) -> list[Message]:
    """Builds the VLM prompt to extract structured facts from a runway look.

    Args:
        designer: The house/designer (known label).
        show: The collection/show (known label).
        image_b64: Base64-encoded look image.

    Returns:
        Chat messages with the image attached.
    """
    system = (
        "You are a fashion analyst. Look at the runway image and describe what "
        "you actually SEE. Output ONLY a JSON object with keys: caption (one "
        "vivid sentence), silhouettes (list), materials (list), aesthetics (list "
        "of style descriptors like 'minimalist', 'deconstructed', 'romantic'), "
        "garments (list), palette (list of colours). Be specific and visual; do "
        "not invent. No prose outside the JSON."
    )
    user = Message(
        "user",
        f"Designer: {designer}. Collection: {show}. Analyse this look and return the JSON:",
        images=[image_b64],
    )
    return [Message("system", system), user]


def parse_look(raw: str) -> dict:
    """Parses the VLM JSON response into a normalized look dict.

    Args:
        raw: The VLM's raw text response.

    Returns:
        ``{"caption", "silhouettes", "materials", "aesthetics", "garments",
        "palette"}`` with list fields (empty on failure).
    """
    empty = {"caption": "", "silhouettes": [], "materials": [],
             "aesthetics": [], "garments": [], "palette": []}
    if not raw:
        return empty
    m = _JSON_OBJ.search(raw)
    if not m:
        return empty
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty

    def _list(key: str) -> list[str]:
        v = data.get(key, [])
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()][:6]

    return {
        "caption": str(data.get("caption", "")).strip(),
        "silhouettes": _list("silhouettes"),
        "materials": _list("materials"),
        "aesthetics": _list("aesthetics"),
        "garments": _list("garments"),
        "palette": _list("palette"),
    }


def look_to_triples(look: dict, designer: str, season: str = "") -> list[Triple]:
    """Converts a parsed look into image-grounded KG triples.

    Args:
        look: A dict from :func:`parse_look`.
        designer: Subject entity (the house).
        season: Optional season → a ``from_era`` edge.

    Returns:
        Valid, in-schema triples (``source="vlm_runway"``).
    """
    triples: list[Triple] = []
    rel_field = [
        ("has_silhouette", "silhouettes", "silhouette"),
        ("uses_material", "materials", "material"),
        ("known_for", "aesthetics", "aesthetic"),
    ]
    for relation, field_name, otype in rel_field:
        for obj in look.get(field_name, []):
            triples.append(Triple(designer, relation, obj, "designer", otype, VLM_SOURCE))
    if season:
        triples.append(Triple(designer, "from_era", season, "designer", "era", VLM_SOURCE))
    return [t for t in triples if t.is_valid()]


@dataclass
class ExtractStats:
    """Counters for a VLM extraction run."""

    looks: int = 0
    triples_added: int = 0
    captions: int = 0
    per_designer: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"looks": self.looks, "triples_added": self.triples_added,
                "captions": self.captions, "per_designer": self.per_designer}


def extract_runway_kg(
    vlm: Any,
    kg: Any,
    source_root: str | Path | None = None,
    per_collection: int = 3,
    limit: int | None = None,
    captions_path: str | Path | None = None,
    on_note: Callable[[str], None] | None = None,
) -> ExtractStats:
    """Runs the VLM over sampled runway looks, enriching the KG + writing captions.

    Args:
        vlm: A vision-capable ``LLM``.
        kg: A ``KnowledgeGraph`` to add triples to.
        source_root: Runway dir; defaults to ``data/raw/vogue_runway``.
        per_collection: Looks to sample per collection (keeps runtime bounded).
        limit: Optional cap on total looks.
        captions_path: Where to append captions (JSONL); defaults under data/processed.
        on_note: Progress callback.

    Returns:
        An :class:`ExtractStats` summary.

    Raises:
        FileNotFoundError: If no runway sidecars are found.
        RuntimeError: If Pillow is unavailable.
    """
    from PIL import Image

    note = on_note or logger.info
    root = Path(source_root) if source_root else settings.data_dir / "raw" / "vogue_runway"
    jsons = sorted(root.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No runway sidecars under {root}.")

    cap_path = Path(captions_path) if captions_path else settings.data_dir / "processed" / "runway_captions.jsonl"
    cap_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample: group sidecars by collection folder, take the first N of each.
    by_collection: dict[Path, list[Path]] = defaultdict(list)
    for jp in jsons:
        by_collection[jp.parent].append(jp)
    sampled = [jp for group in by_collection.values() for jp in group[:per_collection]]

    stats = ExtractStats()
    with cap_path.open("a", encoding="utf-8") as cap_fh:
        for jp in sampled:
            try:
                info = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            img_path = jp.with_suffix(".png")
            if not img_path.exists():
                continue
            designer = info.get("designer", "")
            show = info.get("show", "")
            season = info.get("season", "")
            try:
                b64 = encode_image(Image.open(img_path))
                raw = vlm.chat(build_extraction_prompt(designer, show, b64), max_tokens=350)
            except Exception as exc:  # noqa: BLE001
                note(f"VLM failed on {img_path.name}: {exc}")
                continue

            look = parse_look(raw)
            triples = look_to_triples(look, designer, season)
            added = kg.add_triples(triples)
            stats.looks += 1
            stats.triples_added += added
            stats.per_designer[designer] = stats.per_designer.get(designer, 0) + added
            if look["caption"]:
                cap_fh.write(json.dumps({
                    "designer": designer, "show": show,
                    "caption": look["caption"], "image_path": str(img_path),
                }) + "\n")
                stats.captions += 1
            if stats.looks % 10 == 0:
                note(f"Extracted {stats.looks} looks, +{stats.triples_added} triples")
            if limit is not None and stats.looks >= limit:
                break

    logger.info("VLM extraction complete: %s", stats.as_dict())
    return stats
