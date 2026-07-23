"""Fashion knowledge-graph schema — entity types, relations, and the Triple.

Deliberately small and fixed. A constrained schema is what makes LLM-assisted
extraction reliable (per the Farfetch NER+EL approach): the model fills known
slots instead of inventing structure. Entities and relations normalise to a
canonical vocabulary so "Prada" and "prada" collapse to one node.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

#: Node types in the fashion KG.
ENTITY_TYPES: frozenset[str] = frozenset({
    "brand", "designer", "collection", "garment", "silhouette",
    "material", "colour", "era", "trend", "aesthetic", "city",
    "property", "season", "texture",
})

#: Allowed relations (subject_type hints in comments).
RELATION_TYPES: frozenset[str] = frozenset({
    "founded_by",          # brand -> designer
    "creative_director",   # brand -> designer
    "part_of",             # collection/garment -> brand/collection
    "uses_material",       # brand/garment -> material
    "has_silhouette",      # brand/collection/garment -> silhouette
    "from_era",            # brand/trend/garment -> era
    "associated_with",     # brand/designer -> trend/aesthetic
    "influenced_by",       # brand/designer -> brand/designer/aesthetic
    "known_for",           # brand/designer -> garment/silhouette/aesthetic
    "based_in",            # brand -> city
    "successor_of",        # designer -> designer
    "collaborated_with",   # brand/designer -> brand/designer
    "has_property",        # material -> property (lightweight, breathable, …)
    "suits_season",        # material -> season
    "has_texture",         # material -> texture (smooth, plush, crisp, …)
})


#: Maps common LLM relation variants → canonical relations, so we recover
#: triples that would otherwise be dropped for being "off-schema".
RELATION_SYNONYMS: dict[str, str] = {
    # based_in
    "located_in": "based_in", "headquartered_in": "based_in", "from": "based_in",
    "originates_from": "based_in", "founded_in": "based_in",
    # founded_by
    "founded": "founded_by", "established_by": "founded_by",
    "created_by": "founded_by", "started_by": "founded_by",
    # creative_director
    "designed_by": "creative_director", "led_by": "creative_director",
    "artistic_director": "creative_director", "headed_by": "creative_director",
    "director": "creative_director", "helmed_by": "creative_director",
    # uses_material
    "made_of": "uses_material", "made_from": "uses_material",
    "uses": "uses_material", "features_material": "uses_material",
    "material": "uses_material",
    # part_of
    "belongs_to": "part_of", "sub_brand_of": "part_of",
    "owned_by": "part_of", "subsidiary_of": "part_of", "division_of": "part_of",
    # has_silhouette
    "silhouette": "has_silhouette", "signature_silhouette": "has_silhouette",
    # from_era
    "era": "from_era", "dates_from": "from_era", "period": "from_era",
    # associated_with
    "linked_to": "associated_with", "connected_to": "associated_with",
    "related_to": "associated_with", "part_of_movement": "associated_with",
    # influenced_by
    "inspired_by": "influenced_by", "influence": "influenced_by",
    # known_for
    "famous_for": "known_for", "recognized_for": "known_for",
    "celebrated_for": "known_for", "iconic_for": "known_for",
    "pioneered": "known_for", "specializes_in": "known_for",
    # collaborated_with
    "collaborated": "collaborated_with", "worked_with": "collaborated_with",
    "partnered_with": "collaborated_with",
    # successor_of
    "succeeded": "successor_of", "replaced": "successor_of",
    "predecessor_of": "successor_of",
    # fabric relations
    "property": "has_property", "characterized_by": "has_property",
    "feels": "has_texture", "texture": "has_texture", "feels_like": "has_texture",
    "suited_for": "suits_season", "good_for_season": "suits_season",
    "worn_in": "suits_season", "season": "suits_season",
}


#: Trailing tokens stripped during entity resolution (legal/product suffixes).
_CORPORATE_SUFFIXES: frozenset[str] = frozenset({
    "se", "sa", "inc", "ltd", "llc", "group", "holding", "holdings",
    "couture", "parfums", "parfum", "beauty", "cosmetics", "eyewear",
    "watchmaking", "freres", "brand", "label", "maison", "house",
})

#: Explicit variant → canonical merges (curated; extend as needed).
_ENTITY_ALIASES: dict[str, str] = {
    "christian dior": "dior", "dior homme": "dior", "miss dior": "dior",
    "yves saint laurent": "saint laurent", "ysl": "saint laurent",
    "saint laurent paris": "saint laurent",
    "louis vuitton malletier": "louis vuitton",
    "cristobal balenciaga": "balenciaga",
    "gabrielle chanel": "chanel", "coco chanel": "chanel",
    "the row": "the row",  # protect from "the" handling
}

_YEAR_TOKEN_RE = re.compile(r"^(1[89]\d0s?|20\d0s?|\d{4})$")


def _strip_suffixes(key: str) -> str:
    """Repeatedly drops trailing corporate/product tokens (keeps ≥1 token)."""
    tokens = key.split()
    while len(tokens) > 1 and tokens[-1] in _CORPORATE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def canonical_relation(relation: str) -> str:
    """Maps a raw relation to a canonical one, or ``""`` if unmappable.

    Args:
        relation: Raw relation string from the model.

    Returns:
        A member of :data:`RELATION_TYPES`, or ``""`` if it can't be mapped.
    """
    rel = "_".join(relation.strip().lower().split()).replace("-", "_")
    if rel in RELATION_TYPES:
        return rel
    return RELATION_SYNONYMS.get(rel, "")


def normalize_entity(name: str) -> str:
    """Canonicalises an entity surface form for node identity.

    Folds accents (``Céline`` → ``celine``), lowercases, converts underscores/
    hyphens/punctuation to spaces (fixing ``christian_dior`` → ``christian
    dior``), and collapses whitespace — so trivial variants map to one node.
    Display casing is preserved separately by the store.

    Note: this is *normalisation*, not full entity *linking* (merging
    ``Christian Dior`` with ``Dior``, or ``Dior Homme`` with ``Dior``). That
    alias-resolution step is a separate, more careful pass.

    Args:
        name: Raw entity string.

    Returns:
        The canonical key (may be empty if input was blank).
    """
    # Strip accents/diacritics: Céline → Celine.
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower().replace("_", " ").replace("-", " ")
    folded = _PUNCT_RE.sub(" ", folded)  # drop remaining punctuation (’, ., &, …)
    return " ".join(folded.split())


def canonical_entity(name: str) -> str:
    """Resolves a surface form to a canonical node key (the entity-linking step).

    Pipeline: :func:`normalize_entity` → strip corporate/product suffixes →
    apply the alias map. So ``Christian Dior Couture`` → ``dior`` and ``YSL`` →
    ``saint laurent``. This is the Farfetch "link" step; keep the alias map
    conservative to avoid wrong merges.

    Args:
        name: Raw entity string.

    Returns:
        The canonical node key.
    """
    key = normalize_entity(name)
    if key in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[key]
    key = _strip_suffixes(key)
    return _ENTITY_ALIASES.get(key, key)


def is_plausible_entity(name: str) -> bool:
    """Rejects extraction-noise entities (phrase fragments, years, stubs).

    Filters the junk we observed (``ne``, ``john galliano from givenchy dior
    and his eponymous line``, ``1950s christian dior silhouettes``).

    Args:
        name: Raw entity surface form.

    Returns:
        ``True`` if the entity looks like a real, concise fashion entity.
    """
    key = normalize_entity(name)
    if len(key) < 3:                          # stubs like "ne" (sliced "Céli|ne")
        return False
    words = key.split()
    if len(words) > 6:                       # phrase, not an entity
        return False
    # A bare decade ("1990s") and a season ("Fall 2026") are valid; reject a year
    # only inside a longer phrase ("ysl fall 1960 dior collection").
    if len(words) > 2 and any(_YEAR_TOKEN_RE.match(w) for w in words):
        return False
    # Sentence-fragment markers.
    if any(m in f" {key} " for m in (" from ", " and his ", " including ", " where ")):
        return False
    return True


@dataclass(frozen=True)
class Triple:
    """A single subject–relation–object fact.

    Attributes:
        subject: Head entity (surface form).
        relation: One of :data:`RELATION_TYPES`.
        object: Tail entity (surface form).
        subject_type: One of :data:`ENTITY_TYPES` (or ``""`` if unknown).
        object_type: One of :data:`ENTITY_TYPES` (or ``""`` if unknown).
        source: Provenance (e.g. the document title it came from).
    """

    subject: str
    relation: str
    object: str
    subject_type: str = ""
    object_type: str = ""
    source: str = ""

    def is_valid(self) -> bool:
        """Returns whether the triple is well-formed, in-schema, and non-noisy."""
        return bool(
            self.subject.strip()
            and self.object.strip()
            and self.relation in RELATION_TYPES
            and self.subject_key != self.object_key
            and is_plausible_entity(self.subject)
            and is_plausible_entity(self.object)
        )

    @property
    def subject_key(self) -> str:
        """Canonical (resolved) subject key."""
        return canonical_entity(self.subject)

    @property
    def object_key(self) -> str:
        """Canonical (resolved) object key."""
        return canonical_entity(self.object)
