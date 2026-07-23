"""Curated fabric / material knowledge → KG property edges.

Materials already exist as KG nodes (designers link to them via ``uses_material``),
but only as names. This adds their intrinsic **properties** — weight, drape,
warmth, texture, season, and hand-feel — as ``has_property`` / ``has_texture`` /
``suits_season`` edges. Shared property nodes ("lightweight", "warm") connect
fabrics to each other and enable reasoning like *"which designers favour heavy
winter fabrics"* (designer → uses_material → wool → suits_season → winter).

A curated ontology beats a scraped dataset here: the domain is small, stable, and
benefits from authoritative values. Visual texture grounding (fabric-texture
image prototypes, the "mirror node" idea) is a future extension.
"""

from __future__ import annotations

import logging
from typing import Any

from fg.kg.schema import Triple

logger: logging.Logger = logging.getLogger(__name__)

FABRIC_SOURCE = "fabric_ontology"

#: fabric → {weight, drape, warmth, season, texture[], properties[]}
FABRICS: dict[str, dict[str, Any]] = {
    "silk": {"weight": "light", "drape": "fluid", "warmth": "cool",
             "season": ["summer", "transitional"], "texture": ["smooth", "lustrous"],
             "properties": ["breathable", "lightweight", "delicate", "natural"]},
    "cotton": {"weight": "medium", "drape": "moderate", "warmth": "moderate",
               "season": ["all-season"], "texture": ["soft", "matte"],
               "properties": ["breathable", "durable", "versatile", "natural"]},
    "linen": {"weight": "light", "drape": "crisp", "warmth": "cool",
              "season": ["summer"], "texture": ["textured", "crisp"],
              "properties": ["breathable", "absorbent", "wrinkle-prone", "natural"]},
    "wool": {"weight": "heavy", "drape": "structured", "warmth": "warm",
             "season": ["winter", "autumn"], "texture": ["soft", "textured"],
             "properties": ["insulating", "resilient", "moisture-wicking", "natural"]},
    "cashmere": {"weight": "light", "drape": "soft", "warmth": "warm",
                 "season": ["winter"], "texture": ["plush", "soft"],
                 "properties": ["luxurious", "insulating", "delicate", "natural"]},
    "denim": {"weight": "heavy", "drape": "structured", "warmth": "moderate",
              "season": ["all-season"], "texture": ["rugged", "textured"],
              "properties": ["durable", "sturdy", "casual"]},
    "leather": {"weight": "heavy", "drape": "structured", "warmth": "warm",
                "season": ["autumn", "winter"], "texture": ["smooth", "supple"],
                "properties": ["durable", "protective", "luxurious"]},
    "suede": {"weight": "medium", "drape": "soft", "warmth": "warm",
              "season": ["autumn", "winter"], "texture": ["napped", "soft"],
              "properties": ["soft", "textured", "luxurious"]},
    "velvet": {"weight": "heavy", "drape": "soft", "warmth": "warm",
               "season": ["winter"], "texture": ["plush", "soft"],
               "properties": ["luxurious", "light-absorbing", "opulent"]},
    "chiffon": {"weight": "light", "drape": "fluid", "warmth": "cool",
                "season": ["summer"], "texture": ["sheer", "airy"],
                "properties": ["delicate", "floaty", "sheer"]},
    "satin": {"weight": "medium", "drape": "fluid", "warmth": "moderate",
              "season": ["all-season"], "texture": ["glossy", "smooth"],
              "properties": ["lustrous", "elegant", "slippery"]},
    "tweed": {"weight": "heavy", "drape": "structured", "warmth": "warm",
              "season": ["winter", "autumn"], "texture": ["coarse", "textured"],
              "properties": ["durable", "insulating", "heritage"]},
    "corduroy": {"weight": "medium", "drape": "structured", "warmth": "warm",
                 "season": ["autumn", "winter"], "texture": ["ribbed", "textured"],
                 "properties": ["durable", "casual", "retro"]},
    "jersey": {"weight": "light", "drape": "fluid", "warmth": "moderate",
               "season": ["all-season"], "texture": ["soft", "smooth"],
               "properties": ["stretchy", "comfortable", "versatile"]},
    "tulle": {"weight": "light", "drape": "stiff", "warmth": "cool",
              "season": ["all-season"], "texture": ["netted", "sheer"],
              "properties": ["voluminous", "sheer", "decorative"]},
    "lace": {"weight": "light", "drape": "structured", "warmth": "cool",
             "season": ["all-season"], "texture": ["openwork", "delicate"],
             "properties": ["decorative", "romantic", "delicate"]},
    "crepe": {"weight": "medium", "drape": "fluid", "warmth": "moderate",
              "season": ["all-season"], "texture": ["crinkled", "matte"],
              "properties": ["drapey", "textured", "elegant"]},
    "organza": {"weight": "light", "drape": "stiff", "warmth": "cool",
                "season": ["all-season"], "texture": ["sheer", "crisp"],
                "properties": ["structured", "sheer", "ethereal"]},
    "taffeta": {"weight": "medium", "drape": "stiff", "warmth": "moderate",
                "season": ["all-season"], "texture": ["crisp", "smooth"],
                "properties": ["structured", "lustrous", "rustling"]},
    "flannel": {"weight": "medium", "drape": "soft", "warmth": "warm",
                "season": ["winter", "autumn"], "texture": ["brushed", "soft"],
                "properties": ["cozy", "warm", "casual"]},
    "gabardine": {"weight": "medium", "drape": "structured", "warmth": "moderate",
                  "season": ["all-season"], "texture": ["smooth", "tight"],
                  "properties": ["durable", "water-resistant", "tailored"]},
    "mohair": {"weight": "light", "drape": "soft", "warmth": "warm",
               "season": ["winter"], "texture": ["fuzzy", "fluffy"],
               "properties": ["insulating", "textured", "natural"]},
    "alpaca": {"weight": "light", "drape": "soft", "warmth": "warm",
               "season": ["winter"], "texture": ["plush", "soft"],
               "properties": ["insulating", "hypoallergenic", "luxurious"]},
    "viscose": {"weight": "light", "drape": "fluid", "warmth": "cool",
                "season": ["summer", "transitional"], "texture": ["smooth", "soft"],
                "properties": ["drapey", "breathable", "semi-synthetic"]},
    "polyester": {"weight": "medium", "drape": "moderate", "warmth": "moderate",
                  "season": ["all-season"], "texture": ["smooth", "matte"],
                  "properties": ["durable", "wrinkle-resistant", "synthetic"]},
    "nylon": {"weight": "light", "drape": "fluid", "warmth": "moderate",
              "season": ["all-season"], "texture": ["smooth", "slick"],
              "properties": ["strong", "water-resistant", "synthetic"]},
    "twill": {"weight": "medium", "drape": "structured", "warmth": "moderate",
              "season": ["all-season"], "texture": ["diagonal", "textured"],
              "properties": ["durable", "structured", "tailored"]},
    "poplin": {"weight": "light", "drape": "crisp", "warmth": "cool",
               "season": ["summer", "transitional"], "texture": ["smooth", "crisp"],
               "properties": ["breathable", "crisp", "tailored"]},
    "fleece": {"weight": "medium", "drape": "soft", "warmth": "warm",
               "season": ["winter"], "texture": ["plush", "soft"],
               "properties": ["warm", "cozy", "synthetic"]},
}


def fabrics_to_triples(fabrics: dict[str, dict] | None = None) -> list[Triple]:
    """Converts the fabric ontology into KG triples.

    Args:
        fabrics: Fabric dict; defaults to :data:`FABRICS`.

    Returns:
        Valid, in-schema triples (``source="fabric_ontology"``).
    """
    fabrics = fabrics or FABRICS
    triples: list[Triple] = []
    for name, f in fabrics.items():
        props = list(f.get("properties", []))
        for key in ("weight", "drape", "warmth"):
            if f.get(key):
                props.append(f"{f[key]} {key}" if key in ("weight", "drape") else f[key])
        for p in props:
            triples.append(Triple(name, "has_property", p, "material", "property", FABRIC_SOURCE))
        for t in f.get("texture", []):
            triples.append(Triple(name, "has_texture", t, "material", "texture", FABRIC_SOURCE))
        for s in f.get("season", []):
            triples.append(Triple(name, "suits_season", s, "material", "season", FABRIC_SOURCE))
    return [t for t in triples if t.is_valid()]


def add_fabrics_to_kg(kg: Any, fabrics: dict[str, dict] | None = None) -> int:
    """Adds the fabric ontology to a knowledge graph.

    Args:
        kg: A ``KnowledgeGraph``.
        fabrics: Optional fabric dict override.

    Returns:
        Number of new triples inserted.
    """
    triples = fabrics_to_triples(fabrics)
    added = kg.add_triples(triples)
    logger.info("Added %d fabric triples (%d fabrics).", added, len(fabrics or FABRICS))
    return added
