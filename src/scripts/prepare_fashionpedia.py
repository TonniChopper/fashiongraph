"""Convert the Fashionpedia dataset to DeepFashion2Dataset format.

Loads a Fashionpedia dataset previously saved with
``datasets.Dataset.save_to_disk()`` and exports it as:

- ``data/raw/deepfashion2/images/{idx:06d}.jpg``
- ``data/raw/deepfashion2/train_annotations.csv``
  with columns: ``image_path``, ``category``, ``attributes``

Usage::

    python -m src.scripts.prepare_fashionpedia
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import load_from_disk, Dataset
from PIL import Image

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

FASHIONPEDIA_PATH: str = (
    "/content/drive/MyDrive/fashiongraph/data/fashionpedia"
)
OUTPUT_DIR: Path = Path("data/raw/deepfashion2")
IMAGES_DIR: Path = OUTPUT_DIR / "images"
CSV_PATH: Path = OUTPUT_DIR / "train_annotations.csv"
LOG_EVERY: int = 1000

# ------------------------------------------------------------------
# Fashionpedia label maps
# ------------------------------------------------------------------

SUPERCATEGORIES: dict[int, str] = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
    26: "umbrella",
    27: "hood",
    28: "collar",
    29: "lapel",
    30: "epaulette",
    31: "sleeve",
    32: "pocket",
    33: "neckline",
    34: "buckle",
    35: "zipper",
    36: "applique",
    37: "bead",
    38: "bow",
    39: "flower",
    40: "fringe",
    41: "ribbon",
    42: "rivet",
    43: "ruffle",
    44: "sequin",
    45: "tassel",
}

ATTRIBUTES: dict[int, str] = {
    0: "floral",
    1: "graphic",
    2: "striped",
    3: "solid",
    4: "lattice",
    5: "dotted",
    6: "checked",
    7: "herringbone",
    8: "houndstooth",
    9: "paisley",
    10: "toile",
    11: "camouflage",
    12: "plain",
    13: "denim",
    14: "corduroy",
    15: "velvet",
    16: "leather",
    17: "faux leather",
    18: "sheer",
    19: "satin",
    20: "silk",
    21: "sequined",
    22: "knit",
    23: "lace",
    24: "fur",
    25: "faux fur",
    26: "feather",
    27: "cotton",
    28: "chiffon",
    29: "nylon",
    30: "mesh",
    31: "tweed",
    32: "fleece",
    33: "terrycloth",
    34: "suede",
    35: "woven",
    36: "long sleeve",
    37: "short sleeve",
    38: "sleeveless",
    39: "maxi length",
    40: "midi length",
    41: "mini length",
    42: "crew neckline",
    43: "v neckline",
    44: "turtleneck",
    45: "sweetheart neckline",
    46: "asymmetric",
    47: "one shoulder",
    48: "off the shoulder",
    49: "halter",
    50: "strapless",
    51: "high-low",
    52: "tiered",
    53: "pleated",
    54: "a-line",
    55: "ruffled",
    56: "straight",
    57: "peplum",
    58: "wrap",
    59: "oversized",
    60: "cropped",
    61: "distressed",
    62: "embroidered",
    63: "embellished",
    64: "frayed",
    65: "tapered",
    66: "slim fit",
    67: "wide leg",
    68: "high waisted",
    69: "low waisted",
    70: "bell bottom",
    71: "drawstring",
    72: "layered",
    73: "hooded",
    74: "belted",
    75: "double breasted",
    76: "single breasted",
    77: "collarless",
    78: "notched lapel",
    79: "shawl lapel",
    80: "peak lapel",
    81: "patch pocket",
    82: "welt pocket",
    83: "flap pocket",
    84: "zip pocket",
    85: "cargo pocket",
    86: "button down",
    87: "henley",
    88: "polo",
    89: "smocked",
    90: "shirred",
    91: "gathered",
    92: "cutout",
    93: "slit",
    94: "backless",
    95: "metallic",
    96: "neon",
    97: "tie dye",
    98: "color block",
    99: "animal print",
    100: "tropical print",
    101: "abstract print",
    102: "geometric print",
    103: "logo",
    104: "text",
    105: "applique",
    106: "beaded",
    107: "crystal",
    108: "studded",
    109: "fringed",
    110: "tasseled",
    111: "ribbon trimmed",
    112: "lace trimmed",
    113: "fur trimmed",
    114: "chain detail",
    115: "buckle detail",
    116: "zipper detail",
    117: "button detail",
    118: "snap detail",
    119: "velcro",
    120: "padded",
    121: "quilted",
    122: "lined",
    123: "unlined",
    124: "waterproof",
    125: "insulated",
    126: "reversible",
    127: "convertible",
    128: "adjustable",
    129: "stretch",
    130: "non-stretch",
    131: "shiny",
    132: "matte",
    133: "textured",
    134: "smooth",
    135: "transparent",
    136: "opaque",
    137: "lightweight",
    138: "heavyweight",
    139: "mid-weight",
    140: "breathable",
    141: "moisture-wicking",
    142: "wrinkle-resistant",
    143: "machine washable",
    144: "dry clean only",
    145: "hand wash",
    146: "iron safe",
    147: "bleach safe",
    148: "tumble dry",
    149: "hang dry",
    150: "sustainable",
    151: "organic",
    152: "recycled",
    153: "vegan",
    154: "fair trade",
    155: "handmade",
    156: "limited edition",
    157: "vintage",
    158: "designer",
    159: "luxury",
    160: "casual",
    161: "formal",
    162: "sporty",
    163: "bohemian",
    164: "minimalist",
    165: "maximalist",
    166: "preppy",
    167: "punk",
    168: "goth",
    169: "streetwear",
    170: "athleisure",
    171: "resort wear",
    172: "workwear",
    173: "evening wear",
    174: "bridal",
    175: "maternity",
    176: "plus size",
    177: "petite",
    178: "tall",
    179: "unisex",
    180: "gender neutral",
    181: "kids",
    182: "baby",
    183: "toddler",
    184: "teen",
    185: "adult",
    186: "senior",
    187: "custom fit",
    188: "standard fit",
    189: "relaxed fit",
    190: "fitted",
    191: "loose fit",
    192: "bodycon",
    193: "boxy",
    194: "tailored",
    195: "structured",
    196: "unstructured",
    197: "draped",
    198: "gathered waist",
    199: "empire waist",
    200: "drop waist",
    201: "natural waist",
    202: "no waist",
    203: "elasticated",
    204: "button closure",
    205: "zip closure",
    206: "hook and eye",
    207: "tie closure",
    208: "toggle closure",
    209: "magnetic closure",
    210: "pullover",
    211: "open front",
    212: "wrap front",
    213: "concealed closure",
    214: "exposed closure",
    215: "back closure",
    216: "side closure",
    217: "front closure",
    218: "shoulder closure",
    219: "no closure",
    220: "multi-way",
    221: "detachable",
    222: "built-in bra",
    223: "underwire",
    224: "wireless",
    225: "padded bra",
    226: "unpadded",
    227: "push up",
    228: "sports bra",
    229: "bralette",
    230: "bandeau",
    231: "halter bra",
    232: "racerback",
    233: "t-shirt bra",
    234: "plunge bra",
    235: "balconette",
    236: "full coverage",
    237: "demi coverage",
    238: "triangle",
    239: "longline",
    240: "multiway bra",
    241: "adhesive bra",
    242: "nipple cover",
    243: "pasties",
    244: "breast form",
    245: "breast enhancer",
    246: "breast minimizer",
    247: "nursing bra",
    248: "mastectomy bra",
    249: "post-surgery bra",
    250: "compression",
    251: "shapewear",
    252: "body shaper",
    253: "waist cincher",
    254: "corset",
    255: "bustier",
    256: "camisole",
    257: "slip",
    258: "teddy",
    259: "bodysuit",
    260: "romper",
    261: "catsuit",
    262: "unitard",
    263: "leotard",
    264: "onesie",
    265: "robe",
    266: "kimono",
    267: "caftan",
    268: "muumuu",
    269: "poncho",
    270: "shawl",
    271: "wrap",
    272: "stole",
    273: "bolero",
    274: "shrug",
    275: "capelet",
    276: "mantle",
    277: "cloak",
    278: "tabard",
    279: "tunic",
    280: "kaftan",
    281: "dashiki",
    282: "kurta",
    283: "salwar kameez",
    284: "sari",
    285: "lehenga",
    286: "cheongsam",
    287: "qipao",
    288: "ao dai",
    289: "hanbok",
    290: "yukata",
    291: "dirndl",
    292: "lederhosen",
    293: "kilt",
    294: "sarong",
    295: "pareo",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_categories(
    category_ids: list[int],
) -> str:
    """Maps numeric category IDs to supercategory names.

    Args:
        category_ids: List of integer category IDs from Fashionpedia.

    Returns:
        Semicolon-joined string of unique supercategory names.
    """
    names: list[str] = []
    seen: set[str] = set()
    for cid in category_ids:
        name: str = SUPERCATEGORIES.get(cid, f"unknown_{cid}")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return "; ".join(names) if names else "unknown"


def _resolve_attributes(
    attribute_ids: list[int],
) -> str:
    """Maps numeric attribute IDs to human-readable names.

    Args:
        attribute_ids: List of integer attribute IDs from Fashionpedia.

    Returns:
        Comma-joined string of attribute names.
    """
    names: list[str] = []
    seen: set[str] = set()
    for aid in attribute_ids:
        name: str = ATTRIBUTES.get(aid, f"attr_{aid}")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return ", ".join(names) if names else ""


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def main() -> None:
    """Entry point: loads Fashionpedia and exports to DeepFashion2 format."""
    logger.info("Loading Fashionpedia dataset from %s …", FASHIONPEDIA_PATH)
    try:
        ds: Dataset = load_from_disk(FASHIONPEDIA_PATH)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        raise

    logger.info("Dataset loaded: %d samples.", len(ds))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped: int = 0

    for idx in range(len(ds)):
        sample = ds[idx]

        # ---- image --------------------------------------------------
        image = sample.get("image")
        if image is None:
            logger.warning("[%d] No image field, skipping.", idx)
            skipped += 1
            continue

        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as exc:
                logger.warning("[%d] Cannot open image: %s", idx, exc)
                skipped += 1
                continue

        image_filename: str = f"{idx:06d}.jpg"
        image_dest: Path = IMAGES_DIR / image_filename

        if not image_dest.exists():
            try:
                image.convert("RGB").save(str(image_dest), "JPEG", quality=95)
            except Exception as exc:
                logger.warning("[%d] Failed to save image: %s", idx, exc)
                skipped += 1
                continue

        # ---- categories ---------------------------------------------
        cat_ids: list[int] = sample.get("categories", sample.get("category_id", []))
        if isinstance(cat_ids, int):
            cat_ids = [cat_ids]
        category: str = _resolve_categories(cat_ids)

        # ---- attributes ---------------------------------------------
        attr_ids: list[int] = sample.get("attributes", sample.get("attribute_ids", []))
        if isinstance(attr_ids, int):
            attr_ids = [attr_ids]
        attributes: str = _resolve_attributes(attr_ids)

        rows.append({
            "image_path": image_filename,
            "category": category,
            "attributes": attributes,
        })

        if (idx + 1) % LOG_EVERY == 0:
            logger.info(
                "Progress: %d / %d items processed (%d skipped).",
                idx + 1, len(ds), skipped,
            )

    # ---- write CSV --------------------------------------------------
    df: pd.DataFrame = pd.DataFrame(rows)

    try:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write CSV: %s", exc)
        raise

    logger.info(
        "✅ Done. Saved %d rows to %s (%d skipped). Images in %s.",
        len(df), CSV_PATH, skipped, IMAGES_DIR,
    )


if __name__ == "__main__":
    main()

