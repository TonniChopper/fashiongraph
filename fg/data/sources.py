"""Source loaders — turn each raw dataset into normalized ``Document`` objects.

A small registry so the ingest CLI can list and build sources by name. Heavy
libs (pandas/pyarrow) are imported lazily inside loaders so importing this
module stays cheap and dependency-free.

Add a new dataset by writing a loader and registering a ``SourceSpec``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fg.config import settings
from fg.data.schema import Document

logger: logging.Logger = logging.getLogger(__name__)

#: Attribute columns we know how to describe (subset used if present).
_PRODUCT_ATTR_COLS: tuple[str, ...] = (
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
)


def _find_tabular_files(root: Path) -> list[Path]:
    """Returns parquet/csv files under *root* (recursively), sorted."""
    files = sorted(
        [*root.rglob("*.parquet"), *root.rglob("*.csv")]
    )
    return files


def load_fashion_products(root: Path) -> Iterator[Document]:
    """Loads catalog products (e.g. ashraq/fashion-product-images-small).

    Builds a descriptive sentence per product from whatever attribute columns
    are present, plus provenance metadata.

    Args:
        root: Directory containing product parquet/csv files.

    Yields:
        One ``Document`` per product row.

    Raises:
        FileNotFoundError: If no tabular files are found under *root*.
        RuntimeError: If pandas is unavailable.
    """
    files = _find_tabular_files(root)
    if not files:
        raise FileNotFoundError(
            f"No parquet/csv found under {root}. Run the download script first."
        )
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas required: pip install pandas pyarrow") from exc

    for fp in files:
        df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)
        cols = set(df.columns)
        name_col = next(
            (c for c in ("productDisplayName", "name", "title", "caption")
             if c in cols),
            None,
        )
        attr_cols = [c for c in _PRODUCT_ATTR_COLS if c in cols]
        for row in df.itertuples(index=False):
            rowd = row._asdict()
            text = describe_product(rowd, name_col, attr_cols)
            if not text:
                continue
            meta = {
                "source": "fashion_products",
                "source_type": "catalog",
                "title": str(rowd.get(name_col, "")) if name_col else "",
            }
            for c in ("baseColour", "articleType", "season", "year", "usage"):
                if c in rowd and rowd[c] is not None:
                    key = {"baseColour": "colour", "articleType": "category"}.get(c, c)
                    meta[key] = str(rowd[c])
            yield Document(text=text, metadata=meta)


def describe_product(
    row: dict, name_col: str | None, attr_cols: list[str]
) -> str:
    """Builds a natural-language product description from a catalog row.

    Args:
        row: Mapping of column → value for one product.
        name_col: Column holding the product name/title, if any.
        attr_cols: Attribute columns present in this table.

    Returns:
        A descriptive sentence, or ``""`` if there's nothing usable.
    """
    name = str(row.get(name_col, "")).strip() if name_col else ""
    parts: list[str] = []
    for c in attr_cols:
        val = row.get(c)
        if val is None:
            continue
        val = str(val).strip()
        if val and val.lower() not in {"nan", "none", ""}:
            parts.append(f"{c}: {val}")
    if not name and not parts:
        return ""
    head = name or "Fashion product"
    return f"{head}. " + "; ".join(parts) if parts else head


def load_text_files(root: Path) -> Iterator[Document]:
    """Loads a directory of ``.txt`` knowledge files (e.g. Wikipedia corpus).

    The filename (minus extension, dashes → spaces) becomes the title.

    Args:
        root: Directory of ``.txt`` files.

    Yields:
        One ``Document`` per file.

    Raises:
        FileNotFoundError: If no ``.txt`` files are found.
    """
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files under {root}.")
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        title = fp.stem.replace("-", " ").replace("_", " ").strip()
        yield Document(
            text=text,
            metadata={
                "source": "wikipedia",
                "source_type": "encyclopedia",
                "title": title,
            },
        )


def load_jsonl_instructions(root: Path) -> Iterator[Document]:
    """Loads style-instruction JSONL/parquet as reference styling examples.

    Used for the RAG "styling examples" surface (the same data also seeds the
    Phase-5 LoRA set). Reads ``input``/``context``/``completion`` triples.

    Args:
        root: Directory containing the instruction files.

    Yields:
        One ``Document`` per styling example.

    Raises:
        FileNotFoundError: If no instruction files are found.
    """
    files = sorted([*root.rglob("*.jsonl"), *root.rglob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No instruction files under {root}.")
    rows: list[dict] = []
    for fp in files:
        if fp.suffix == ".parquet":
            import pandas as pd

            rows.extend(pd.read_parquet(fp).to_dict("records"))
        else:
            with fp.open(encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
    for r in rows:
        inp = str(r.get("input", "")).strip()
        ctx = str(r.get("context", "")).strip()
        out = str(r.get("completion", r.get("response", ""))).strip()
        if not out:
            continue
        text = f"Styling example. Profile: {inp}. Occasion: {ctx}. Advice: {out}"
        yield Document(
            text=text,
            metadata={"source": "style_instruct", "source_type": "styling"},
        )


@dataclass(frozen=True)
class SourceSpec:
    """Describes one ingestible source.

    Attributes:
        name: Registry key (used in the CLI).
        description: Human-readable description.
        subdir: Default sub-directory under ``data/raw`` holding the files.
        loader: Callable ``(root: Path) -> Iterator[Document]``.
    """

    name: str
    description: str
    subdir: str
    loader: Callable[[Path], Iterator[Document]]

    def default_root(self) -> Path:
        """Returns the default raw-data directory for this source."""
        return settings.data_dir / "raw" / self.subdir


#: The source registry. Add new datasets here.
SOURCES: dict[str, SourceSpec] = {
    "fashion_products": SourceSpec(
        "fashion_products",
        "Catalog products with attributes (ashraq/fashion-product-images-small)",
        "fashion-product-images-small",
        load_fashion_products,
    ),
    "wikipedia": SourceSpec(
        "wikipedia",
        "Curated Wikipedia fashion knowledge (houses, designers, eras, fabrics)",
        "wikipedia",
        load_text_files,
    ),
    "style_instruct": SourceSpec(
        "style_instruct",
        "Styling examples (neuralwork/fashion-style-instruct)",
        "fashion-style-instruct",
        load_jsonl_instructions,
    ),
}


def get_source(name: str) -> SourceSpec:
    """Looks up a source spec by name.

    Args:
        name: Registry key.

    Returns:
        The matching ``SourceSpec``.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in SOURCES:
        raise KeyError(
            f"Unknown source {name!r}. Known: {', '.join(sorted(SOURCES))}"
        )
    return SOURCES[name]
