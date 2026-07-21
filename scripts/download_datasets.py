#!/usr/bin/env python3
"""Safe dataset downloader for FashionGraph (run on your machine).

Design goals:
  * Only ever writes under ``data/raw/`` — never deletes anything.
  * Idempotent — skips a dataset if its folder already exists (use --force to
    re-download).
  * No Kaggle credentials needed: pulls public mirrors from the Hugging Face Hub,
    the Surrey aesthetics zip, and a curated Wikipedia page list.
  * Big datasets are opt-in (you must name their set explicitly).

Usage:
    pip install huggingface_hub requests
    python scripts/download_datasets.py --sets core            # default, light
    python scripts/download_datasets.py --sets core,aesthetics
    python scripts/download_datasets.py --sets text            # LLM corpus
    python scripts/download_datasets.py --list                 # show sets
    python scripts/download_datasets.py --sets hnm --force     # big, opt-in

Sets:
    core        product attributes (small) + styling seed + Wikipedia knowledge
    text        LLM corpus: styling seed + fashion200k descriptions + H&M captions + Wikipedia
    vision      product images (small) + Fashionpedia
    aesthetics  Surrey body-shape + aesthetic pairwise dataset (small zip)
    hnm         FULL H&M recommendations (LARGE — opt-in)
    art         WikiArt (art movements, for aesthetic lineage) — LARGE
    polyvore    Polyvore outfits: likes + compatibility (LARGE) — outfit-level taste
    aesthetics-plus  AVA subset + TAD66K general image-aesthetics priors (LARGE)
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "raw"

# Hugging Face dataset repos → target sub-folder under data/raw/
HF_DATASETS: dict[str, str] = {
    "ashraq/fashion-product-images-small": "fashion-product-images-small",
    "neuralwork/fashion-style-instruct": "fashion-style-instruct",
    "Marqo/fashion200k": "fashion200k",
    "tomytjandra/h-and-m-fashion-caption": "hm-caption",
    "detection-datasets/fashionpedia": "fashionpedia",
    "einrafh/hnm-fashion-recommendations-data": "hnm-full",
    "huggan/wikiart": "wikiart",
    # Outfit-level fashion data with implicit preference (likes) + compatibility.
    "Marqo/polyvore": "polyvore",
    "owj0421/polyvore-outfits": "polyvore-outfits",
    # General image-aesthetics priors (human ratings) — transfer signal.
    "trojblue/AVA-aesthetics-10pct-min50-10bins": "ava-aesthetics",
    "Shuai1995/TAD66K_for_Image_Aesthetics_Assessment": "tad66k",
}

# Named sets → list of tasks. Each task: ("hf", repo_id) | ("zip", url, subdir) | ("wiki",)
SETS: dict[str, list[tuple]] = {
    "core": [
        ("hf", "ashraq/fashion-product-images-small"),
        ("hf", "neuralwork/fashion-style-instruct"),
        ("wiki",),
    ],
    "text": [
        ("hf", "neuralwork/fashion-style-instruct"),
        ("hf", "Marqo/fashion200k"),
        ("hf", "tomytjandra/h-and-m-fashion-caption"),
        ("wiki",),
    ],
    "vision": [
        ("hf", "ashraq/fashion-product-images-small"),
        ("hf", "detection-datasets/fashionpedia"),
    ],
    "aesthetics": [
        ("zip",
         "http://kahlan.eps.surrey.ac.uk/featurespace/fashion/fashion_data.zip",
         "surrey-aesthetics"),
    ],
    "hnm": [
        ("hf", "einrafh/hnm-fashion-recommendations-data"),
    ],
    "art": [
        ("hf", "huggan/wikiart"),
    ],
    "polyvore": [
        ("hf", "Marqo/polyvore"),
        ("hf", "owj0421/polyvore-outfits"),
    ],
    "aesthetics-plus": [
        ("hf", "trojblue/AVA-aesthetics-10pct-min50-10bins"),
        ("hf", "Shuai1995/TAD66K_for_Image_Aesthetics_Assessment"),
    ],
}

BIG_SETS = {"hnm", "art", "polyvore", "aesthetics-plus"}

# Curated Wikipedia pages — houses, designers, eras, garments, fabrics, weeks.
WIKI_TITLES: list[str] = [
    "Fashion", "History of fashion design", "Haute couture", "Ready-to-wear",
    "Fashion design", "Fashion week", "Paris Fashion Week", "Milan Fashion Week",
    "Chanel", "Christian Dior", "Gucci", "Prada", "Louis Vuitton", "Balenciaga",
    "Versace", "Givenchy", "Yves Saint Laurent (brand)", "Alexander McQueen (brand)",
    "Bottega Veneta", "Loewe (brand)", "Hermès", "Burberry", "Valentino (fashion house)",
    "Comme des Garçons", "Maison Margiela", "The Row (fashion label)", "Miu Miu",
    "Coco Chanel", "Karl Lagerfeld", "Virgil Abloh", "Rei Kawakubo", "Miuccia Prada",
    "Phoebe Philo", "Rick Owens", "Vivienne Westwood",
    "Little black dress", "Trench coat", "Denim", "Little black dress",
    "Streetwear", "Athleisure", "Normcore", "Quiet luxury", "Sustainable fashion",
    "1920s in Western fashion", "1970s in Western fashion", "1990s in fashion",
    "2020s in fashion", "Silk", "Cashmere wool", "Tweed",
]


def _skip(target: Path, force: bool) -> bool:
    """Returns True if *target* already has content and we shouldn't re-download."""
    if force:
        return False
    return target.exists() and any(target.iterdir())


def download_hf(repo_id: str, raw_dir: Path, force: bool) -> str:
    """Downloads a Hugging Face dataset repo snapshot into data/raw/<subdir>."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "SKIP (pip install huggingface_hub)"
    target = raw_dir / HF_DATASETS[repo_id]
    if _skip(target, force):
        return f"skip (exists: {target.name})"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return f"OK → {target.name}"


def download_zip(url: str, subdir: str, raw_dir: Path, force: bool) -> str:
    """Downloads and extracts a zip into data/raw/<subdir>."""
    import requests

    target = raw_dir / subdir
    if _skip(target, force):
        return f"skip (exists: {target.name})"
    target.mkdir(parents=True, exist_ok=True)
    zpath = target / "download.zip"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with zpath.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    try:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
    except zipfile.BadZipFile:
        return f"FAIL (not a zip): {url}"
    return f"OK → {target.name}"


def download_wikipedia(raw_dir: Path, force: bool) -> str:
    """Fetches curated Wikipedia fashion pages as plain-text files."""
    import requests

    target = raw_dir / "wikipedia"
    target.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "FashionGraph/0.1 (research; contact via github)"}
    got, skipped = 0, 0
    for title in dict.fromkeys(WIKI_TITLES):  # de-dup, keep order
        slug = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
        out = target / f"{slug}.txt"
        if out.exists() and not force:
            skipped += 1
            continue
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "redirects": 1, "format": "json", "titles": title,
                },
                headers=headers, timeout=30,
            )
            resp.raise_for_status()
            pages = resp.json()["query"]["pages"]
            extract = next(iter(pages.values())).get("extract", "")
            if extract and len(extract) > 200:
                out.write_text(extract, encoding="utf-8")
                got += 1
            time.sleep(0.3)  # be polite to the API
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {title}: {exc}")
    return f"OK → wikipedia ({got} fetched, {skipped} already present)"


def run(sets: list[str], raw_dir: Path, force: bool) -> None:
    """Executes the selected download sets, continuing past failures."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {raw_dir}\nSets: {', '.join(sets)}\n")

    for set_name in sets:
        if set_name not in SETS:
            print(f"[{set_name}] unknown set — skipping")
            continue
        if set_name in BIG_SETS:
            print(f"[{set_name}] ⚠ LARGE dataset — this may take a while / lots of disk.")
        for task in SETS[set_name]:
            try:
                if task[0] == "hf":
                    print(f"[{set_name}] HF {task[1]} … {download_hf(task[1], raw_dir, force)}")
                elif task[0] == "zip":
                    print(f"[{set_name}] zip {task[2]} … {download_zip(task[1], task[2], raw_dir, force)}")
                elif task[0] == "wiki":
                    print(f"[{set_name}] wikipedia … {download_wikipedia(raw_dir, force)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{set_name}] {task} FAILED: {exc}")

    print("\nDone. Next: `fgraph data build --source fashion_products,wikipedia,style_instruct`")
    print("      (or: `python -m fg.cli data build --source ...`)")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Download FashionGraph datasets (safe).")
    p.add_argument("--sets", default="core",
                   help="Comma-separated set names (default: core). See --list.")
    p.add_argument("--data-dir", default=str(DEFAULT_RAW),
                   help="Target raw-data dir (default: data/raw).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the folder already exists.")
    p.add_argument("--list", action="store_true", help="List available sets and exit.")
    args = p.parse_args()

    if args.list:
        print("Available sets:")
        for name, tasks in SETS.items():
            big = " (LARGE)" if name in BIG_SETS else ""
            print(f"  {name}{big}: {len(tasks)} task(s)")
        sys.exit(0)

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    run(sets, Path(args.data_dir), args.force)


if __name__ == "__main__":
    main()
