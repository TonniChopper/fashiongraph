"""Tests for source loaders (the parts that need no heavy deps)."""

import pytest

from fg.data.schema import Document
from fg.data.sources import (
    SOURCES,
    describe_product,
    get_source,
    load_text_files,
)


def test_registry_has_core_sources():
    assert {"fashion_products", "wikipedia", "style_instruct"} <= set(SOURCES)


def test_get_source_unknown_raises():
    with pytest.raises(KeyError):
        get_source("nope")


def test_describe_product_builds_sentence():
    row = {
        "productDisplayName": "Nike Court Sneakers",
        "gender": "Men",
        "baseColour": "White",
        "season": "Summer",
        "year": 2024,
        "usage": "Casual",
    }
    text = describe_product(
        row, "productDisplayName", ["gender", "baseColour", "season", "year", "usage"]
    )
    assert text.startswith("Nike Court Sneakers.")
    assert "baseColour: White" in text
    assert "usage: Casual" in text


def test_describe_product_empty_row():
    assert describe_product({}, None, []) == ""


def test_describe_product_skips_nan():
    row = {"name": "Wool Coat", "baseColour": "nan", "season": "Winter"}
    text = describe_product(row, "name", ["baseColour", "season"])
    assert "nan" not in text.lower()
    assert "season: Winter" in text


def test_load_text_files(tmp_path):
    (tmp_path / "coco-chanel.txt").write_text(
        "Coco Chanel founded the house of Chanel.", encoding="utf-8"
    )
    docs = list(load_text_files(tmp_path))
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["title"] == "coco chanel"
    assert docs[0].metadata["source"] == "wikipedia"


def test_load_text_files_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_text_files(tmp_path / "does-not-exist"))
