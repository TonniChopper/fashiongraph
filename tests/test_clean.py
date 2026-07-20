"""Tests for text cleaning, quality gating, and dedup."""

from fg.data.clean import (
    clean_documents,
    clean_text,
    content_hash,
    dedup_documents,
    is_low_quality,
    normalize_whitespace,
    strip_boilerplate,
)
from fg.data.schema import Document


def test_normalize_whitespace_collapses():
    assert normalize_whitespace("a   b\t\tc") == "a b c"


def test_normalize_whitespace_limits_blank_lines():
    assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"


def test_strip_boilerplate_removes_nav():
    text = "Great trench coats.\nAccept cookies\nSubscribe\nWool is warm."
    out = strip_boilerplate(text)
    assert "cookies" not in out.lower()
    assert "Subscribe" not in out
    assert "trench" in out


def test_content_hash_is_case_and_space_insensitive():
    assert content_hash("Wide-leg  JEANS") == content_hash("wide-leg jeans")


def test_content_hash_differs_for_different_text():
    assert content_hash("peplum top") != content_hash("barrel-leg denim")


def test_is_low_quality_short():
    assert is_low_quality("too short") is True
    assert is_low_quality("word " * 40) is False


def test_dedup_documents_keeps_first():
    docs = [
        Document("Quiet luxury is understated.", {"source": "a"}),
        Document("quiet luxury is understated.", {"source": "b"}),  # dup
        Document("Maximalism is bold.", {"source": "c"}),
    ]
    out = dedup_documents(docs)
    assert len(out) == 2
    assert out[0].metadata["source"] == "a"


def test_clean_documents_end_to_end():
    docs = [
        Document("  Tailoring   returns.\nAccept cookies\n" + "detail " * 30, {"source": "x"}),
        Document("junk", {"source": "y"}),  # too short → dropped
        Document("Tailoring returns. " + "detail " * 30, {"source": "z"}),  # dup of first
    ]
    out = clean_documents(docs)
    assert len(out) == 1
    assert "cookies" not in out[0].text.lower()


def test_clean_text_empty():
    assert clean_text("") == ""
