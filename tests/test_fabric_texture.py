"""Tests for fabric-texture linking (mirror nodes) — synthetic index."""

import numpy as np

from fg.vision.fabric_texture import FabricTextureLinker
from fg.vision.index import VisualIndex


def _texture_index(tmp_path):
    # Two silk swatches (axis 0), two wool swatches (axis 1).
    emb = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]], dtype=np.float32)
    meta = [{"fabric": "silk"}, {"fabric": "silk"}, {"fabric": "wool"}, {"fabric": "wool"}]
    return VisualIndex(emb, meta).save(tmp_path / "tex.npz")


def test_identify_fabric(tmp_path):
    linker = FabricTextureLinker(_texture_index(tmp_path))
    assert linker.identify(np.array([1.0, 0.0]))[0][0] == "silk"
    assert linker.identify(np.array([0.0, 1.0]))[0][0] == "wool"


def test_centroids_are_per_fabric_and_normalised(tmp_path):
    cents = FabricTextureLinker(_texture_index(tmp_path)).centroids()
    assert set(cents) == {"silk", "wool"}
    for v in cents.values():
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    # silk centroid points along axis 0.
    assert cents["silk"][0] > cents["silk"][1]
