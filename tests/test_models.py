"""Model tests that require torch. Skipped cleanly if torch is absent
(heavy training deps live on Colab/Kaggle, not the dev sandbox)."""

import pytest

torch = pytest.importorskip("torch")


def test_contrastive_loss_shapes_and_positivity():
    from fg.models.clip_encoder import FashionContrastiveLoss

    loss_fn = FashionContrastiveLoss(temperature=0.07)
    emb = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    loss = loss_fn(emb, emb.clone())
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_contrastive_loss_perfect_alignment_is_low():
    """Identical, well-separated embeddings → near-zero loss."""
    from fg.models.clip_encoder import FashionContrastiveLoss

    loss_fn = FashionContrastiveLoss(temperature=0.01)
    emb = torch.nn.functional.normalize(torch.eye(8), dim=-1)
    assert loss_fn(emb, emb).item() < 0.5


@pytest.mark.skipif(
    pytest.importorskip("torch_geometric", reason="pyg not installed") is None,
    reason="pyg not installed",
)
def test_temporal_gnn_output_shape():
    from fg.models.temporal_gnn import TemporalFashionGNN

    n_nodes = 5
    gnn = TemporalFashionGNN(num_nodes=n_nodes, hidden_dim=8)
    snapshots = [torch.rand(n_nodes) for _ in range(3)]
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = gnn(snapshots, edge_index)
    assert out.shape == (n_nodes,)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid output
