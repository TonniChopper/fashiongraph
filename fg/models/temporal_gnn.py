"""Temporal Graph Neural Network for fashion trend forecasting."""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

logger: logging.Logger = logging.getLogger(__name__)


class InteractionLearning(nn.Module):
    """Interaction-Learning module using a GRU on adjacent seasons.

    Captures short-range temporal cycles (e.g. seasonal recurrences).

    Attributes:
        gru: Gated Recurrent Unit layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the InteractionLearning module.

        Args:
            input_dim: Dimensionality of input features per time step.
            hidden_dim: Dimensionality of the GRU hidden state.

        Raises:
            ValueError: If ``input_dim`` or ``hidden_dim`` is not positive.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.gru: nn.GRU = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Runs the GRU over a temporal sequence.

        Args:
            x_seq: Input tensor of shape ``(batch, time_steps, features)``.

        Returns:
            GRU output of shape ``(batch, time_steps, hidden_dim)``.
        """
        output, _ = self.gru(x_seq)
        return output


class GlobalTimeAttention(nn.Module):
    """Global Temporal Attention Module (GTAM).

    Applies multi-head self-attention across the full temporal sequence to
    capture long-range cycles (~20 years in fashion).

    Attributes:
        attention: Multi-head self-attention layer.
        layer_norm: Layer normalisation applied after the residual connection.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        """Initializes the GlobalTimeAttention module.

        Args:
            hidden_dim: Dimensionality of input/output features.
            num_heads: Number of attention heads.

        Raises:
            ValueError: If ``hidden_dim`` is not positive or not divisible by
                ``num_heads``.
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention with a residual connection and layer norm.

        Args:
            x: Input tensor of shape ``(batch, time_steps, hidden_dim)``.

        Returns:
            Output tensor of the same shape as *x*.
        """
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)


class TemporalFashionGNN(nn.Module):
    """Temporal Fashion GNN following the GNNctd architecture.

    Nodes represent fashion elements (e.g. wide-leg jeans, chunky boots).
    Edges represent co-occurrence within a given season.
    The model predicts a ``trend_score`` in ``[0, 1]`` for each element in the
    next season.

    Attributes:
        num_nodes: Number of fashion element nodes in the graph.
        node_embed: Linear projection from scalar score to hidden_dim.
        gcn: Graph convolution layer.
        il: Short-range temporal GRU module (InteractionLearning).
        gtam: Long-range temporal attention module (GlobalTimeAttention).
        predictor: MLP that produces per-node trend scores.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_seasons: int = 32,
    ) -> None:
        """Initializes the TemporalFashionGNN.

        Args:
            num_nodes: Number of fashion element nodes in the graph.
            hidden_dim: Hidden dimensionality used throughout the model.
            num_seasons: Total number of seasonal snapshots (reserved for
                future use).

        Raises:
            ValueError: If ``num_nodes`` or ``hidden_dim`` is not positive.
        """
        super().__init__()
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.num_nodes: int = num_nodes
        self.node_embed: nn.Linear = nn.Linear(1, hidden_dim)
        self.gcn: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.il: InteractionLearning = InteractionLearning(hidden_dim, hidden_dim)
        self.gtam: GlobalTimeAttention = GlobalTimeAttention(hidden_dim)
        self.predictor: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        logger.info(
            "TemporalFashionGNN initialised: num_nodes=%d, hidden_dim=%d",
            num_nodes,
            hidden_dim,
        )

    def forward(
        self,
        snapshots: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the forward pass over seasonal graph snapshots.

        Args:
            snapshots: List of tensors, each of shape ``(num_nodes,)``,
                representing per-element trend scores for each season.
            edge_index: Edge index of shape ``(2, num_edges)`` describing the
                element compatibility graph.

        Returns:
            Predicted trend scores of shape ``(num_nodes,)`` for the next
            season, with values in ``[0, 1]``.

        Raises:
            ValueError: If *snapshots* is empty or tensors have an unexpected
                shape.
        """
        if not snapshots:
            raise ValueError("snapshots must be a non-empty list of tensors.")

        for i, snap in enumerate(snapshots):
            if snap.dim() != 1 or snap.shape[0] != self.num_nodes:
                raise ValueError(
                    f"Snapshot {i} has shape {tuple(snap.shape)}; "
                    f"expected ({self.num_nodes},)."
                )

        gcn_outputs: list[torch.Tensor] = []
        for snap in snapshots:
            x: torch.Tensor = self.node_embed(snap.unsqueeze(-1))
            gcn_outputs.append(self.gcn(x, edge_index))

        seq: torch.Tensor = torch.stack(gcn_outputs, dim=1)
        il_out: torch.Tensor = self.il(seq)
        gtam_out: torch.Tensor = self.gtam(seq)
        combined: torch.Tensor = torch.cat([il_out[:, -1], gtam_out[:, -1]], dim=-1)
        return self.predictor(combined).squeeze(-1)

