import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class InteractionLearning(nn.Module):
    """IL модуль: GRU на соседних сезонах — ловит короткие циклы."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x_seq):
        # x_seq: (batch, time_steps, features)
        out, _ = self.gru(x_seq)
        return out


class GlobalTimeAttention(nn.Module):
    """GTAM модуль: Transformer Attention по всему ряду — ловит циклы ~20 лет."""
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class TemporalFashionGNN(nn.Module):
    """
    Архитектура по образцу GNNctd.
    Узлы = fashion элементы (wide-leg jeans, chunky boots...).
    Рёбра = "носились вместе в сезон X".
    Выход: trend_score (0-1) для каждого элемента в следующем сезоне.
    """
    def __init__(self, num_nodes, hidden_dim=64, num_seasons=32):
        super().__init__()
        self.gcn = GCNConv(num_nodes, hidden_dim)
        self.il  = InteractionLearning(hidden_dim, hidden_dim)
        self.gtam = GlobalTimeAttention(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # trend_score в [0, 1]
        )

    def forward(self, snapshots, edge_index):
        """
        snapshots: list of tensors (num_nodes,) — trend scores по сезонам
        edge_index: (2, num_edges) — граф совместимости вещей
        """
        gcn_outs = []
        for snap in snapshots:
            x = snap.unsqueeze(-1).expand(-1, snap.shape[0])
            gcn_outs.append(self.gcn(x, edge_index))

        seq = torch.stack(gcn_outs, dim=1)  # (nodes, time, hidden)
        il_out   = self.il(seq)
        gtam_out = self.gtam(seq)

        # Fusion: IL (короткий) + GTAM (длинный)
        combined = torch.cat([il_out[:, -1], gtam_out[:, -1]], dim=-1)
        return self.predictor(combined).squeeze(-1)
