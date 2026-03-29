"""Player embedding model (PyTorch).

Maps player + map + context features into a low-dimensional vector space
where similar players cluster together.
"""

import torch.nn as nn


class PlayerEmbeddingModel(nn.Module):
    """Embedding model that learns player representations from match stats."""

    def __init__(self, n_players, n_maps, n_roles, n_continuous, embed_dim=8):
        super().__init__()
        raise NotImplementedError

    def forward(self, player_idx, map_idx, role_idx, continuous):
        raise NotImplementedError
