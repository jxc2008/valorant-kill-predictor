"""Player embedding model (PyTorch).

Learns an 8-dim fingerprint per player-map observation by training to
predict kills. Similar playstyles end up close in embedding space.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PlayerEmbeddingModel(nn.Module):
    """
    Architecture:
        player_idx -> Embedding(n_players, 16)
        map_idx    -> Embedding(n_maps,    8)
        role_idx   -> Embedding(n_roles,   4)
        continuous -> Linear(n_continuous, 16) -> ReLU

        concat(44) -> Linear(44, 32) -> ReLU -> Linear(32, embed_dim)
                                                        ^--- the embedding
        embedding  -> Linear(embed_dim, 1)              kill prediction head
    """

    def __init__(self, n_players: int, n_maps: int, n_roles: int = 4,
                 n_continuous: int = 7, embed_dim: int = 8):
        super().__init__()
        self.player_emb = nn.Embedding(n_players, 16)
        self.map_emb    = nn.Embedding(n_maps,    8)
        self.role_emb   = nn.Embedding(n_roles,   4)
        self.cont_proj  = nn.Sequential(nn.Linear(n_continuous, 16), nn.ReLU())

        fused_dim = 16 + 8 + 4 + 16   # = 44
        self.backbone = nn.Sequential(
            nn.Linear(fused_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim),
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, player_idx, map_idx, role_idx, continuous):
        p = self.player_emb(player_idx)
        m = self.map_emb(map_idx)
        r = self.role_emb(role_idx)
        c = self.cont_proj(continuous)
        fused = torch.cat([p, m, r, c], dim=1)
        embedding = self.backbone(fused)
        kills_pred = self.head(embedding).squeeze(1)
        return kills_pred, embedding


def build_index_maps(df) -> tuple[dict, dict]:
    """Build player_name -> int and map_name -> int lookup dicts."""
    players = {name: i for i, name in enumerate(df["player_name"].unique())}
    maps    = {name: i for i, name in enumerate(df["map_name"].unique())}
    return players, maps


def train_embeddings(
    train_df,
    continuous_cols: list[str],
    embed_dim: int = 8,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple["PlayerEmbeddingModel", dict, dict]:
    """
    Train the embedding model and return (model, player_index, map_index).

    continuous_cols: the 7 continuous feature column names (already normalized).
    """
    player_idx_map, map_idx_map = build_index_maps(train_df)

    p_idx = torch.tensor(train_df["player_name"].map(player_idx_map).values, dtype=torch.long)
    m_idx = torch.tensor(train_df["map_name"].map(map_idx_map).values,       dtype=torch.long)
    r_idx = torch.tensor(train_df["role_idx"].values,                         dtype=torch.long)
    cont  = torch.tensor(train_df[continuous_cols].values,                    dtype=torch.float32)
    y     = torch.tensor(train_df["kills"].values,                            dtype=torch.float32)

    dataset = TensorDataset(p_idx, m_idx, r_idx, cont, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PlayerEmbeddingModel(
        n_players=len(player_idx_map),
        n_maps=len(map_idx_map),
        n_continuous=len(continuous_cols),
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for p_b, m_b, r_b, c_b, y_b in loader:
            p_b, m_b, r_b, c_b, y_b = (t.to(device) for t in (p_b, m_b, r_b, c_b, y_b))
            optimizer.zero_grad()
            pred, _ = model(p_b, m_b, r_b, c_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return model, player_idx_map, map_idx_map


def extract_embeddings(
    model: "PlayerEmbeddingModel",
    df,
    player_idx_map: dict,
    map_idx_map: dict,
    continuous_cols: list[str],
    device: str = "cpu",
    out_path: str = "data/embeddings.npz",
) -> np.ndarray:
    """
    Run all rows through the trained model and extract the 8-dim embedding.
    Saves embeddings.npz and returns the embedding array (N, embed_dim).
    """
    # Unknown players/maps in df default to index 0 (rare edge case)
    p_idx = torch.tensor(
        df["player_name"].map(lambda x: player_idx_map.get(x, 0)).values, dtype=torch.long)
    m_idx = torch.tensor(
        df["map_name"].map(lambda x: map_idx_map.get(x, 0)).values, dtype=torch.long)
    r_idx = torch.tensor(df["role_idx"].values, dtype=torch.long)
    cont  = torch.tensor(df[continuous_cols].values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        _, embeddings = model(
            p_idx.to(device), m_idx.to(device), r_idx.to(device), cont.to(device)
        )
    emb_np = embeddings.cpu().numpy()

    np.savez(out_path,
        embeddings=emb_np,
        player_names=df["player_name"].values,
        map_names=df["map_name"].values,
    )
    print(f"Saved embeddings -> {out_path}  shape={emb_np.shape}")
    return emb_np
