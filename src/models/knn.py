"""k-Nearest Neighbors over player embeddings (from scratch, NumPy only).

Retrieves the most historically similar player-map matchups by L2 distance
in embedding space. No sklearn or library KNN calls.
"""

import numpy as np


def l2_distances(query: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Compute L2 distance from query (embed_dim,) to every row of index (N, embed_dim).
    Returns distances shape (N,).
    """
    diff = index - query           # broadcast: (N, D) - (D,)
    return np.sqrt((diff ** 2).sum(axis=1))


class KNNRetrieval:
    """
    From-scratch k-NN using L2 distance over player embedding vectors.

    Usage:
        knn = KNNRetrieval(k=5)
        knn.fit(embeddings_array, metadata_list)
        results = knn.query(query_embedding)
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._embeddings: np.ndarray | None = None
        self._metadata:   list[dict]        = []

    def fit(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Store the embedding index.

        embeddings: shape (N, embed_dim)
        metadata:   list of N dicts, each with keys:
                    player_name, map_name, kills, kpr, adr, role, cluster (optional)
        """
        assert len(embeddings) == len(metadata), "embeddings and metadata must have same length"
        self._embeddings = embeddings.astype(np.float32)
        self._metadata   = metadata

    def query(self, embedding: np.ndarray, exclude_player: str | None = None) -> list[dict]:
        """
        Find the k most similar player-map observations.

        embedding:      shape (embed_dim,)
        exclude_player: optionally skip results from the same player (avoids self-match)

        Returns list of k dicts, each with:
            player_name, map_name, distance, kills, kpr, adr, role, cluster
        """
        if self._embeddings is None:
            raise RuntimeError("Call fit() before query()")

        distances = l2_distances(embedding.astype(np.float32), self._embeddings)

        # Sort ascending; take top k*3 then filter if needed
        ranked = np.argsort(distances)
        results = []
        for idx in ranked:
            meta = self._metadata[idx]
            if exclude_player and meta.get("player_name") == exclude_player:
                continue
            results.append({**meta, "distance": float(distances[idx])})
            if len(results) == self.k:
                break

        return results

    @classmethod
    def from_npz(cls, embeddings_path: str, df, k: int = 5) -> "KNNRetrieval":
        """
        Convenience constructor: load embeddings.npz and build metadata from df.

        df must be the full (train + test) cleaned DataFrame from loader.py.
        """
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings   = data["embeddings"]
        player_names = data["player_names"]
        map_names    = data["map_names"]

        # Build per-row median kills for each player-map from df
        med = (
            df.groupby(["player_name", "map_name"])["kills"]
            .median()
            .to_dict()
        )

        metadata = []
        for pname, mname in zip(player_names, map_names):
            row = df[(df["player_name"] == pname) & (df["map_name"] == mname)]
            metadata.append({
                "player_name": pname,
                "map_name":    mname,
                "kills":       float(med.get((pname, mname), np.nan)),
                "kpr":         float(row["kpr"].mean()) if len(row) else float("nan"),
                "adr":         float(row["adr"].mean()) if len(row) else float("nan"),
                "role":        row["role"].iloc[0] if len(row) else "unknown",
                "cluster":     int(row["cluster"].iloc[0]) if "cluster" in row.columns and len(row) else -1,
            })

        inst = cls(k=k)
        inst.fit(embeddings, metadata)
        return inst