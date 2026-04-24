"""K-means clustering for player role archetypes (from scratch, NumPy only).

Groups players into role archetypes (e.g. entry fragger, sentinel, IGL-style)
based on their embedding vectors. Cluster labels serve as additional features
for the quantile regression model.
"""

import numpy as np


class KMeansClustering:
    """From-scratch k-means clustering over player embeddings."""

    def __init__(self, n_clusters=4, max_iter=100, random_state=42, tol=1e-4):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _validate_embeddings(self, embeddings, require_enough_rows=False):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array with shape (N, D)")
        if require_enough_rows and embeddings.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot be greater than number of rows")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contains NaN or infinite values")
        return embeddings

    def _assign_labels(self, embeddings):
        diff = embeddings[:, None, :] - self.centroids[None, :, :]
        distances_sq = np.sum(diff * diff, axis=2)
        return np.argmin(distances_sq, axis=1), distances_sq

    def fit(self, embeddings):
        """Fit k-means centroids to player embeddings."""
        embeddings = self._validate_embeddings(embeddings, require_enough_rows=True)
        rng = np.random.default_rng(self.random_state)

        initial_idx = rng.choice(embeddings.shape[0], size=self.n_clusters, replace=False)
        self.centroids = embeddings[initial_idx].copy()

        for iteration in range(1, self.max_iter + 1):
            labels, distances_sq = self._assign_labels(embeddings)
            new_centroids = self.centroids.copy()

            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = embeddings[mask].mean(axis=0)
                else:
                    # Re-seed empty clusters with a real point so prediction stays defined.
                    new_centroids[cluster_id] = embeddings[rng.integers(embeddings.shape[0])]

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iter_ = iteration

            if shift <= self.tol:
                break

        self.labels_, distances_sq = self._assign_labels(embeddings)
        self.inertia_ = float(np.sum(distances_sq[np.arange(embeddings.shape[0]), self.labels_]))
        return self

    def predict(self, embeddings):
        """Assign cluster (role archetype) labels to player embeddings."""
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict()")
        embeddings = self._validate_embeddings(embeddings)
        if embeddings.shape[1] != self.centroids.shape[1]:
            raise ValueError("embedding dimension does not match fitted centroids")
        labels, _ = self._assign_labels(embeddings)
        return labels.astype(np.int32)

    def fit_predict(self, embeddings):
        """Fit and return cluster labels in one step."""
        self.fit(embeddings)
        return self.labels_.astype(np.int32)
