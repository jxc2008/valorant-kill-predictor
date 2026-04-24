"""PCA projection and scatter plots of player embedding space."""

import os
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def standardize_embeddings(embeddings):
    """Z-score each embedding dimension before visualization."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (N, D)")

    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (embeddings - mean) / std


def pca_project(embeddings, n_components=2, return_variance=False):
    """Project high-dimensional embeddings to 2D/3D via PCA."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array with shape (N, D)")
    if not 1 <= n_components <= embeddings.shape[1]:
        raise ValueError("n_components must be between 1 and embedding dimension")

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    projected = centered @ components.T

    if not return_variance:
        return projected

    variances = (singular_values ** 2) / max(embeddings.shape[0] - 1, 1)
    explained_ratio = variances[:n_components] / variances.sum()
    return projected, explained_ratio


def plot_embedding_space(
    projected,
    labels,
    roles=None,
    out_path="data/cluster_pca.png",
    title="Embedding PCA by cluster",
):
    """Scatter plot of player embeddings colored by role/archetype."""
    projected = np.asarray(projected, dtype=np.float32)
    labels = np.asarray(labels)
    if projected.ndim != 2 or projected.shape[1] < 2:
        raise ValueError("projected must have shape (N, >=2)")
    if len(projected) != len(labels):
        raise ValueError("projected and labels must have the same number of rows")
    if roles is not None and len(roles) != len(labels):
        raise ValueError("roles and labels must have the same number of rows")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=7,
            alpha=0.55,
            color=cmap(i % 10),
            label=f"cluster {label}",
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2.5, frameon=False)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path)

    return fig, ax
