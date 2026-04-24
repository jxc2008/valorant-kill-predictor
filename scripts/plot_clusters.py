"""Generate a PCA scatter plot of embedding clusters."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.embedding_viz import (
    pca_project,
    plot_embedding_space,
    standardize_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Plot KMeans clusters in PCA space")
    parser.add_argument("--embeddings", default="data/embeddings.npz")
    parser.add_argument("--clusters", default="data/cluster_labels.npz")
    parser.add_argument("--out", default="data/cluster_pca.png")
    parser.add_argument("--title", default="Embedding PCA by cluster")
    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()

    emb_data = np.load(args.embeddings, allow_pickle=True)
    cluster_data = np.load(args.clusters, allow_pickle=True)

    embeddings = emb_data["embeddings"]
    labels = cluster_data["labels"]

    if args.standardize:
        embeddings = standardize_embeddings(embeddings)

    projected, explained = pca_project(embeddings, n_components=2, return_variance=True)
    plot_embedding_space(projected, labels, out_path=args.out, title=args.title)

    counts = np.bincount(labels, minlength=int(cluster_data["k"]))
    print(f"Saved PCA cluster plot -> {args.out}")
    print(f"Standardized embeddings: {args.standardize}")
    print(f"Explained variance ratio: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")
    print(f"Cluster counts: {counts.tolist()}")


if __name__ == "__main__":
    main()
