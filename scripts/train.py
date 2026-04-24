"""
Train the full pipeline: features -> embeddings -> clustering -> features2 -> mlp/quantile.

Stages (run in order):
    features    — build and save features.npz from the raw CSV
    embeddings  — train embedding model, save embeddings.npz
    features2   — re-run features with cluster one-hots appended (after clustering)
    mlp         — train Ian's NumPy MLP quantile model from data/features.npz

Usage:
    python scripts/train.py --stage features
    python scripts/train.py --stage embeddings --epochs 100
    python scripts/train.py --stage features2
    python scripts/train.py --stage mlp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_player_stats
from src.data.split  import temporal_split, save_split_boundary
from src.data.features import build_and_save, CONTINUOUS_COLS


def stage_features(args):
    print("=== Stage: features ===")
    df = load_player_stats(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")
    train_df, test_df = temporal_split(df, train_ratio=0.8)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    save_split_boundary(test_df)
    build_and_save(train_df, test_df)


def stage_embeddings(args):
    print("=== Stage: embeddings ===")
    import numpy as np
    from src.models.embeddings import train_embeddings, extract_embeddings

    df = load_player_stats(args.data)
    train_df, _ = temporal_split(df, train_ratio=0.8)

    data = np.load("data/features.npz", allow_pickle=True)
    scaler = np.load("data/scaler_params.npz")

    train_df = train_df.copy()
    for i, col in enumerate(CONTINUOUS_COLS):
        train_df[col] = (train_df[col] - scaler["mean"][i]) / scaler["std"][i]

    print(f"Training embedding model on {len(train_df)} observations...")
    model, player_idx_map, map_idx_map = train_embeddings(
        train_df,
        continuous_cols=CONTINUOUS_COLS,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
    )

    all_df = load_player_stats(args.data)
    scaler = np.load("data/scaler_params.npz")
    for i, col in enumerate(CONTINUOUS_COLS):
        all_df[col] = (all_df[col] - scaler["mean"][i]) / scaler["std"][i]

    extract_embeddings(model, all_df, player_idx_map, map_idx_map, CONTINUOUS_COLS)
    print("Done.")


def stage_features2(args):
    """Re-run feature building with cluster one-hots appended (after clustering)."""
    print("=== Stage: features2 (with cluster labels) ===")
    import numpy as np

    if not os.path.exists("data/cluster_labels.npz"):
        print("ERROR: data/cluster_labels.npz not found. Run clustering first (Shengyang's step).")
        sys.exit(1)

    cluster_data = np.load("data/cluster_labels.npz", allow_pickle=True)
    labels = cluster_data["labels"]
    k = int(cluster_data["k"])

    emb_data = np.load("data/embeddings.npz", allow_pickle=True)
    player_names = emb_data["player_names"]
    map_names    = emb_data["map_names"]

    df = load_player_stats(args.data)
    label_lookup = {}
    for pname, mname, label in zip(player_names, map_names, labels):
        label_lookup[(pname, mname)] = int(label)

    df["cluster"] = df.apply(
        lambda r: label_lookup.get((r["player_name"], r["map_name"]), 0), axis=1
    )

    for c in range(k):
        df[f"cluster_{c}"] = (df["cluster"] == c).astype(float)

    train_df, test_df = temporal_split(df, train_ratio=0.8)
    build_and_save(train_df, test_df,
                   out_path="data/features.npz",
                   scaler_path="data/scaler_params.npz")
    print("features.npz updated with cluster columns.")


def stage_mlp(args):
    from src.models.mlp import train_from_features
    train_from_features(
        features_path=args.features,
        out_path=args.output,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=(args.hidden1, args.hidden2),
        include_extra_cluster_keys=not args.no_extra_cluster_keys,
        require_cluster_features=args.require_cluster_features,
    )
    print(f"Saved MLP model to {args.output}")


ALL_STAGES = ["features", "embeddings", "features2", "mlp"]


def main():
    parser = argparse.ArgumentParser(description="Valorant kill predictor training pipeline")
    parser.add_argument("--stage", required=True, choices=ALL_STAGES, help="Pipeline stage to run")
    parser.add_argument("--data",       default="data/player_map_stats.csv")
    parser.add_argument("--embed-dim",  type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=600)
    parser.add_argument("--features",   default="data/features.npz")
    parser.add_argument("--output",     default="data/mlp_model.npz")
    parser.add_argument("--lr",         type=float, default=0.015)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden1",    type=int, default=64)
    parser.add_argument("--hidden2",    type=int, default=32)
    parser.add_argument("--no-extra-cluster-keys",    action="store_true")
    parser.add_argument("--require-cluster-features", action="store_true")
    args = parser.parse_args()

    if args.stage == "features":
        stage_features(args)
    elif args.stage == "embeddings":
        stage_embeddings(args)
    elif args.stage == "features2":
        stage_features2(args)
    elif args.stage == "mlp":
        stage_mlp(args)


if __name__ == "__main__":
    main()
