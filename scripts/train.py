"""
Train the full pipeline: features -> embeddings -> (clustering -> features again).

Stages (run in order):
    features    — build and save features.npz from the raw CSV
    embeddings  — train embedding model, save embeddings.npz
    clustering  — (run by Shengyang) reads embeddings.npz, writes cluster_labels.npz
    features2   — re-run features with cluster one-hots appended (after clustering)

Usage:
    python scripts/train.py --stage features
    python scripts/train.py --stage embeddings --epochs 100
    python scripts/train.py --stage features2   # after Shengyang produces cluster_labels.npz
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
    from src.data.loader import load_player_stats
    from src.data.split import temporal_split

    df = load_player_stats(args.data)
    train_df, _ = temporal_split(df, train_ratio=0.8)

    # Load normalized continuous features from features.npz for training
    data = np.load("data/features.npz", allow_pickle=True)
    scaler = np.load("data/scaler_params.npz")

    # Re-attach normalized continuous columns to train_df
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

    # Extract embeddings for ALL rows (train + test) for downstream use
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
    from src.data.loader import load_player_stats
    from src.data.split import temporal_split

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
    # Map cluster labels back onto df rows by (player_name, map_name) first occurrence
    label_lookup = {}
    for pname, mname, label in zip(player_names, map_names, labels):
        label_lookup[(pname, mname)] = int(label)

    df["cluster"] = df.apply(
        lambda r: label_lookup.get((r["player_name"], r["map_name"]), 0), axis=1
    )

    # One-hot encode cluster
    for c in range(k):
        df[f"cluster_{c}"] = (df["cluster"] == c).astype(float)

    train_df, test_df = temporal_split(df, train_ratio=0.8)
    build_and_save(train_df, test_df,
                   out_path="data/features.npz",
                   scaler_path="data/scaler_params.npz")
    print("features.npz updated with cluster columns.")


def main():
    parser = argparse.ArgumentParser(description="Valorant kill predictor training pipeline")
    parser.add_argument("--stage",  required=True,
                        choices=["features", "embeddings", "features2"],
                        help="Pipeline stage to run")
    parser.add_argument("--data",      default="data/player_map_stats.csv")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--epochs",    type=int, default=50)
    args = parser.parse_args()

    if args.stage == "features":
        stage_features(args)
    elif args.stage == "embeddings":
        stage_embeddings(args)
    elif args.stage == "features2":
        stage_features2(args)


if __name__ == "__main__":
    main()
