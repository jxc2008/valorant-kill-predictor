"""CLI for training project stages.

Currently implemented:
- mlp: train Ian's NumPy MLP quantile model from data/features.npz
"""

import argparse

from src.models.mlp import train_from_features


ALL_STAGES = {"features", "embeddings", "clustering", "quantile", "mlp", "evaluate"}
IMPLEMENTED_STAGES = {"mlp"}


def main():
    parser = argparse.ArgumentParser(description="Train kill prediction pipeline")
    parser.add_argument("--stage", required=True, choices=sorted(ALL_STAGES))
    parser.add_argument("--features", default="data/features.npz", help="Path to features.npz")
    parser.add_argument("--output", default="data/mlp_model.npz", help="Output model path")
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument(
        "--no-extra-cluster-keys",
        action="store_true",
        help=(
            "Disable auto-appending cluster arrays from features.npz keys "
            "('X_cluster_train' or 'cluster_features_train')."
        ),
    )
    parser.add_argument(
        "--require-cluster-features",
        action="store_true",
        help=(
            "Fail fast if no clustering features are detected "
            "(neither cluster_* in feature_names nor separate cluster arrays)."
        ),
    )
    args = parser.parse_args()

    if args.stage != "mlp":
        raise NotImplementedError(
            f"Stage '{args.stage}' is not implemented yet. Implemented stages: {sorted(IMPLEMENTED_STAGES)}"
        )

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


if __name__ == "__main__":
    main()
