"""CLI to train the full pipeline: features -> embeddings -> quantile regression."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train kill prediction pipeline")
    parser.add_argument("--data", required=True, help="Path to player_map_stats.csv")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", default="trained_models/")
    args = parser.parse_args()

    raise NotImplementedError("Training pipeline not yet implemented")


if __name__ == "__main__":
    main()
