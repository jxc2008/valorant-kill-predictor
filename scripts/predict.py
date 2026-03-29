"""CLI to run kill predictions for a player-map-opponent combination."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Predict player kill distribution")
    parser.add_argument("--player", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--opponent", default=None)
    parser.add_argument("--model-dir", default="trained_models/")
    args = parser.parse_args()

    raise NotImplementedError("Prediction pipeline not yet implemented")


if __name__ == "__main__":
    main()
