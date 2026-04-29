"""
Backtesting: compare MLP kill predictions vs a naive historical frequency baseline.

Baseline: for each player-map in the test set, predict over/under based purely
on how often that player historically went over the kill line in training data.

Model: use MLP predicted 50th percentile vs the kill line to determine over/under.

Metric: accuracy — % of test observations correctly predicted over/under.

Usage:
    python src/evaluation/backtest.py
    python src/evaluation/backtest.py --kill-line 15.5
    python src/evaluation/backtest.py --features data/features.npz --mlp data/mlp_model.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.mlp import MLPQuantileRegressor


def load_data(features_path: str, csv_path: str):
    """Load test features and raw CSV for baseline computation."""
    feat = np.load(features_path, allow_pickle=True)
    X_test = feat["X_test"]
    y_test  = feat["y_test"]
    X_train = feat["X_train"]
    y_train = feat["y_train"]
    df = pd.read_csv(csv_path)
    return X_train, y_train, X_test, y_test, df


def historical_frequency_baseline(
    df: pd.DataFrame,
    kill_line: float,
    split_boundary_path: str = "data/split_boundary.json",
) -> dict:
    """
    For each player, compute how often they historically went OVER the kill line
    in training data. Returns a dict mapping player_name -> P(kills > kill_line).
    """
    import json

    # Load split boundary to separate train/test rows
    if os.path.exists(split_boundary_path):
        with open(split_boundary_path) as f:
            boundary = json.load(f)
        boundary_date = boundary.get("split_date") or boundary.get("boundary_date")
        if boundary_date and "date" in df.columns:
            train_df = df[df["date"] < boundary_date]
        else:
            n = int(len(df) * 0.8)
            train_df = df.iloc[:n]
    else:
        n = int(len(df) * 0.8)
        train_df = df.iloc[:n]

    baseline = {}
    for player, grp in train_df.groupby("player_name"):
        n_over  = (grp["kills"] > kill_line).sum()
        n_total = len(grp)
        baseline[player] = n_over / n_total if n_total > 0 else 0.5

    return baseline


def run_backtest(
    features_path: str = "data/features.npz",
    mlp_path: str      = "data/mlp_model.npz",
    csv_path: str      = "data/player_map_stats.csv",
    kill_line: float   = 15.5,
):
    print(f"\n{'='*55}")
    print(f"  BACKTEST  —  kill line: {kill_line}")
    print(f"{'='*55}\n")

    # ── Load data ──────────────────────────────────────────────
    X_train, y_train, X_test, y_test, df = load_data(features_path, csv_path)
    print(f"Test set:  {len(X_test):,} observations")
    print(f"Train set: {len(X_train):,} observations\n")

    # ── MLP predictions ────────────────────────────────────────
    mlp = MLPQuantileRegressor.load(mlp_path)
    preds = mlp.predict(X_test)          # shape (N, 5) — quantiles [.10,.25,.50,.75,.90]
    median_pred = preds[:, 2]            # 50th percentile as point estimate
    mlp_over    = (median_pred > kill_line).astype(int)
    actual_over = (y_test > kill_line).astype(int)

    mlp_acc = (mlp_over == actual_over).mean()

    # ── Naive baseline ─────────────────────────────────────────
    # Match test rows back to player names using split boundary
    import json
    split_path = "data/split_boundary.json"
    if os.path.exists(split_path):
        with open(split_path) as f:
            boundary = json.load(f)
        boundary_date = boundary.get("split_date") or boundary.get("boundary_date")
        if boundary_date and "date" in df.columns:
            test_df = df[df["date"] >= boundary_date].reset_index(drop=True)
        else:
            test_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
    else:
        test_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

    # Align lengths
    n = min(len(X_test), len(test_df))
    test_df     = test_df.iloc[:n]
    actual_over = actual_over[:n]
    mlp_over    = mlp_over[:n]

    freq_map    = historical_frequency_baseline(df, kill_line, split_path)
    baseline_prob = test_df["player_name"].map(
        lambda p: freq_map.get(p, 0.5)
    ).values
    baseline_over = (baseline_prob >= 0.5).astype(int)
    baseline_acc  = (baseline_over == actual_over).mean()

    # ── Results ────────────────────────────────────────────────
    print(f"  {'Method':<35} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Naive historical frequency baseline':<35} {baseline_acc*100:>9.1f}%")
    print(f"  {'MLP (50th percentile)':<35} {mlp_acc*100:>9.1f}%")
    print(f"  {'-'*45}")
    improvement = (mlp_acc - baseline_acc) * 100
    sign = "+" if improvement >= 0 else ""
    print(f"  {'MLP improvement over baseline':<35} {sign}{improvement:>8.1f}%")

    # ── Over/under breakdown ───────────────────────────────────
    n_over_total  = actual_over.sum()
    n_under_total = len(actual_over) - n_over_total
    print(f"\n  Test set breakdown at line {kill_line}:")
    print(f"    Over:  {n_over_total:,} ({n_over_total/len(actual_over)*100:.1f}%)")
    print(f"    Under: {n_under_total:,} ({n_under_total/len(actual_over)*100:.1f}%)")

    # ── Pinball loss comparison ────────────────────────────────
    quantiles = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    preds_n   = mlp.predict(X_test[:n])
    y_n       = y_test[:n]
    y_col     = y_n.reshape(-1, 1)
    err       = y_col - preds_n
    q         = quantiles.reshape(1, -1)
    loss_per_q = np.mean(np.maximum(q * err, (q - 1) * err), axis=0)

    print(f"\n  MLP pinball loss per quantile (test set):")
    labels = ["10th", "25th", "50th", "75th", "90th"]
    for label, loss in zip(labels, loss_per_q):
        print(f"    {label}: {loss:.4f}")

    print(f"\n{'='*55}\n")

    return {
        "baseline_accuracy": float(baseline_acc),
        "mlp_accuracy":      float(mlp_acc),
        "improvement":       float(improvement),
        "pinball_per_quantile": loss_per_q.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest MLP vs naive baseline")
    parser.add_argument("--features",  default="data/features.npz")
    parser.add_argument("--mlp",       default="data/mlp_model.npz")
    parser.add_argument("--csv",       default="data/player_map_stats.csv")
    parser.add_argument("--kill-line", type=float, default=15.5,
                        help="Kill line threshold for over/under (default: 15.5)")
    args = parser.parse_args()

    run_backtest(
        features_path=args.features,
        mlp_path=args.mlp,
        csv_path=args.csv,
        kill_line=args.kill_line,
    )