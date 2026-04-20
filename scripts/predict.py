"""
CLI to run kill predictions and find similar players for a given player-map combo.

Usage:
    python scripts/predict.py --player aspas --map Haven --line 20.5
    python scripts/predict.py --player aspas --map Haven --similar --k 5
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def interpolate_p_over(quantiles: dict, line: float) -> float:
    """
    Estimate P(kills > line) by linear interpolation over the 5 quantile points.
    quantiles: {'q10': v, 'q25': v, 'q50': v, 'q75': v, 'q90': v}
    """
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    vs = [quantiles["q10"], quantiles["q25"], quantiles["q50"],
          quantiles["q75"], quantiles["q90"]]

    if line <= vs[0]:  return 0.90
    if line >= vs[-1]: return 0.10

    for i in range(len(vs) - 1):
        if vs[i] <= line <= vs[i + 1]:
            frac = (line - vs[i]) / (vs[i + 1] - vs[i])
            p_under = qs[i] + frac * (qs[i + 1] - qs[i])
            return round(1.0 - p_under, 4)
    return 0.50


def main():
    parser = argparse.ArgumentParser(description="Valorant kill predictor")
    parser.add_argument("--player",   required=True,       help="Player IGN")
    parser.add_argument("--map",      required=True,       help="Map name")
    parser.add_argument("--line",     type=float,          help="Kill line for over/under")
    parser.add_argument("--model",    default="quantile",  choices=["quantile", "mlp"])
    parser.add_argument("--similar",  action="store_true", help="Show similar players via KNN")
    parser.add_argument("--k",        type=int, default=5, help="Number of similar players")
    parser.add_argument("--data",     default="data/player_map_stats.csv")
    args = parser.parse_args()

    # Load feature matrix
    if not os.path.exists("data/features.npz"):
        print("ERROR: data/features.npz not found. Run: python scripts/train.py --stage features")
        sys.exit(1)

    feat_data = np.load("data/features.npz", allow_pickle=True)
    player_names = feat_data["player_names_train"].tolist() + feat_data["player_names_test"].tolist()
    map_names    = feat_data["map_names_train"].tolist()    + feat_data["map_names_test"].tolist()
    X_all        = np.vstack([feat_data["X_train"], feat_data["X_test"]])

    # Find matching rows
    matches = [
        i for i, (p, m) in enumerate(zip(player_names, map_names))
        if p.lower() == args.player.lower() and m.lower() == args.map.lower()
    ]
    if not matches:
        print(f"No data found for player='{args.player}' map='{args.map}'")
        sys.exit(1)

    X_query = X_all[matches].mean(axis=0, keepdims=True)   # average across appearances

    # Load quantile model
    model_path = f"data/quantile_model.npz" if args.model == "quantile" else "data/mlp_model.npz"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Train the model first.")
        sys.exit(1)

    # Lazy import to avoid hard dependency when model not yet trained
    if args.model == "quantile":
        from src.models.quantile_reg import QuantileRegression
        qr = QuantileRegression()
        qr.load(model_path)
        preds = qr.predict(X_query)[0]   # shape (5,)
    else:
        raise NotImplementedError("MLP predict CLI not yet wired — Ian's model pending")

    quantiles = {
        "q10": round(float(preds[0]), 2),
        "q25": round(float(preds[1]), 2),
        "q50": round(float(preds[2]), 2),
        "q75": round(float(preds[3]), 2),
        "q90": round(float(preds[4]), 2),
    }

    print(f"\n{'='*40}")
    print(f"  {args.player} on {args.map}  [{args.model}]")
    print(f"{'='*40}")
    print(f"  10th pct : {quantiles['q10']} kills")
    print(f"  25th pct : {quantiles['q25']} kills")
    print(f"  Median   : {quantiles['q50']} kills")
    print(f"  75th pct : {quantiles['q75']} kills")
    print(f"  90th pct : {quantiles['q90']} kills")

    if args.line is not None:
        p_over  = interpolate_p_over(quantiles, args.line)
        p_under = round(1.0 - p_over, 4)
        print(f"\n  Line     : {args.line}")
        print(f"  P(over)  : {p_over:.1%}")
        print(f"  P(under) : {p_under:.1%}")

    if args.similar and os.path.exists("data/embeddings.npz"):
        from src.models.knn import KNNRetrieval
        from src.data.loader import load_player_stats

        df  = load_player_stats(args.data)
        knn = KNNRetrieval.from_npz("data/embeddings.npz", df, k=args.k)

        emb_data = np.load("data/embeddings.npz", allow_pickle=True)
        p_arr    = emb_data["player_names"].tolist()
        m_arr    = emb_data["map_names"].tolist()
        embs     = emb_data["embeddings"]

        query_emb_indices = [
            i for i, (p, m) in enumerate(zip(p_arr, m_arr))
            if p.lower() == args.player.lower() and m.lower() == args.map.lower()
        ]
        if query_emb_indices:
            q_emb   = embs[query_emb_indices].mean(axis=0)
            similar = knn.query(q_emb, exclude_player=args.player)
            print(f"\n  Top-{args.k} similar player-map matchups:")
            for i, s in enumerate(similar, 1):
                print(f"  {i}. {s['player_name']} on {s['map_name']}"
                      f"  kills={s['kills']:.1f}  role={s['role']}"
                      f"  dist={s['distance']:.3f}")

    print()


if __name__ == "__main__":
    main()
