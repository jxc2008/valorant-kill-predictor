"""Flask API serving kill predictions and player similarity queries."""

from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mlp import MLPQuantileRegressor
from src.models.quantile_reg import QuantileRegression
from src.models.knn import KNNRetrieval
from src.data.features import CONTINUOUS_COLS, ROLE_COLS

app = Flask(__name__)
CORS(app)  # allow Next.js frontend to call this API

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(BASE, "data", "player_map_stats.csv")
FEAT_NPZ  = os.path.join(BASE, "data", "features.npz")
MLP_NPZ   = os.path.join(BASE, "data", "mlp_model.npz")
QR_NPZ    = os.path.join(BASE, "data", "quantile_model.npz")
EMB_NPZ   = os.path.join(BASE, "data", "embeddings.npz")

# ── Load everything once at startup ─────────────────────────────────────────
print("[API] Loading data...")
df = pd.read_csv(DATA_CSV)

# Compute derived features from raw columns
rounds_col = "map_score"  # approximate rounds played
df["rounds"] = df["map_score"].apply(
    lambda s: sum(int(x) for x in str(s).split("-")) if pd.notna(s) and "-" in str(s) else 24
).clip(lower=1)
df["kpr"]  = df["kills"]        / df["rounds"]
df["dpr"]  = df["deaths"]       / df["rounds"]
df["apr"]  = df["assists"]      / df["rounds"]
df["fbpr"] = df["first_bloods"] / df["rounds"]
df["role"] = "duelist"  # default role since CSV has no role column

feat_data   = np.load(FEAT_NPZ, allow_pickle=True)
scaler_data = np.load(os.path.join(BASE, "data", "scaler_params.npz"))
FEAT_MEAN   = scaler_data["mean"]   # shape (n_continuous,)
FEAT_STD    = scaler_data["std"]    # shape (n_continuous,)

mlp_model = MLPQuantileRegressor.load(MLP_NPZ)
print("[API] MLP loaded. W1 shape:", mlp_model.W1.shape)

qr_data = np.load(QR_NPZ, allow_pickle=True)
qr_model = QuantileRegression(quantiles=tuple(qr_data["quantiles"].tolist()))
qr_model.weights = qr_data["weights"]
qr_model.biases  = qr_data["biases"]
print("[API] Quantile Regression loaded.")

emb_data     = np.load(EMB_NPZ, allow_pickle=True)
embeddings   = emb_data["embeddings"]
player_names = list(emb_data["player_names"])
map_names    = list(emb_data["map_names"])

metadata = []
for pname, mname in zip(player_names, map_names):
    rows = df[(df["player_name"] == pname) & (df["map_name"] == mname)]
    metadata.append({
        "player_name": pname,
        "map_name":    mname,
        "kills":       float(rows["kills"].mean()) if len(rows) else float("nan"),
        "adr":         float(rows["adr"].mean())   if len(rows) else float("nan"),
        "kpr":         float("nan"),
        "role":        "unknown",
        "cluster":     -1,
    })

knn = KNNRetrieval(k=5)
knn.fit(embeddings, metadata)
print("[API] KNN index built.")

CLUSTER_ARCHETYPES = {
    0: "Archetype A",
    1: "Archetype B",
    2: "Archetype C",
    3: "Archetype D",
}

FEATURE_NAMES = [
    "ACS (Avg Combat Score)",
    "KPR (Kills / Round)",
    "ADR (Avg Damage / Round)",
    "KAST %",
    "First Bloods / Round",
    "Deaths / Round",
    "Assists / Round",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_feature_vector(player: str, map_name: str) -> np.ndarray | None:
    """
    Look up a player's average stats on a given map and return a
    normalized feature vector ready for the models.
    Returns None if the player is not found.
    """
    mask = (df["player_name"].str.lower() == player.lower())
    if map_name.lower() != "any":
        mask &= (df["map_name"].str.lower() == map_name.lower())

    rows = df[mask]
    if rows.empty:
        # Fall back to all maps for this player
        rows = df[df["player_name"].str.lower() == player.lower()]
    if rows.empty:
        return None

    # Continuous features — mean over matching rows
    cont = np.array([rows[c].mean() for c in CONTINUOUS_COLS], dtype=np.float32)

    # Normalize continuous features
    cont_norm = (cont - FEAT_MEAN) / FEAT_STD

    # Role one-hot — most common role
    role = rows["role"].mode().iloc[0] if "role" in rows.columns else "duelist"
    role_vec = np.array(
        [float(role == r) for r in ["duelist", "initiator", "controller", "sentinel"]],
        dtype=np.float32,
    )

    # Cluster one-hot — most common cluster
    cluster_cols = sorted(
        [c for c in rows.columns if c.startswith("cluster_")],
        key=lambda n: int(n.split("_", 1)[1]),
    )
    if cluster_cols:
        cluster_vec = np.array([rows[c].mean() for c in cluster_cols], dtype=np.float32)
        cluster_id  = int(np.argmax(cluster_vec))
    else:
        cluster_vec = np.array([], dtype=np.float32)
        cluster_id  = 0

    if len(cluster_vec) == 0:
        cluster_vec = np.zeros(4, dtype=np.float32)
    x = np.concatenate([cont_norm, role_vec, cluster_vec])
    return x, int(cluster_id), cont


def _quantiles_to_over_prob(quantile_preds: np.ndarray, kill_line: float) -> float:
    """
    Interpolate P(kills > kill_line) from predicted quantiles.
    quantile_preds: shape (5,) for quantiles [0.10, 0.25, 0.50, 0.75, 0.90]
    """
    qs = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    vals = quantile_preds

    if kill_line <= vals[0]:
        return 0.90
    if kill_line >= vals[-1]:
        return 0.10

    # Linear interpolation between adjacent quantiles
    for i in range(len(vals) - 1):
        if vals[i] <= kill_line <= vals[i + 1]:
            t = (kill_line - vals[i]) / max(vals[i + 1] - vals[i], 1e-6)
            prob_under = qs[i] + t * (qs[i + 1] - qs[i])
            return round(float(1.0 - prob_under), 3)

    return 0.50


def _get_similar_players(player: str, map_name: str, k: int = 3) -> list[str]:
    """Return k similar player names via KNN on embeddings."""
    try:
        idxs = [i for i, n in enumerate(player_names) if n.lower() == player.lower()]
        if not idxs:
            return []

        query_emb     = embeddings[idxs[0]]
        canonical     = player_names[idxs[0]]
        results       = knn.query(query_emb, exclude_player=canonical)
        seen, unique  = set(), []
        for r in results:
            name = r["player_name"]
            if name.lower() == player.lower() or name.lower() in seen:
                continue
            seen.add(name.lower())
            unique.append(name)
            if len(unique) >= k:
                break
        return unique
    except Exception:
        return []


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST", "GET"])
def predict_kills():
    """Predict kill distribution for a player on a given map."""
    if request.method == "GET":
        data = request.args
    else:
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            return jsonify({"error": "Invalid JSON body"}), 400

    player    = (data.get("player") or "").strip()
    map_name  = data.get("map") or "any"
    raw_line  = data.get("killLine", 15.5)
    model     = data.get("model") or "mlp"

    try:
        kill_line = float(raw_line)
    except (TypeError, ValueError):
        return jsonify({"error": f"killLine must be numeric, got {raw_line!r}"}), 400

    if not player:
        return jsonify({"error": "player is required"}), 400
    if kill_line < 0:
        return jsonify({"error": "killLine must be non-negative"}), 400
    if model not in ("mlp", "quantile_regression"):
        return jsonify({"error": f"unknown model {model!r}; expected 'mlp' or 'quantile_regression'"}), 400

    result = _build_feature_vector(player, map_name)
    if result is None:
        return jsonify({"error": f"Player '{player}' not found in dataset"}), 404

    x, cluster_id, raw_cont = result
    archetype = CLUSTER_ARCHETYPES.get(cluster_id, "Unknown")

    if model == "mlp":
        preds      = mlp_model.predict(x.reshape(1, -1))[0]   # shape (5,)
        kill_low   = int(round(float(preds[1])))               # 25th percentile
        kill_high  = int(round(float(preds[3])))               # 75th percentile
        over_prob  = _quantiles_to_over_prob(preds, kill_line)
        model_label = "MLP · 2-layer neural network · pinball loss"
        payload = {
            "killRange":      [kill_low, kill_high],
            "overProbability": over_prob,
            "archetype":      archetype,
            "similarPlayers": _get_similar_players(player, map_name),
            "model":          model_label,
        }

    else:  # quantile regression
        preds      = qr_model.predict(x.reshape(1, -1))[0]    # shape (5,)
        kill_low   = int(round(float(preds[1])))
        kill_high  = int(round(float(preds[3])))
        over_prob  = _quantiles_to_over_prob(preds, kill_line)
        model_label = "Quantile Regression · linear · pinball loss"

        # Feature importance — absolute weights at median quantile (index 2)
        median_weights = np.abs(qr_model.weights[:, 2])
        top_idx        = np.argsort(median_weights)[::-1][:7]
        cluster_cols = [f"cluster_{i}" for i in range(4)]
        all_cols = CONTINUOUS_COLS + ROLE_COLS + cluster_cols
        coeff_list = []
        for i in top_idx:
            col_name = all_cols[i] if i < len(all_cols) else f"feature_{i}"
            display_names = {
                "kpr": "KPR (Kills / Round)",
                "dpr": "DPR (Deaths / Round)",
                "apr": "APR (Assists / Round)",
                "adr": "ADR (Avg Damage / Round)",
                "kast": "KAST %",
                "fbpr": "First Bloods / Round",
                "acs": "ACS (Avg Combat Score)",
                "role_duelist": "Role: Duelist",
                "role_initiator": "Role: Initiator",
                "role_controller": "Role: Controller",
                "role_sentinel": "Role: Sentinel",
                "cluster_0": "Cluster A",
                "cluster_1": "Cluster B",
                "cluster_2": "Cluster C",
                "cluster_3": "Cluster D",
            }
            coeff_list.append({
                "feature": display_names.get(col_name, col_name.upper().replace("_", " ")),
                "weight": round(float(median_weights[i]), 4),
            })

        payload = {
            "killRange":           [kill_low, kill_high],
            "overProbability":     over_prob,
            "archetype":           archetype,
            "similarPlayers":      _get_similar_players(player, map_name),
            "model":               model_label,
            "featureCoefficients": coeff_list,
        }

    return jsonify(payload)


@app.route("/api/similar", methods=["GET"])
def similar_players():
    """Return k most similar player-map matchups via k-NN."""
    player   = request.args.get("player", "")
    map_name = request.args.get("map", "any")
    k        = int(request.args.get("k", 3))
    return jsonify({"similarPlayers": _get_similar_players(player, map_name, k=k)})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
