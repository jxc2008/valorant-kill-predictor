# Data Contract — Valorant Kill Predictor

Defines every shared interface between team members. If you consume data produced by another module, follow this spec exactly. If you produce it, match it exactly.

---

## 1. Raw Data — `data/player_map_stats.csv`

**Owner: Joseph**  
**Consumers: Joseph (features), Thomas, Ian, Shengyang, Alexandra**

### Schema
| Column | Type | Description |
|---|---|---|
| `player_name` | str | IGN (lowercase, no spaces) |
| `map_name` | str | Title-case map name (e.g. `Haven`) |
| `agent` | str | Agent played (e.g. `Cypher`) |
| `kills` | int | **Target variable** |
| `deaths` | int | |
| `assists` | int | |
| `acs` | float | Average combat score |
| `adr` | float | Average damage per round |
| `kast` | float | KAST % (0–100) |
| `first_bloods` | int | First bloods this map |
| `map_score` | str | Final score string, e.g. `13-11` |
| `map_number` | int | Map index in the series (1, 2, 3) |
| `team1` | str | Player's team |
| `team2` | str | Opponent team |
| `event_name` | str | Tournament name |
| `year` | int | Season year |

### Derived columns (added by Joseph's `features.py`, not scraped)
| Column | Formula |
|---|---|
| `rounds_played` | parsed from `map_score`: sum of both sides |
| `kpr` | `kills / rounds_played` |
| `dpr` | `deaths / rounds_played` |
| `apr` | `assists / rounds_played` |
| `fbpr` | `first_bloods / rounds_played` |
| `role` | mapped from `agent`: `duelist`, `initiator`, `controller`, `sentinel` |

---

## 2. Feature Matrix — `data/features.npz`

**Owner: Joseph (`src/data/features.py`)**  
**Consumers: Thomas, Ian, Shengyang**

Produced by `features.py` after loading and cleaning the CSV. Saved as a compressed NumPy archive.

```python
np.savez("data/features.npz",
    X_train=...,   # shape (N_train, F)
    X_test=...,    # shape (N_test,  F)
    y_train=...,   # shape (N_train,) — raw kill counts (int)
    y_test=...,    # shape (N_test,)
    feature_names=...,  # shape (F,) — string array of column names
    player_names=...,   # shape (N,) — parallel to X rows, for KNN lookup
    map_names=...,      # shape (N,) — parallel to X rows
)
```

### Feature columns (in order, index 0..F-1)
| Index | Name | Notes |
|---|---|---|
| 0 | `kpr` | z-score normalized |
| 1 | `dpr` | z-score normalized |
| 2 | `apr` | z-score normalized |
| 3 | `adr` | z-score normalized |
| 4 | `kast` | z-score normalized (input is 0–100) |
| 5 | `fbpr` | z-score normalized |
| 6 | `acs` | z-score normalized |
| 7–10 | `role_duelist`, `role_initiator`, `role_controller`, `role_sentinel` | one-hot |
| 11+ | `cluster_0` … `cluster_{K-1}` | **one-hot cluster label — appended by Shengyang** |

**Normalization:** z-score on train set statistics only. Test set uses train mean/std. Joseph saves the scaler parameters in `data/scaler_params.npz` (`mean`, `std`, both shape `(7,)` for the 7 continuous features).

### Train/test split
- **80/20** chronological split: sort by `event_name` + `year`, last 20% of unique matches → test.
- No random shuffling — temporal split prevents leakage.
- Split boundary saved in `data/split_boundary.json`: `{"test_match_ids": [...]}`.

---

## 3. Embeddings — `data/embeddings.npz`

**Owner: Joseph (`src/models/embeddings.py`)**  
**Consumers: Joseph (KNN), Shengyang (K-Means)**

```python
np.savez("data/embeddings.npz",
    embeddings=...,   # shape (N, 8) — float32, one row per player-map row in features.npz
    player_names=..., # shape (N,)  — matches features.npz order
    map_names=...,    # shape (N,)
)
```

- `embed_dim = 8` (fixed — do not change without coordinating with Shengyang).
- Embeddings are **not** normalized post-training; Shengyang normalizes before k-means if needed.

---

## 4. KNN Output — `src/models/knn.py`

**Owner: Joseph**  
**Consumers: Alexandra (via `/api/similar`)**

```python
def query(player_name: str, map_name: str, k: int = 5) -> list[dict]:
    """
    Returns k most similar player-map observations by embedding distance.
    Each dict:
        {
            "player_name": str,
            "map_name":    str,
            "distance":    float,   # L2 distance in embedding space
            "kills":       float,   # median kills for this player on this map
            "kpr":         float,
            "adr":         float,
            "role":        str,
            "cluster":     int,
        }
    """
```

---

## 5. Cluster Labels — `data/cluster_labels.npz`

**Owner: Shengyang (`src/models/clustering.py`)**  
**Consumers: Joseph (appends to feature matrix), Thomas, Ian**

```python
np.savez("data/cluster_labels.npz",
    labels=...,        # shape (N,) — int, values in 0..K-1
    centroids=...,     # shape (K, 8) — cluster centroids in embedding space
    k=4,               # number of clusters used
    cluster_names=..., # shape (K,) — optional human-readable labels e.g. "entry", "sentinel"
)
```

- `K = 4` by default. Coordinate with Joseph before changing — it changes the feature matrix width.
- Labels are **parallel to `data/embeddings.npz`** rows (same index = same player-map obs).
- After Shengyang produces this file, Joseph re-runs `features.py` to append the one-hot cluster columns to `features.npz`.

---

## 6. Model Outputs

### 6a. Quantile Regression — Thomas (`src/models/quantile_reg.py`)

```python
def predict(X: np.ndarray) -> np.ndarray:
    """
    X: shape (N, F)
    Returns: shape (N, 5) — columns are quantiles [0.10, 0.25, 0.50, 0.75, 0.90]
    """
```

Saved artifact: `data/quantile_model.npz` — weights per quantile.

### 6b. MLP — Ian

```python
def predict(X: np.ndarray) -> np.ndarray:
    """
    Same signature as QuantileRegression.predict.
    X: shape (N, F)
    Returns: shape (N, 5) — columns are quantiles [0.10, 0.25, 0.50, 0.75, 0.90]
    """
```

Saved artifact: `data/mlp_model.npz` — layer weights and biases.

**Both models must return the same output shape.** Alexandra's frontend and the backtest (`src/evaluation/backtest.py`) treat them identically.

---

## 7. API Contract — `app/api.py`

**Owner: Alexandra**  
**Consumers: Frontend (Next.js)**

### `GET /api/predict`

**Query params:**
| Param | Type | Required | Description |
|---|---|---|---|
| `player` | str | yes | IGN, case-insensitive |
| `map` | str | yes | Map name, case-insensitive |
| `line` | float | yes | Kill line (e.g. `19.5`) |
| `model` | str | no | `"quantile"` (default) or `"mlp"` |

**Response `200`:**
```json
{
  "player": "aspas",
  "map": "Haven",
  "line": 19.5,
  "model": "quantile",
  "quantiles": {
    "q10": 12.0,
    "q25": 16.0,
    "q50": 20.0,
    "q75": 24.0,
    "q90": 28.0
  },
  "p_over": 0.54,
  "p_under": 0.46,
  "cluster": 2,
  "cluster_name": "entry"
}
```

`p_over` / `p_under`: fraction of the empirical quantile distribution above/below the line. Computed by linear interpolation between the 5 quantile points.

**Response `404`:** `{"error": "player not found"}` — player or map has no data.

---

### `GET /api/similar`

**Query params:**
| Param | Type | Required | Description |
|---|---|---|---|
| `player` | str | yes | IGN |
| `map` | str | yes | Map name |
| `k` | int | no | Number of results (default 5, max 20) |

**Response `200`:**
```json
{
  "query": {"player": "aspas", "map": "Haven"},
  "similar": [
    {
      "player_name": "TenZ",
      "map_name": "Haven",
      "distance": 0.32,
      "kills": 22.0,
      "kpr": 0.81,
      "adr": 158.0,
      "role": "duelist",
      "cluster": 2
    }
  ]
}
```

---

## 8. Evaluation Interface — `src/evaluation/backtest.py`

**Owner: Alexandra**

```python
def backtest(model, X_test, y_test, lines, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """
    model: any object with .predict(X) -> shape (N, 5)
    lines: shape (N,) — the kill line for each test row
    Returns dict with keys: pinball_loss (per quantile), coverage (per quantile), brier_score
    """
```

Both Thomas and Ian pass their model objects into this function. Alexandra owns the implementation.

---

## 9. Dependency Order

```
Joseph scrapes → player_map_stats.csv
Joseph runs features.py → features.npz (without cluster cols), scaler_params.npz
Joseph runs embeddings.py → embeddings.npz
Shengyang runs clustering.py → cluster_labels.npz
Joseph re-runs features.py → features.npz (with cluster one-hot cols appended)
Thomas trains quantile_reg.py → quantile_model.npz
Ian trains mlp.py → mlp_model.npz
Alexandra wires app/api.py to both models + KNN → serves frontend
```

---

## 10. File Checklist

| File | Owner | Consumers |
|---|---|---|
| `data/player_map_stats.csv` | Joseph | all |
| `data/features.npz` | Joseph | Thomas, Ian, Shengyang |
| `data/scaler_params.npz` | Joseph | Thomas, Ian, Alexandra |
| `data/split_boundary.json` | Joseph | Thomas, Ian, Alexandra |
| `data/embeddings.npz` | Joseph | Shengyang, Joseph (KNN) |
| `data/cluster_labels.npz` | Shengyang | Joseph, Thomas, Ian |
| `data/quantile_model.npz` | Thomas | Alexandra |
| `data/mlp_model.npz` | Ian | Alexandra |
