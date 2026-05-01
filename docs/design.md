# Design Document

## Repo Structure

```
valorant-kill-predictor/
├── README.md                          # Project overview, setup, and usage
├── requirements.txt                   # Python dependencies
├── data/
│   ├── player_map_stats.csv           # 41K+ rows of VCT 2025-2026 player-map stats
│   ├── features.npz                   # Normalized feature matrix (X_train, X_test, y_train, y_test)
│   ├── scaler_params.npz              # Mean and std for continuous feature normalization
│   ├── embeddings.npz                 # 8-dim learned player-map embeddings
│   ├── cluster_labels.npz             # K-means cluster assignments (K=4)
│   ├── features_with_cluster.npz      # features.npz with cluster one-hots appended
│   ├── mlp_model.npz                  # Trained MLP weights
│   ├── quantile_model.npz             # Trained quantile regression weights
│   └── split_boundary.json            # Chronological train/test boundary date
├── src/
│   ├── data/
│   │   ├── loader.py                  # CSV loading, score parsing, KPR/ADR/APR/FBPR derivation
│   │   ├── features.py                # Feature extraction, normalization, one-hot encoding
│   │   └── split.py                   # Chronological 80/20 train/test split
│   └── models/
│       ├── embeddings.py              # PyTorch embedding model (8-dim player-map fingerprint)
│       ├── knn.py                     # From-scratch KNN via L2 distance over embedding space
│       ├── clustering.py              # From-scratch K-means clustering (K=4 archetypes)
│       ├── mlp.py                     # From-scratch 2-layer MLP with pinball loss (NumPy)
│       └── quantile_reg.py            # From-scratch linear quantile regression with pinball loss (NumPy)
├── app/
│   ├── api.py                         # Flask API — /api/predict, /api/similar, /api/health
│   └── frontend/                      # Next.js UI — player search, kill range, model toggle
├── scripts/
│   ├── train.py                       # CLI pipeline (features, embeddings, features2, mlp stages)
│   └── predict.py                     # CLI for running kill predictions
├── tests/                             # Unit tests for model shapes, loss, and API endpoints
└── docs/
    ├── overview.md                    # Full project overview and pipeline description
    └── design.md                      # This document
```

---

## Division of Labor

| Team Member | Module | Deliverables |
|-------------|--------|--------------|
| **Joseph Cheng** (jxc2008) | Data pipeline, embeddings, KNN, Flask API | `src/data/loader.py`, `src/data/features.py`, `src/data/split.py`, `src/models/embeddings.py`, `src/models/knn.py`, `app/api.py` — CSV loading, feature extraction, temporal splitting, PyTorch embedding model, KNN retrieval, and Flask API wiring |
| **Ian Lu** (yl12003) | MLP neural network | `src/models/mlp.py` — from-scratch 2-layer MLP with ReLU activations, backpropagation, and pinball loss implemented in NumPy only |
| **Thomas Yanle Li** (yl12316) | Quantile regression | `src/models/quantile_reg.py` — from-scratch linear quantile regression with pinball loss and gradient descent implemented in NumPy only |
| **Shengyang Tao** (st5393) | K-means clustering | `src/models/clustering.py` — from-scratch K-means (K=4) over player embeddings, cluster labels appended as features for downstream models |
| **Alexandra Lugo** (anl3528) | Frontend, integration, evaluation | `app/frontend/` (Next.js UI), `tests/` — model toggle UI, kill distribution visualization, API integration, and end-to-end pipeline merging |

---

## Key Design Decisions

- **Pinball loss** used for both models — directly optimizes calibrated quantile predictions rather than MSE
- **Chronological split** — no shuffling; last 20% of matches by date form the test set to prevent data leakage
- **K=4 clusters fixed** — k-means on 8-dim embeddings; clusters capture performance tier (not playstyle) with ~93% variance on PC1
- **MLP as default model** — achieves comparable pinball loss to quantile regression but converges in ~400 epochs vs ~1500; QR available as interpretable alternative with per-feature coefficients
- **No sklearn** for Ian, Thomas, or Shengyang — all implemented from scratch in NumPy per course requirements
