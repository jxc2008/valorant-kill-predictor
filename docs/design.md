# Design Document

## Repo Structure

```
valorant-kill-predictor/
├── README.md                          # Project overview, setup, and usage
├── requirements.txt                   # Python dependencies
├── data/                              # Raw and processed datasets
│   └── player_map_stats.csv           # 41K+ rows of VCT player performance data
├── notebooks/                         # Exploratory data analysis and experiments
├── src/
│   ├── data/
│   │   ├── loader.py                  # CSV/SQLite data loading and cleaning
│   │   ├── features.py                # Feature extraction (KPR, ADR, KAST, role, opponent strength)
│   │   └── split.py                   # Temporal train/test split (no data leakage)
│   ├── models/
│   │   ├── embeddings.py              # Player embedding model (PyTorch)
│   │   ├── quantile_reg.py            # Quantile regression with pinball loss (from scratch)
│   │   └── knn.py                     # k-NN similarity search with cosine similarity (from scratch)
│   ├── evaluation/
│   │   ├── backtest.py                # ML vs baseline comparison on held-out data
│   │   └── calibration.py             # Quantile calibration metrics and plots
│   └── visualization/
│       └── embedding_viz.py           # PCA projection and embedding scatter plots
├── app/
│   ├── api.py                         # Flask API serving predictions and similarity queries
│   └── frontend/                      # Next.js web interface
├── scripts/
│   ├── train.py                       # CLI for end-to-end model training
│   └── predict.py                     # CLI for running kill predictions
├── tests/                             # Unit and integration tests
├── trained_models/                    # Saved model weights and metadata
└── docs/
    └── design.md                      # This document
```

## Division of Labor

| Team Member | Module | Deliverables |
|-------------|--------|--------------|
| **Joseph Cheng** (jxc2008) | Data pipeline + Embeddings | `src/data/*`, `src/models/embeddings.py`, `scripts/train.py` — data scraping, feature extraction, temporal splitting, and PyTorch embedding model |
| **Ian Lu** (yl12003) | Quantile regression | `src/models/quantile_reg.py` — from-scratch implementation of pinball loss and gradient descent, kill percentile predictions |
| **Shengyang Tao** (st5393) | k-NN retrieval | `src/models/knn.py` — from-scratch cosine similarity and k-nearest neighbor search over embedding space |
| **Alexandra Lugo** (anl3528) | Evaluation + Calibration | `src/evaluation/*` — backtesting framework, ML vs baseline metrics, quantile calibration analysis |
| **Thomas Yanle Li** (yl12316) | Visualization + Frontend | `src/visualization/*`, `app/*` — PCA embedding plots, Flask API endpoints, Next.js web interface |
