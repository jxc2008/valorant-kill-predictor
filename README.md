# Valorant Kill Line Predictor

Predicting professional Valorant player kill counts using player embeddings, k-means clustering, quantile regression, and a from-scratch MLP neural network trained on historical VCT match data.

**Course:** CSCI-UA 473 — Foundations of Machine Learning (Spring 2026)

---

## What is this project?

We built a machine learning system that predicts how many kills a professional Valorant player will get in a single map, and turns that prediction into a calibrated probability distribution — e.g. "there's a 60% chance aspas gets over 20 kills on Haven."

**Valorant** is a 5v5 competitive FPS esport by Riot Games with a professional circuit called the VCT (Valorant Champions Tour). Each match is played across multiple maps. Key stats we model:

- **Kills** — the primary prediction target
- **ACS** — Average Combat Score, a composite performance metric
- **ADR** — Average Damage per Round
- **KAST** — % of rounds with a Kill, Assist, Survive, or Trade
- **Agent role** — Duelist, Initiator, Controller, or Sentinel (determines playstyle)

---

## Team

| Name | NetID | Role |
|------|-------|------|
| Joseph Cheng | jxc2008 | Data pipeline, embeddings, KNN, Flask API |
| Ian Lu | yl12003 | MLP neural network (from scratch, NumPy) |
| Thomas Yanle Li | yl12316 | Quantile regression (from scratch, NumPy) |
| Shengyang Tao | st5393 | K-means clustering (from scratch, NumPy) |
| Alexandra Lugo | anl3528 | Frontend (Next.js), integration |

---

## ML Pipeline

```
Raw CSV (rib.gg + vlr.gg)
     │
     ▼
[Joseph] Feature Extraction + Normalization
     │  → data/features.npz
     │
     ├──► [Joseph] Embedding Model (PyTorch)
     │         → data/embeddings.npz
     │                │
     │                ├──► [Joseph] KNN retrieval
     │                │
     │                └──► [Shengyang] K-Means Clustering
     │                          → data/cluster_labels.npz
     │
     ├──► [Thomas] Quantile Regression (pinball loss, NumPy)
     │
     └──► [Ian] MLP Neural Network (pinball loss, NumPy)
                        │
                        ▼
             [Alexandra] Flask API + Next.js UI
```

Both models predict kill count at 5 quantiles: **[10th, 25th, 50th, 75th, 90th]**. The MLP is the default model — it achieves comparable pinball loss to quantile regression but converges in fewer epochs. Quantile regression is available as an interpretable alternative with per-feature coefficients.

---

## Dataset

- **Source:** rib.gg (player stats) + vlr.gg (team/pick-ban data)
- **Coverage:** VCT 2025–2026, 35 events
- **Size:** 1,707 matches · 41,246 player-map rows · 2,246 unique players
- **Split:** Chronological 80/20 train/test (last 20% of matches by date)

The dataset is included in `data/player_map_stats.csv`.

---

## Setup

```bash
git clone https://github.com/jxc2008/valorant-kill-predictor.git
cd valorant-kill-predictor
pip install -r requirements.txt
```

Node.js v18+ is required for the frontend:
```bash
cd app/frontend
npm install
```

---

## Running the project

All trained model files are included in `data/` — you do not need to retrain to run the app.

**Step 1 — Start the Flask API (Terminal 1):**
```bash
python app/api.py
```
Wait for `[API] KNN index built.` before proceeding.

**Step 2 — Start the frontend (Terminal 2):**
```bash
cd app/frontend
npm run dev
```

**Step 3 — Open the app:**

Go to `http://localhost:3000`

---

## Retraining from scratch (optional)

```bash
# Build feature matrix
python scripts/train.py --stage features

# Train embedding model
python scripts/train.py --stage embeddings

# Run clustering (produces cluster_labels.npz)
# Run Shengyang's clustering script in src/models/clustering.py

# Re-run features with cluster labels appended
python scripts/train.py --stage features2

# Train MLP
python scripts/train.py --stage mlp

# Train quantile regression
# Run Thomas's script in src/models/quantile_reg.py
```

---

## API Endpoints

```bash
# Predict kill distribution
POST /api/predict
Body: { "player": "TenZ", "map": "Bind", "killLine": 15.5, "model": "mlp" }

# Find similar players
GET /api/similar?player=TenZ&map=Bind&k=3

# Health check
GET /api/health
```

---

## Repo Structure

```
valorant-kill-predictor/
├── app/
│   ├── api.py                         # Flask API (predict, similar, health endpoints)
│   └── frontend/                      # Next.js UI
│       ├── app/
│       │   ├── api/predict/route.ts   # Next.js API route proxying to Flask
│       │   ├── page.tsx               # Main UI page
│       │   └── layout.tsx             # App layout
│       ├── package.json
│       └── tsconfig.json
├── data/
│   ├── player_map_stats.csv           # Raw VCT 2025-2026 match data
│   ├── features.npz                   # Normalized feature matrix
│   ├── features_with_cluster.npz      # Features with cluster one-hots appended
│   ├── scaler_params.npz              # Normalization parameters
│   ├── embeddings.npz                 # 8-dim learned player-map embeddings
│   ├── cluster_labels.npz             # K-means cluster assignments
│   ├── cluster_pca.png                # PCA visualization of embedding space
│   ├── mlp_model.npz                  # Trained MLP weights
│   ├── quantile_model.npz             # Trained quantile regression weights
│   └── split_boundary.json            # Chronological train/test boundary
├── docs/
│   ├── overview.md                    # Full project overview and pipeline
│   ├── design.md                      # Repo structure and division of labor
│   └── data_contract.md               # File format and interface specifications
├── scripts/
│   ├── train.py                       # CLI pipeline (features, embeddings, mlp)
│   ├── predict.py                     # CLI prediction tool
│   └── plot_clusters.py               # Generate PCA cluster visualization
├── src/
│   ├── data/
│   │   ├── loader.py                  # CSV loading and stat derivation
│   │   ├── features.py                # Feature extraction and normalization
│   │   ├── split.py                   # Chronological train/test split
│   │   └── view_features.py           # Feature inspection utility
│   ├── models/
│   │   ├── embeddings.py              # PyTorch embedding model
│   │   ├── knn.py                     # From-scratch KNN (L2 distance)
│   │   ├── clustering.py              # From-scratch K-means (K=4)
│   │   ├── mlp.py                     # From-scratch MLP with pinball loss
│   │   └── quantile_reg.py            # From-scratch quantile regression
│   ├── evaluation/
│   │   └── backtest.py                # MLP vs naive baseline accuracy comparison
│   └── visualization/
│       └── embedding_viz.py           # PCA projection and scatter plots
├── tests/
│   └── __init__.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Key design decisions

- **Pinball loss** (quantile loss) is used for both models — it directly optimizes the quantile predictions rather than mean squared error, producing calibrated uncertainty estimates
- **Chronological split** — no shuffling, last 20% of matches by date form the test set to avoid data leakage
- **K=4 clusters** fixed via k-means on player embeddings — clusters capture performance tier rather than playstyle (dominant variance on PC1)
- **MLP vs QR tradeoff** — MLP converges faster (~400 epochs vs ~1500) with slightly lower 10th percentile loss; QR provides interpretable feature coefficients per quantile
- **No sklearn** for Ian, Thomas, or Shengyang's models — all implemented from scratch in NumPy
