# Project Overview — Valorant Kill Predictor

## What is this project?

We're building a machine learning system that predicts how many kills a professional Valorant player will get in a single map, and turns that prediction into a probability distribution (e.g. "there's a 60% chance aspas gets over 20 kills on Haven").

This has real-world use: sports betting platforms like PrizePicks set "kill lines" for pro players (e.g. "aspas over/under 20.5 kills"). If our model is well-calibrated, it can identify edges where the line is mispriced.

---

## Quick VALORANT Primer (for non-players)

**Valorant** is a 5v5 competitive first-person shooter. Each match is played on a **map** (like Haven, Bind, Split — think of them as different arenas). A map is won by the first team to win 13 rounds (or more in overtime).

- **Kills** — the main stat we're predicting. A player kills an enemy by eliminating them in a round.
- **Agent** — each player picks a character class (like a role in a team sport). Agents fall into 4 roles:
  - **Duelist** — aggressive fraggers, highest kill counts (e.g. Jett, Neon, Reyna)
  - **Initiator** — support/info gatherers, medium kills (e.g. Sova, Fade)
  - **Controller** — smoke/utility players, fewer kills (e.g. Omen, Astra)
  - **Sentinel** — defensive/anchor players, lowest kills (e.g. Cypher, Killjoy)
- **ACS** (Average Combat Score) — a composite performance metric, roughly "how impactful was this player"
- **ADR** (Average Damage per Round) — damage dealt per round, correlates with kills
- **KAST** — % of rounds where the player got a Kill, Assist, Survived, or Traded (measures consistency)
- **First Blood** — killing the first enemy in a round (high-value play)

Each player-map observation in our dataset is one player's stats for one map played in one match.

---

## Pipeline Overview

```
Raw Data (CSV)
     │
     ▼
[Joseph] Feature Extraction + Normalization
     │  → features.npz  (normalized stat vectors for each player-map)
     │
     ├──► [Joseph] Embedding Model (PyTorch)
     │         → embeddings.npz  (8-dim learned representation per player-map)
     │                │
     │                ├──► [Joseph] KNN  (find similar player-map matchups)
     │                │
     │                └──► [Shengyang] K-Means Clustering
     │                          → cluster_labels.npz  (role archetype per player)
     │
     ├──► [Thomas] Quantile Regression   ─┐
     │         (predicts kill percentiles) │
     │                                     ├──► [Alexandra] Flask API + Next.js UI
     └──► [Ian] MLP Neural Network       ─┘         + Evaluation / Backtest
               (predicts kill percentiles)
```

The **features.npz** file (produced by Joseph) is the shared input that Thomas, Ian, and Shengyang all consume. Nothing downstream can start until Joseph's piece is done.

---

## Team Assignments

---

### Joseph — Data Pipeline, Embeddings, KNN
**Files:** `src/data/loader.py`, `src/data/features.py`, `src/data/split.py`, `src/models/embeddings.py`, `src/models/knn.py`

**What you're building:**
1. Load and clean `data/player_map_stats.csv` — parse scores, derive KPR/ADR/APR/FBPR, map agents to roles
2. Build a normalized feature matrix and save it as `data/features.npz`
3. Train a small PyTorch embedding model that learns a compact 8-number "fingerprint" for each player-map observation (trained by predicting kills)
4. Use those embeddings for KNN: given "aspas on Haven", find the 5 most similar player-map combos historically

**Key output:** `data/features.npz`, `data/embeddings.npz`
**Dependency:** Thomas, Ian, and Shengyang are blocked until you produce `features.npz`

---

### Thomas — Quantile Regression
**Files:** `src/models/quantile_reg.py`

**What you're building:**
A linear model trained with **pinball loss** (also called quantile loss) that predicts not just the average kills, but the full distribution — specifically the 10th, 25th, 50th, 75th, and 90th percentile.

**Why quantile regression?**
Instead of saying "aspas will get 22 kills", we say "there's a 10% chance he gets under 14, a 50% chance under 22, and a 90% chance under 30." This is much more useful for evaluating betting lines.

**Pinball loss** for a single quantile `q`:
```
loss = (y_true - y_pred) * q          if y_true >= y_pred
loss = (y_pred - y_true) * (1 - q)   if y_true < y_pred
```

You train **5 separate models**, one per quantile. No sklearn — implement gradient descent from scratch with NumPy.

**Input:** `data/features.npz` (produced by Joseph)
**Output:** `data/quantile_model.npz`, and a `predict(X)` method returning shape `(N, 5)`

**Getting started:**
```python
import numpy as np
data = np.load("data/features.npz")
X_train, y_train = data["X_train"], data["y_train"]
# X_train shape: (N, F) — each row is one player-map observation
# y_train shape: (N,)   — kill count for that observation
```

---

### Ian — MLP Neural Network
**Files:** `src/models/mlp.py` (create this file)

**What you're building:**
A small 2-3 layer neural network with the **same pinball loss** as Thomas, but using a non-linear architecture. Think of it as: can a neural net beat Thomas's linear model?

**Architecture (suggestion):**
```
Input (F features) → Linear → ReLU → Linear → ReLU → Linear → 5 outputs (one per quantile)
```

Implement from scratch using only NumPy (no PyTorch, no sklearn). You need:
- Forward pass (matrix multiplications + ReLU)
- Backward pass (backpropagation — compute gradients by hand)
- Weight update (gradient descent or SGD)

**Input:** Same `data/features.npz` as Thomas
**Output:** `data/mlp_model.npz`, and a `predict(X)` method returning shape `(N, 5)`

**Your model must have the same output shape as Thomas's** (`(N, 5)`) so Alexandra's evaluation code can swap them interchangeably.

---

### Shengyang — K-Means Clustering
**Files:** `src/models/clustering.py`

**What you're building:**
Group players into **role archetypes** based on their embedding vectors. For example, cluster 0 might be "aggressive duelists with high kills", cluster 1 might be "support initiators with low kills but high assists". 

After clustering, the cluster label becomes an **additional feature** for Thomas and Ian's models — the intuition is that knowing "this player is a sentinel" helps predict their kill count.

**K-Means from scratch (NumPy only):**
1. Initialize K centroids randomly from the data
2. Assign each embedding to its nearest centroid (L2 distance)
3. Recompute centroids as the mean of all assigned points
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Input:** `data/embeddings.npz` (produced by Joseph)
**Output:** `data/cluster_labels.npz` with labels shape `(N,)` and centroids shape `(K, 8)`

**After you produce this file**, Joseph will re-run the feature pipeline to append cluster labels as one-hot columns to `features.npz`, which Thomas and Ian then retrain on.

**Optional extension:** Implement GMMs (Gaussian Mixture Models) as an alternative. GMMs give soft cluster assignments (probabilities) instead of hard labels.

---

### Alexandra — Frontend, API, Evaluation
**Files:** `app/api.py`, `src/evaluation/backtest.py`, `src/evaluation/calibration.py`, `tests/`, Next.js frontend

**What you're building:**

**1. Flask API** (`app/api.py`):
- `GET /api/predict?player=aspas&map=Haven&line=20.5` → kill distribution + over/under probability
- `GET /api/similar?player=aspas&map=Haven&k=5` → similar historical player-map performances
- See `docs/data_contract.md` Section 7 for exact response format

**2. Backtest** (`src/evaluation/backtest.py`):
- Load test set from `features.npz`
- Run both Thomas's and Ian's models on it
- Compute pinball loss per quantile, coverage (did the 90th percentile actually contain 90% of outcomes?), Brier score
- Output a comparison table

**3. Next.js Frontend** (create `frontend/` directory):
- Player search bar
- Show kill distribution as a chart (5 quantile points as a box plot or range bar)
- Show cluster label ("role archetype")
- Show top-5 similar players
- Deploy on Vercel, point at Flask API

**4. Tests** (`tests/`):
- Test that model outputs have the right shape
- Test that pinball loss is non-negative
- Test that `p_over + p_under ≈ 1.0`
- Test API endpoints return 200 with correct keys

---

## Data Contract

Full interface specification (file formats, array shapes, function signatures) is in **`docs/data_contract.md`**. Read it before you start coding — it defines exactly what each file should look like so everyone can work independently.

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Build feature matrix (Joseph runs this first)
python scripts/train.py --stage features

# Train embedding model
python scripts/train.py --stage embeddings

# Run clustering (Shengyang)
python scripts/train.py --stage clustering

# Train prediction models
python scripts/train.py --stage quantile
python scripts/train.py --stage mlp

# Evaluate both models
python scripts/train.py --stage evaluate

# Run API server
python app/api.py

# Make a prediction
python scripts/predict.py --player aspas --map Haven --line 20.5
```

---

## Key Rules

- **No sklearn** for Thomas or Ian's models — implement math from scratch
- **NumPy only** for Thomas, Ian, Shengyang (no PyTorch except in Joseph's embedding model)
- **Chronological train/test split** — no shuffling, last 20% of matches by date = test set
- **K=4 clusters fixed** — coordinate with Joseph before changing
- **Both prediction models must return `(N, 5)` arrays** with quantiles `[0.10, 0.25, 0.50, 0.75, 0.90]`
