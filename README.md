# Valorant Kill Line Predictor

Predicting professional Valorant player kill counts using player embeddings, quantile regression, and k-NN retrieval on historical VCT match data.

**Course:** CSCI-UA 473 — Foundations of Machine Learning (Spring 2026)

## Team

| Name | NetID | Role |
|------|-------|------|
| Alexandra Lugo | anl3528 | Evaluation & calibration |
| Ian Lu | yl12003 | Quantile regression (from scratch) |
| Joseph Cheng | jxc2008 | Data pipeline & embeddings |
| Shengyang Tao | st5393 | k-NN retrieval (from scratch) |
| Thomas Yanle Li | yl12316 | Visualization & frontend |

## Setup

```bash
pip install -r requirements.txt
```

## Data

VCT 2025-2026 professional match data scraped from rib.gg and vlr.gg. Place `player_map_stats.csv` in `data/`.

## Usage

```bash
# Train embedding model
python scripts/train.py --data data/player_map_stats.csv

# Run predictions
python scripts/predict.py --player "TenZ" --map "Bind"

# Launch API
python app/api.py
```
