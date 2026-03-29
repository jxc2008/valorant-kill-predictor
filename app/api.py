"""Flask API serving kill predictions and player similarity queries."""

from flask import Flask

app = Flask(__name__)


@app.route("/api/predict", methods=["GET"])
def predict_kills():
    """Predict kill distribution for a player on a given map vs opponent."""
    raise NotImplementedError


@app.route("/api/similar", methods=["GET"])
def similar_players():
    """Return k most similar player-map matchups via k-NN."""
    raise NotImplementedError


if __name__ == "__main__":
    app.run(port=5000, debug=True)
