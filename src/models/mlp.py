"""NumPy-only MLP for quantile kill prediction.

Implements a small feed-forward network trained with pinball loss to predict
kill-count quantiles [0.10, 0.25, 0.50, 0.75, 0.90].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray) -> float:
    """Average pinball loss across all samples and quantiles.

    Args:
        y_true: Shape (N,).
        y_pred: Shape (N, Q).
        quantiles: Shape (Q,).

    Returns:
        Scalar mean pinball loss.
    """
    y_true_col = y_true.reshape(-1, 1)
    error = y_true_col - y_pred
    q = quantiles.reshape(1, -1)
    loss = np.maximum(q * error, (q - 1.0) * error)
    return float(np.mean(loss))

def pinball_loss_per_quantile(
    y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Pinball loss for each quantile separately.

    Args:
        y_true: Shape (N,).
        y_pred: Shape (N, Q).
        quantiles: Shape (Q,).

    Returns:
        Shape (Q,) with average loss for each quantile.
    """
    y_true_col = y_true.reshape(-1, 1)
    error = y_true_col - y_pred
    q = quantiles.reshape(1, -1)
    loss = np.maximum(q * error, (q - 1.0) * error)
    return np.mean(loss, axis=0)
@dataclass
class TrainHistory:
    """Simple training history container."""

    train_loss: list[float]


class MLPQuantileRegressor:
    """Two hidden-layer MLP with ReLU activations and pinball loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (64, 32),
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dims = hidden_dims
        self.quantiles = np.array(quantiles, dtype=float)
        self.output_dim = len(quantiles)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must have exactly two values")

        rng = np.random.default_rng(seed)

        h1, h2 = hidden_dims
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, h1))
        self.b1 = np.zeros((1, h1))

        self.W2 = rng.normal(0.0, np.sqrt(2.0 / h1), size=(h1, h2))
        self.b2 = np.zeros((1, h2))

        self.W3 = rng.normal(0.0, np.sqrt(2.0 / h2), size=(h2, self.output_dim))
        self.b3 = np.zeros((1, self.output_dim))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(x.dtype)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        y_hat = a2 @ self.W3 + self.b3

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return y_hat, cache

    def _pinball_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Gradient dL/dy_pred for mean pinball loss with shape (N, Q)."""
        y_true_col = y_true.reshape(-1, 1)
        err = y_true_col - y_pred
        q = self.quantiles.reshape(1, -1)

        # d/dy_pred = -q when err > 0 else (1-q)
        grad = np.where(err > 0.0, -q, 1.0 - q)
        grad /= y_pred.shape[0] * y_pred.shape[1]
        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 400,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> TrainHistory:
        """Train the model with mini-batch SGD."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n = X.shape[0]
        history = TrainHistory(train_loss=[])
        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            indices = rng.permutation(n)
            X_shuf = X[indices]
            y_shuf = y[indices]

            for start in range(0, n, batch_size):
                stop = start + batch_size
                xb = X_shuf[start:stop]
                yb = y_shuf[start:stop]

                y_hat, cache = self._forward(xb)
                dL_dy = self._pinball_grad(yb, y_hat)

                a2 = cache["a2"]
                a1 = cache["a1"]
                z2 = cache["z2"]
                z1 = cache["z1"]

                dW3 = a2.T @ dL_dy
                db3 = np.sum(dL_dy, axis=0, keepdims=True)

                da2 = dL_dy @ self.W3.T
                dz2 = da2 * self._relu_grad(z2)
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self._relu_grad(z1)
                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                self.W3 -= lr * dW3
                self.b3 -= lr * db3
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            train_pred = self.predict(X)
            train_loss = pinball_loss(y, train_pred, self.quantiles)
            history.train_loss.append(train_loss)

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"[MLP] epoch={epoch + 1:4d}/{epochs} pinball_loss={train_loss:.4f}")

        return history

    def predict(self, X: np.ndarray, enforce_monotonic: bool = True) -> np.ndarray:
        """Predict quantiles with shape (N, 5)."""
        X = np.asarray(X, dtype=float)
        y_hat, _ = self._forward(X)
        if enforce_monotonic:
            y_hat = np.sort(y_hat, axis=1)
        return y_hat

    def save(self, path: str = "data/mlp_model.npz") -> None:
        """Serialize model parameters to disk."""
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            quantiles=self.quantiles,
            input_dim=self.input_dim,
            hidden_dims=np.array(self.hidden_dims, dtype=int),
        )

    @classmethod
    def load(cls, path: str = "data/mlp_model.npz") -> "MLPQuantileRegressor":
        """Load a serialized model from disk."""
        data = np.load(path)
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=tuple(data["hidden_dims"].tolist()),
            quantiles=tuple(data["quantiles"].tolist()),
        )

        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        return model


def train_from_features(
    features_path: str = "data/features.npz",
    out_path: str = "data/mlp_model.npz",
    lr: float = 1e-3,
    epochs: int = 400,
    batch_size: int = 128,
    hidden_dims: tuple[int, int] = (64, 32),
    include_extra_cluster_keys: bool = True,
    require_cluster_features: bool = False,
) -> MLPQuantileRegressor:
    """Convenience trainer that consumes the shared features.npz contract."""
    data = np.load(features_path)
    if "X_train" in data and "y_train" in data:
        X_train = data["X_train"]
        y_train = data["y_train"]
    else:
        raise KeyError("features.npz must contain X_train and y_train arrays")
    cluster_feature_count = 0
    if "feature_names" in data:
        feature_names = data["feature_names"]
        cluster_feature_count += int(np.sum(np.char.startswith(feature_names.astype(str), "cluster_")))

    if include_extra_cluster_keys:
        # Backward-compatible support for archives that store clustering features
        # in separate arrays instead of appending them into X_train directly.
        cluster_key_candidates = ("X_cluster_train", "cluster_features_train")
        cluster_key = next((k for k in cluster_key_candidates if k in data), None)
        if cluster_key is not None:
            X_cluster_train = np.asarray(data[cluster_key], dtype=float)
            if X_cluster_train.ndim != 2:
                raise ValueError(f"{cluster_key} must be a 2D array with shape (N, C)")
            if X_cluster_train.shape[0] != X_train.shape[0]:
                raise ValueError(
                    f"{cluster_key} row count ({X_cluster_train.shape[0]}) must match "
                    f"X_train row count ({X_train.shape[0]})"
                )
            X_train = np.concatenate([X_train, X_cluster_train], axis=1)
            cluster_feature_count += X_cluster_train.shape[1]
            print(
                f"[MLP] Appended {X_cluster_train.shape[1]} clustering features "
                f"from '{cluster_key}' (input_dim={X_train.shape[1]})."
            )

    if require_cluster_features and cluster_feature_count == 0:
        raise ValueError(
            "No clustering features found. Expected either cluster_* columns in feature_names "
            "or separate arrays X_cluster_train/cluster_features_train."
        )

    print(f"[MLP] Training with input_dim={X_train.shape[1]} (cluster_features={cluster_feature_count}).")
    model = MLPQuantileRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims)
    model.fit(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)
    # Report per-quantile pinball losses similar to quantile_reg.py output.
    train_pred = model.predict(X_train)
    train_q_loss = pinball_loss_per_quantile(y_train, train_pred, model.quantiles)
    print("[MLP] Final train pinball loss per quantile (10th, 25th, 50th, 75th, 90th):")
    print(train_q_loss)

    if "X_test" in data and "y_test" in data:
        test_pred = model.predict(data["X_test"])
        test_q_loss = pinball_loss_per_quantile(data["y_test"], test_pred, model.quantiles)
        print("[MLP] Final test pinball loss per quantile (10th, 25th, 50th, 75th, 90th):")
        print(test_q_loss)
    model.save(out_path)
    return model
