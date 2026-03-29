"""Quantile regression with pinball loss (from scratch, NumPy only).

Predicts 10th/25th/50th/75th/90th percentile kill counts.
No sklearn or library regression calls.
"""

import numpy as np


def pinball_loss(y_true, y_pred, quantile):
    """Compute pinball (quantile) loss."""
    raise NotImplementedError


class QuantileRegression:
    """From-scratch quantile regression using gradient descent."""

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        self.quantiles = quantiles

    def fit(self, X, y, lr=0.01, epochs=1000):
        """Train quantile regression weights via gradient descent."""
        raise NotImplementedError

    def predict(self, X):
        """Return predicted kill counts at each quantile."""
        raise NotImplementedError
