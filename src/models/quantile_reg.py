"""Quantile regression with pinball loss implemented from scratch.

Predicts 10th/25th/50th/75th/90th percentile kill counts.
"""

import numpy as np


def pinball_loss(y_true, y_pred, quantile):
    """Compute pinball (quantile) loss."""
    error = y_true - y_pred
    # compact pinball loss
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error)) 


class QuantileRegression:
    """From-scratch quantile regression using gradient descent."""

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        self.quantiles = np.array(quantiles)
        self.weights = None
        self.biases = None

    def fit(self, X, y, lr=0.01, epochs=1000):
        """Train quantile regression weights via gradient descent."""
        N, F = X.shape # N samples, F features
        Q = len(self.quantiles)
        
        # Initialize parameters
        self.weights = np.zeros((F, Q))
        self.biases = np.zeros(Q)
        
        # Reshape y for broadcasting against (N, Q) predictions
        y_reshaped = y[:, np.newaxis]
        
        for _ in range(epochs):
            # Forward pass
            y_pred = X.dot(self.weights) + self.biases
            
            # Error and gradient
            error = y_reshaped - y_pred
            grad_y_pred = np.where(error >= 0, -self.quantiles, 1 - self.quantiles)
            
            # Parameter gradients
            grad_weights = X.T.dot(grad_y_pred) / N
            grad_biases = np.mean(grad_y_pred, axis=0)
            
            # Gradient descent update
            self.weights -= lr * grad_weights
            self.biases -= lr * grad_biases

    def predict(self, X):
        """Return predicted kill counts at each quantile."""
        if self.weights is None or self.biases is None:
            raise ValueError("Model has not been fitted yet.")
        return X.dot(self.weights) + self.biases