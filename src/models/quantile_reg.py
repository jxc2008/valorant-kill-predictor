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
        self.loss_history_ = []
        
        # Reshape y for broadcasting against (N, Q) predictions
        y_reshaped = y[:, np.newaxis]
        
        for _ in range(epochs):
            # Forward pass
            y_pred = X.dot(self.weights) + self.biases
            
            # Error and gradient
            error = y_reshaped - y_pred
            
            # Record loss per quantile
            epoch_loss = np.mean(np.maximum(self.quantiles * error, (self.quantiles - 1) * error), axis=0)
            self.loss_history_.append(epoch_loss)
            
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


if __name__ == "__main__":
    import os
    
    # --- CONFIGURATION ---
    DATASET_FILENAME = "features_with_cluster.npz"  # Change to "features_with_cluster.npz" for clusters data appended
    
    OUTPUT_FILENAME = "quantile_model.npz" if DATASET_FILENAME == "features.npz" else "quantile_model_with_cluster.npz"
    # ---------------------
    
    # Resolve absolute paths based on this file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", DATASET_FILENAME)
    out_path = os.path.join(base_dir, "data", OUTPUT_FILENAME)
    
    print(f"Loading data from: {DATASET_FILENAME}...")
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    
    print(f"Training QuantileRegression on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    model = QuantileRegression()
    model.fit(X_train, y_train, lr=0.01, epochs=1000)
    
    print("\nTraining complete!")
    print("Final loss per quantile (10th, 25th, 50th, 75th, 90th):")
    print(model.loss_history_[-1])
    
    print(f"\nSaving model to: {OUTPUT_FILENAME}...")
    np.savez(out_path, weights=model.weights, biases=model.biases, quantiles=model.quantiles, loss_history=model.loss_history_)
    print("Done!")