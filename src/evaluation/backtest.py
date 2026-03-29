"""Backtesting: compare ML predictions vs statistical baseline on held-out data."""


def backtest_model(model, test_data, baseline_fn):
    """Run predictions on test set and compute accuracy vs baseline."""
    raise NotImplementedError


def compute_metrics(y_true, y_pred_ml, y_pred_baseline):
    """Compare MAE, RMSE, and NLL between ML and baseline predictions."""
    raise NotImplementedError
