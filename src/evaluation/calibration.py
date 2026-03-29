"""Calibration metrics: do predicted quantiles match observed frequencies?"""


def calibration_score(y_true, quantile_preds, quantiles):
    """Check if predicted Nth percentile actually occurs ~N% of the time."""
    raise NotImplementedError


def calibration_plot(y_true, quantile_preds, quantiles):
    """Generate predicted vs actual quantile calibration chart."""
    raise NotImplementedError
