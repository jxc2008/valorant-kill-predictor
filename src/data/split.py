"""Temporal train/test splitting to prevent data leakage."""


def temporal_split(df, train_ratio=0.8):
    """Split data chronologically by event ordering.

    Uses event sequence as a time proxy since match dates are unavailable.
    """
    raise NotImplementedError
