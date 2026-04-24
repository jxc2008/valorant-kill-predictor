"""Temporal train/test splitting to prevent data leakage."""

import json
import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split chronologically by (year, event_name) ordering.

    Returns (train_df, test_df). The last (1 - train_ratio) fraction of unique
    events by year become the test set. This prevents leakage: the model never
    sees future matches during training.
    """
    # Rank unique events by year then event name (proxy for chronological order)
    events = (
        df[["year", "event_name"]]
        .drop_duplicates()
        .sort_values(["year", "event_name"])
        .reset_index(drop=True)
    )
    cutoff = int(len(events) * train_ratio)
    train_events = set(events.loc[:cutoff - 1, "event_name"])
    test_events  = set(events.loc[cutoff:,     "event_name"])

    train_df = df[df["event_name"].isin(train_events)].reset_index(drop=True)
    test_df  = df[df["event_name"].isin(test_events)].reset_index(drop=True)
    return train_df, test_df


def save_split_boundary(test_df: pd.DataFrame, path: str = "data/split_boundary.json") -> None:
    """Save the list of test event names so other modules can reproduce the split."""
    boundary = {"test_events": sorted(test_df["event_name"].unique().tolist())}
    with open(path, "w") as f:
        json.dump(boundary, f, indent=2)
