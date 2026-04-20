"""Feature extraction pipeline: raw stats -> model-ready feature vectors."""

import numpy as np
import pandas as pd

# Continuous features in fixed order (indices 0-6)
CONTINUOUS_COLS = ["kpr", "dpr", "apr", "adr", "kast", "fbpr", "acs"]
ROLE_COLS       = ["role_duelist", "role_initiator", "role_controller", "role_sentinel"]


def extract_player_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build the raw (un-normalized) feature matrix from a cleaned DataFrame.

    Returns:
        X            — shape (N, F), float32 feature matrix
        y            — shape (N,),   int32  kill targets
        feature_names— list of F column name strings
    """
    # One-hot encode role
    for role in ["duelist", "initiator", "controller", "sentinel"]:
        df[f"role_{role}"] = (df["role"] == role).astype(float)

    cols = CONTINUOUS_COLS + ROLE_COLS
    X = df[cols].values.astype(np.float32)
    y = df["kills"].values.astype(np.int32)
    return X, y, cols


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize continuous features using train-set statistics only.

    One-hot columns (indices 7-10) are left untouched.
    Returns (X_train_norm, X_test_norm, mean, std).
    """
    n_continuous = len(CONTINUOUS_COLS)
    mean = X_train[:, :n_continuous].mean(axis=0)
    std  = X_train[:, :n_continuous].std(axis=0)
    std  = np.where(std == 0, 1.0, std)          # avoid divide-by-zero

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()
    X_train_norm[:, :n_continuous] = (X_train[:, :n_continuous] - mean) / std
    X_test_norm[:,  :n_continuous] = (X_test[:,  :n_continuous] - mean) / std

    return X_train_norm, X_test_norm, mean, std


def build_and_save(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    out_path: str = "data/features.npz",
    scaler_path: str = "data/scaler_params.npz",
) -> dict:
    """
    Full pipeline: extract -> normalize -> save.
    Returns dict with X_train, X_test, y_train, y_test, feature_names.
    """
    X_train, y_train, feature_names = extract_player_features(train_df.copy())
    X_test,  y_test,  _             = extract_player_features(test_df.copy())

    X_train, X_test, mean, std = normalize_features(X_train, X_test)

    np.savez(out_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names),
        player_names_train=train_df["player_name"].values,
        player_names_test=test_df["player_name"].values,
        map_names_train=train_df["map_name"].values,
        map_names_test=test_df["map_name"].values,
    )
    np.savez(scaler_path, mean=mean, std=std)

    print(f"Saved features -> {out_path}  ({X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features)")
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                feature_names=feature_names)
