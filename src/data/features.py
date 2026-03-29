"""Feature extraction pipeline: raw stats to model-ready feature vectors."""


def extract_player_features(df):
    """Compute per-player-per-map feature vectors (KPR, ADR, KAST, etc.)."""
    raise NotImplementedError


def normalize_features(features):
    """Z-score normalize continuous features, handle missing values."""
    raise NotImplementedError
