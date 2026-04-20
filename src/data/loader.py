"""Load and clean player_map_stats.csv into an analysis-ready DataFrame."""

import pandas as pd
import numpy as np

AGENT_ROLES = {
    # Duelists
    "Jett": "duelist", "Reyna": "duelist", "Phoenix": "duelist",
    "Neon": "duelist", "Yoru": "duelist", "Iso": "duelist", "Waylay": "duelist",
    # Initiators
    "Sova": "initiator", "Breach": "initiator", "Skye": "initiator",
    "KAY/O": "initiator", "Fade": "initiator", "Gekko": "initiator",
    # Controllers
    "Brimstone": "controller", "Viper": "controller", "Omen": "controller",
    "Astra": "controller", "Harbor": "controller", "Clove": "controller",
    # Sentinels
    "Killjoy": "sentinel", "Cypher": "sentinel", "Sage": "sentinel",
    "Chamber": "sentinel", "Deadlock": "sentinel", "Vyse": "sentinel",
}

ROLE_INDEX = {"duelist": 0, "initiator": 1, "controller": 2, "sentinel": 3}


def _parse_rounds(score_str: str) -> int:
    """Parse '13-11' -> 24. Returns 0 on failure."""
    try:
        a, b = str(score_str).strip().split("-")
        return int(a) + int(b)
    except Exception:
        return 0


def load_player_stats(csv_path: str = "data/player_map_stats.csv") -> pd.DataFrame:
    """
    Read player_map_stats.csv, clean it, and add derived columns.

    Added columns:
        rounds_played  — total rounds from map_score
        kpr            — kills per round
        dpr            — deaths per round
        apr            — assists per round
        fbpr           — first bloods per round
        role           — agent role string (duelist/initiator/controller/sentinel)
        role_idx       — integer index for role (0-3)

    Drops rows with missing kills, zero rounds, or missing numeric stats.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df["rounds_played"] = df["map_score"].apply(_parse_rounds)
    df = df[df["rounds_played"] > 0].copy()

    df = df.dropna(subset=["kills"])
    df["kills"] = df["kills"].astype(int)

    df["kpr"]  = df["kills"]        / df["rounds_played"]
    df["dpr"]  = df["deaths"]       / df["rounds_played"]
    df["apr"]  = df["assists"]      / df["rounds_played"]
    df["fbpr"] = df["first_bloods"] / df["rounds_played"]

    df["agent"]    = df["agent"].str.strip()
    df["role"]     = df["agent"].map(AGENT_ROLES).fillna("duelist")
    df["role_idx"] = df["role"].map(ROLE_INDEX).fillna(0).astype(int)

    for col in ["acs", "adr", "kast"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["acs", "adr", "kast"]).reset_index(drop=True)
    return df


def load_match_metadata(csv_path: str = "data/player_map_stats.csv") -> pd.DataFrame:
    """Return one row per unique match (team1, team2, event_name, year)."""
    df = load_player_stats(csv_path)
    return (
        df[["team1", "team2", "event_name", "year", "map_name", "map_number"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
