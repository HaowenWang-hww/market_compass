"""Load and preprocess Wind All-A index data."""

from pathlib import Path

import pandas as pd

import config


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load 万得全A data: skip 6 metadata rows, keep date/close/amt."""
    path = Path(path) if path is not None else config.DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place 万得全A数据.xlsx in the project root."
        )

    df = pd.read_excel(
        path,
        skiprows=6,
        header=None,
        names=["date", "close", "amt"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.dropna(subset=["close"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce").ffill()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df
