"""Directional (signed) causal features for 3-state prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURE_COLS

DIRECTIONAL_FEATURE_COLS = [
    "mom_20",
    "mom_60",
    "mom_120",
    "ma_spread_signed",
    "price_vs_ma60",
    "slope_120_signed",
    "ret_sign_streak",
    "amt_trend",
]

FEATURE_COLS_3STATE = FEATURE_COLS + DIRECTIONAL_FEATURE_COLS


def _ols_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return np.nan
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _signed_streak(daily_ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling sum of sign(daily return) — positive = up streak, negative = down."""

    def _streak(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sign(valid).sum())

    return daily_ret.rolling(window, min_periods=window).apply(_streak, raw=True)


def build_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build signed features using only information <= t."""
    out = df[["date", "close", "amt"]].copy()
    close = out["close"]
    amt = out["amt"]
    log_close = np.log(close)
    daily_ret = close.pct_change()

    out["mom_20"] = log_close.diff(20)
    out["mom_60"] = log_close.diff(60)
    out["mom_120"] = log_close.diff(120)

    ma10 = close.rolling(10, min_periods=10).mean()
    ma30 = close.rolling(30, min_periods=30).mean()
    ma60 = close.rolling(60, min_periods=60).mean()

    out["ma_spread_signed"] = (ma10 - ma30) / ma30
    out["price_vs_ma60"] = (close - ma60) / ma60
    out["slope_120_signed"] = log_close.rolling(120, min_periods=120).apply(
        _ols_slope, raw=True
    )
    out["ret_sign_streak"] = _signed_streak(daily_ret, window=20)

    amt_mean = amt.rolling(252, min_periods=60).mean()
    amt_std = amt.rolling(252, min_periods=60).std()
    out["amt_trend"] = (amt - amt_mean) / amt_std.replace(0, np.nan)

    return out


def build_features_3state(df: pd.DataFrame) -> pd.DataFrame:
    """Merge original 6 features with directional features."""
    from features import build_features

    base = build_features(df)
    directional = build_directional_features(df)
    return base.merge(directional, on=["date", "close", "amt"], how="inner")
