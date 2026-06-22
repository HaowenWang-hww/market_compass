"""Six causal price-volume features (replicating reference report semantics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "Feature_MA_1030",
    "Feature_MA_0510",
    "Feature_price_120",
    "Feature_Volume",
    "Feature_Volatility_past",
    "Feature_Squeeze_Breakout",
]


def _consecutive_relation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """1 if A strictly above B or strictly below B for entire past window."""
    above = (series_a > series_b).astype(int)
    below = (series_a < series_b).astype(int)
    above_run = above.rolling(window, min_periods=window).min()
    below_run = below.rolling(window, min_periods=window).min()
    return ((above_run == 1) | (below_run == 1)).astype(int)


def _bollinger_width(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (upper - lower) / ma


def _trailing_expanding_percentile(
    series: pd.Series, pct: float, lookback: int = 252
) -> pd.Series:
    """Causal percentile: use trailing lookback window only."""
    return series.rolling(lookback, min_periods=60).apply(
        lambda x: np.nanpercentile(x, pct * 100), raw=True
    )


def _trailing_expanding_median(series: pd.Series, lookback: int = 252) -> pd.Series:
    return series.rolling(lookback, min_periods=60).median()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 6 features using only information <= t."""
    out = df[["date", "close", "amt"]].copy()
    close = out["close"]
    amt = out["amt"]

    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma30 = close.rolling(30, min_periods=30).mean()

    out["Feature_MA_1030"] = _consecutive_relation(ma10, ma30, 40)
    out["Feature_MA_0510"] = _consecutive_relation(ma5, ma10, 20)

    # 120-day price slope (causal OLS slope per day)
    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0]

    slope_120 = close.rolling(120, min_periods=120).apply(_slope, raw=True)
    slope_abs = slope_120.abs()
    slope_p80 = _trailing_expanding_percentile(slope_abs, 0.80)
    out["Feature_price_120"] = (slope_abs <= slope_p80).astype(int)

    amt_p70 = _trailing_expanding_percentile(amt, 0.70)
    out["Feature_Volume"] = (amt.shift(1) <= amt_p70.shift(1)).astype(int)

    bb_width = _bollinger_width(close, 20, 2.0)
    bb_mean20 = bb_width.rolling(20, min_periods=20).mean()
    bb_p80 = _trailing_expanding_percentile(bb_mean20, 0.80)
    out["Feature_Volatility_past"] = (bb_mean20 <= bb_p80).astype(int)

    bb_median = _trailing_expanding_median(bb_width)
    compressed = bb_width < bb_median

    ma_bb = close.rolling(20, min_periods=20).mean()
    std_bb = close.rolling(20, min_periods=20).std()
    upper = ma_bb + 2 * std_bb
    lower = ma_bb - 2 * std_bb
    breakout = (close > upper) | (close < lower)
    breakout_count = breakout.rolling(30, min_periods=30).sum()
    out["Feature_Squeeze_Breakout"] = (
        compressed & (breakout_count > 1)
    ).astype(int)

    return out
