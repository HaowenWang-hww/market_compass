"""Zigzag + Binseg binary labeling (reference report replication)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt

import config
from labeling import t_value_of_trend


def _ols_slope(logp: np.ndarray) -> float:
    if len(logp) < 2:
        return 0.0
    x = np.arange(len(logp), dtype=float)
    return float(np.polyfit(x, logp, 1)[0])


def detect_zigzag_pivots(
    close: pd.Series,
    reversal: float = config.ZIGZAG_REVERSAL,
) -> list[tuple[int, float, str]]:
    """Return pivot list: (index, price, 'peak'|'trough')."""
    prices = close.values.astype(float)
    n = len(prices)
    if n < 2:
        return [(0, prices[0], "trough")]

    pivots: list[tuple[int, float, str]] = [(0, prices[0], "trough")]
    mode: str | None = None
    candidate_idx = 0
    candidate_price = prices[0]

    for i in range(1, n):
        p = prices[i]
        if mode is None:
            if p >= prices[0] * (1.0 + reversal):
                mode = "high"
                candidate_idx = i
                candidate_price = p
            elif p <= prices[0] * (1.0 - reversal):
                mode = "low"
                candidate_idx = i
                candidate_price = p
        elif mode == "high":
            if p > candidate_price:
                candidate_idx = i
                candidate_price = p
            elif p <= candidate_price * (1.0 - reversal):
                pivots.append((candidate_idx, candidate_price, "peak"))
                mode = "low"
                candidate_idx = i
                candidate_price = p
        else:
            if p < candidate_price:
                candidate_idx = i
                candidate_price = p
            elif p >= candidate_price * (1.0 + reversal):
                pivots.append((candidate_idx, candidate_price, "trough"))
                mode = "high"
                candidate_idx = i
                candidate_price = p

    last_type = pivots[-1][2]
    if last_type == "peak":
        pivots.append((candidate_idx, candidate_price, "trough"))
    else:
        pivots.append((candidate_idx, candidate_price, "peak"))
    return pivots


def _segment_trend_label(
    start_idx: int,
    end_idx: int,
    close: pd.Series,
    min_ann_ret: float,
    min_duration: int,
) -> int:
    """1 if leg qualifies as trend, else 0."""
    duration = end_idx - start_idx
    if duration < min_duration:
        return 0
    start_p = close.iloc[start_idx]
    end_p = close.iloc[end_idx]
    if start_p <= 0:
        return 0
    total_ret = end_p / start_p - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / duration) - 1.0
    return 1 if abs(ann_ret) >= min_ann_ret else 0


def _binseg_refine_segment(
    labels: np.ndarray,
    close: pd.Series,
    seg_start: int,
    seg_end: int,
    min_duration: int,
    slope_ratio_thresh: float,
) -> None:
    """Relabel trend exhaustion within a trend segment to ranging (0)."""
    seg_len = seg_end - seg_start
    if seg_len < 2 * min_duration:
        return

    logp = np.log(close.iloc[seg_start:seg_end].values.astype(float))
    min_size = max(min_duration // 2, config.MIN_H)
    try:
        algo = rpt.Binseg(model="linear", min_size=min_size).fit(logp.reshape(-1, 1))
        bkps = algo.predict(n_bkps=1)
    except Exception:
        return

    if not bkps or bkps[0] >= seg_len:
        return

    cp = bkps[0]
    if cp < min_size or seg_len - cp < min_size:
        return

    pre_slope = _ols_slope(logp[:cp])
    post_slope = _ols_slope(logp[cp:])
    if abs(pre_slope) < 1e-12:
        return

    if abs(post_slope) < abs(pre_slope) * slope_ratio_thresh:
        abs_start = seg_start + cp
        labels[abs_start:seg_end] = 0


def zigzag_binary_labels(
    close: pd.Series,
    reversal: float = config.ZIGZAG_REVERSAL,
    min_annualized_return: float = config.ZIGZAG_MIN_ANNUALIZED_RETURN,
    min_duration: int = config.ZIGZAG_MIN_DURATION,
    slope_ratio_thresh: float = config.ZIGZAG_BINSEG_SLOPE_RATIO,
) -> pd.DataFrame:
    """
    Two-phase zigzag + Binseg labeling.

    Returns DataFrame indexed by date with zigzag_label_binary in {0, 1}.
    """
    pivots = detect_zigzag_pivots(close, reversal=reversal)
    labels = np.zeros(len(close), dtype=int)

    for (s_idx, _, _), (e_idx, _, _) in zip(pivots[:-1], pivots[1:]):
        lbl = _segment_trend_label(
            s_idx, e_idx, close, min_annualized_return, min_duration
        )
        labels[s_idx:e_idx] = lbl

    i = 0
    while i < len(labels):
        if labels[i] != 1:
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] == 1:
            j += 1
        _binseg_refine_segment(labels, close, i, j, min_duration, slope_ratio_thresh)
        i = j

    out = pd.DataFrame(
        {
            "zigzag_label_binary": labels,
            "label_binary": labels,
        },
        index=close.index,
    )
    return out


def build_zigzag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Attach zigzag binary labels to price data."""
    close = df.set_index("date")["close"]
    zz = zigzag_binary_labels(close)
    merged = df.set_index("date").join(zz, how="inner").reset_index()
    return merged
