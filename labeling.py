"""Trend-scanning labels (López de Prado) and optional changepoint variant."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def t_value_of_trend(prices: np.ndarray) -> float:
    """OLS t-statistic of slope for log-price segment (closed-form, fast)."""
    n = len(prices)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y = prices.astype(float)
    y_mean = y.mean()
    dx = x - x_mean
    ss_x = (dx * dx).sum()
    if ss_x <= 0:
        return 0.0
    b = (dx * (y - y_mean)).sum() / ss_x
    resid = y - (y_mean + b * dx)
    mse = (resid * resid).sum() / (n - 2)
    se_b = np.sqrt(mse / ss_x) if mse > 0 else 0.0
    return float(b / se_b) if se_b > 0 else 0.0


def _compute_raw_t_values(
    close: pd.Series,
    min_h: int,
    max_h: int,
) -> pd.DataFrame:
    """Compute best-window t_value per day (before tau threshold)."""
    logp = np.log(close.values)
    idx = close.index
    out = pd.DataFrame(index=idx, columns=["end_date", "t_value"])

    for i in range(len(close)):
        if i + max_h > len(close):
            continue
        ts: dict = {}
        for L in range(min_h, max_h + 1):
            ts[idx[i + L - 1]] = t_value_of_trend(logp[i : i + L])
        series = pd.Series(ts).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        best = series.abs().idxmax()
        out.iloc[i] = [idx[i + max_h - 1], float(series[best])]

    return out.dropna(subset=["t_value"])


def _apply_tau(raw: pd.DataFrame, tau: float) -> pd.DataFrame:
    result = raw.copy()
    result["label"] = np.where(
        result["t_value"].abs() < tau, 0, np.sign(result["t_value"])
    ).astype(int)
    result["abs_t"] = result["t_value"].abs()
    result["label_binary"] = (result["label"] != 0).astype(int)
    return result


def _labels_from_tau(
    close: pd.Series,
    min_h: int,
    max_h: int,
    tau: float,
) -> pd.DataFrame:
    """Core trend-scanning loop."""
    raw = _compute_raw_t_values(close, min_h, max_h)
    return _apply_tau(raw, tau)


def tune_tau(
    close: pd.Series,
    min_h: int = config.MIN_H,
    max_h: int = config.MAX_H,
    tau_start: float = config.TAU,
    target_range: tuple[float, float] = config.TAU_TARGET_RANGE,
    raw: pd.DataFrame | None = None,
) -> float:
    """Adjust tau so ranging share falls in target_range (no downstream tuning)."""
    raw = raw if raw is not None else _compute_raw_t_values(close, min_h, max_h)
    candidates = np.arange(0.5, 15.0, 0.1)
    best_tau = tau_start
    best_dist = np.inf

    for tau in candidates:
        labels = _apply_tau(raw, tau)
        if labels.empty:
            continue
        share = (labels["label"] == 0).mean()
        if target_range[0] <= share <= target_range[1]:
            return float(tau)
        mid = 0.5 * (target_range[0] + target_range[1])
        dist = abs(share - mid)
        if dist < best_dist:
            best_dist = dist
            best_tau = float(tau)

    return best_tau


def trend_scanning_labels(
    close: pd.Series,
    min_h: int = config.MIN_H,
    max_h: int = config.MAX_H,
    tau: float | None = None,
    auto_tune: bool = True,
) -> pd.DataFrame:
    """
    Generate three-state labels {-1, 0, +1} via trend-scanning on log(close).

    end_date = farthest future point (max_h horizon) for embargo.
    """
    if tau is None:
        tau = tune_tau(close, min_h, max_h) if auto_tune else config.TAU
    return _labels_from_tau(close, min_h, max_h, tau)


def _compute_backward_t_values(
    close: pd.Series,
    min_h: int,
    max_h: int,
) -> pd.DataFrame:
    """Best backward window ending at each day (contemporaneous regime)."""
    logp = np.log(close.values)
    idx = close.index
    out = pd.DataFrame(index=idx, columns=["t_value"])

    for i in range(len(close)):
        ts: dict = {}
        for L in range(min_h, max_h + 1):
            start = i - L + 1
            if start < 0:
                continue
            ts[idx[i]] = t_value_of_trend(logp[start : i + 1])
        series = pd.Series(ts).replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        best = series.abs().idxmax()
        out.iloc[i] = float(series[best])

    return out.dropna(subset=["t_value"])


def _shortest_forward_t_values(
    close: pd.Series,
    min_h: int,
    max_h: int,
    tau: float,
) -> pd.Series:
    """First forward window (shortest L) with |t| >= tau — reactive perfect foresight."""
    logp = np.log(close.values)
    idx = close.index
    labels = pd.Series(0, index=idx, dtype=int)

    for i in range(len(close)):
        for L in range(min_h, max_h + 1):
            if i + L > len(close):
                break
            tv = t_value_of_trend(logp[i : i + L])
            if abs(tv) >= tau:
                labels.iloc[i] = int(np.sign(tv))
                break
    return labels


def build_perfect_trade_labels(
    close: pd.Series,
    min_h: int,
    max_h: int,
    tau: float,
) -> pd.DataFrame:
    """
    Conservative perfect-trade labels for ceiling / mapping validation.

    Combines backward (contemporaneous) and shortest-forward (reactive) labels:
    - Either says down  → down
    - Both say up       → up
    - Otherwise         → ranging
    """
    backward = backward_trend_scanning_labels(close, min_h, max_h, tau=tau, auto_tune=False)
    shortest = _shortest_forward_t_values(close, min_h, max_h, tau)
    bw = backward["label_perfect"].reindex(close.index).fillna(0).astype(int)
    sf = shortest.reindex(close.index).fillna(0).astype(int)

    combined = pd.Series(0, index=close.index, dtype=int)
    for dt in close.index:
        bi, si = int(bw.loc[dt]), int(sf.loc[dt])
        if bi == -1 or si == -1:
            combined.loc[dt] = -1
        elif bi == 1 and si == 1:
            combined.loc[dt] = 1
        else:
            combined.loc[dt] = 0

    return pd.DataFrame(
        {
            "label_perfect": combined.values,
            "label_perfect_binary": (combined != 0).astype(int).values,
        },
        index=close.index,
    )


def backward_trend_scanning_labels(
    close: pd.Series,
    min_h: int = config.MIN_H,
    max_h: int = config.MAX_H,
    tau: float | None = None,
    auto_tune: bool = True,
) -> pd.DataFrame:
    """Backward-looking three-state labels for perfect-regime trading ceiling."""
    raw = _compute_backward_t_values(close, min_h, max_h)
    if tau is None:
        tau = tune_tau(close, min_h, max_h, raw=raw) if auto_tune else config.TAU
    result = _apply_tau(raw, tau)
    result = result.rename(columns={"label": "label_perfect", "label_binary": "label_perfect_binary"})
    return result


def changepoint_labels(
    close: pd.Series,
    tau: float,
    penalty: str = config.CHANGEPOINT_PENALTY,
) -> pd.DataFrame:
    """PELT changepoint variant: segment log(close), label each segment."""
    import ruptures as rpt

    logp = np.log(close.values).reshape(-1, 1)
    algo = rpt.Pelt(model="linear", min_size=config.MIN_H).fit(logp)
    bkps = algo.predict(pen=penalty)
    if bkps and bkps[-1] == len(close):
        bkps = bkps[:-1]

    segments = [0] + list(bkps) + [len(close)]
    labels = np.zeros(len(close), dtype=int)
    t_values = np.zeros(len(close))
    end_dates = pd.Series(index=close.index, dtype="datetime64[ns]")

    for s, e in zip(segments[:-1], segments[1:]):
        seg = logp[s:e, 0]
        if len(seg) < config.MIN_H:
            lbl, tv = 0, 0.0
        else:
            tv = t_value_of_trend(seg)
            lbl = 0 if abs(tv) < tau else int(np.sign(tv))
        labels[s:e] = lbl
        t_values[s:e] = tv
        end_dates.iloc[s:e] = close.index[e - 1]

    out = pd.DataFrame(
        {
            "end_date": end_dates.values,
            "t_value": t_values,
            "label": labels,
        },
        index=close.index,
    )
    out["abs_t"] = out["t_value"].abs()
    out["label_binary"] = (out["label"] != 0).astype(int)
    return out.dropna(subset=["label"])


def build_labeled_dataframe(
    df: pd.DataFrame,
    use_changepoint: bool = config.USE_CHANGEPOINT,
    min_h: int = config.MIN_H,
    max_h: int = config.MAX_H,
    tau: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Merge price data with trend-scanning labels (+ backward perfect-trade labels)."""
    close = df.set_index("date")["close"]
    if use_changepoint:
        if tau is None:
            tau = tune_tau(close, min_h, max_h)
        labels = changepoint_labels(close, tau=tau)
    else:
        raw = _compute_raw_t_values(close, min_h, max_h)
        if tau is None:
            tau = tune_tau(close, min_h, max_h, raw=raw)
        labels = _apply_tau(raw, tau)

    perfect = build_perfect_trade_labels(close, min_h, max_h, tau=tau)
    merged = df.set_index("date").join(labels, how="inner")
    merged = merged.join(perfect, how="left")
    merged = merged.reset_index()
    return merged, float(tau)
