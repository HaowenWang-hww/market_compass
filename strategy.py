"""Strategy backtesting with explicit signal lag and two comparison tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

import config


@dataclass
class StrategyMetrics:
    name: str
    total_return: float
    annual_return: float
    annual_vol: float
    max_drawdown: float
    sharpe: float
    excess_return: float
    excess_sharpe: float


def _annualize_return(total_ret: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (1 + total_ret) ** (252 / n_days) - 1


def _compute_metrics(
    nav: pd.Series,
    name: str,
    benchmark_nav: pd.Series,
) -> StrategyMetrics:
    rets = nav.pct_change().dropna()
    bench_rets = benchmark_nav.pct_change().dropna()
    aligned = pd.concat([rets, bench_rets], axis=1, join="inner").dropna()
    if aligned.empty:
        return StrategyMetrics(name, 0, 0, 0, 0, 0, 0, 0)
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(rets)
    annual_return = _annualize_return(total_return, n_days)
    annual_vol = r.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    cummax = nav.cummax()
    max_dd = ((nav - cummax) / cummax).min()

    bench_total = benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1
    bench_annual = _annualize_return(bench_total, n_days)
    bench_vol = b.std() * np.sqrt(252)
    excess_vol = (r - b).std() * np.sqrt(252)
    excess_sharpe = (annual_return - bench_annual) / excess_vol if excess_vol > 0 else 0.0

    return StrategyMetrics(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        annual_vol=annual_vol,
        max_drawdown=max_dd,
        sharpe=sharpe,
        excess_return=total_return - bench_total,
        excess_sharpe=excess_sharpe,
    )


def signal_to_target_position(
    signal: pd.Series,
    mode: Literal["direct", "binary_replicate"],
    initial_pos: float = 0.5,
) -> pd.Series:
    """Map signal to target position (before execution lag)."""
    target = pd.Series(initial_pos, index=signal.index, dtype=float)

    if mode == "direct":
        for dt, lbl in signal.items():
            if pd.isna(lbl):
                continue
            lbl = int(lbl)
            if lbl == 1:
                target.loc[dt] = 1.0
            elif lbl == -1:
                target.loc[dt] = 0.0
            else:
                target.loc[dt] = 0.5
    else:
        for dt, lbl in signal.items():
            if pd.isna(lbl):
                continue
            target.loc[dt] = 0.5  # ranging baseline; replicate adjusts on rebalance

    return target


def apply_execution_lag(
    target_position: pd.Series,
    initial_pos: float = 0.5,
) -> pd.Series:
    """Signal at t executes at t+1 (close-to-close)."""
    return target_position.shift(1).fillna(initial_pos)


def _apply_costs(
    executed_position: pd.Series,
    close: pd.Series,
    cost_bp: float = config.COST_BP,
) -> pd.Series:
    """Portfolio NAV from executed positions (already lagged)."""
    turnover = executed_position.diff().abs().fillna(executed_position.iloc[0])
    daily_cost = turnover * (cost_bp / 10000)
    price_rets = close.pct_change().fillna(0)
    strat_rets = executed_position * price_rets - daily_cost
    return (1 + strat_rets).cumprod()


def replicate_strategy(
    df: pd.DataFrame,
    signal_col: str,
    name: str,
    initial_pos: float = 0.5,
    cost_bp: float = config.COST_BP,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Replicate report strategy on binary signal (1=trend, 0=ranging).
    Rebalance every Wednesday; momentum in trend, reversal in ranging.
    """
    work = df.dropna(subset=[signal_col, "close"]).copy().set_index("date")
    close = work["close"]
    signal = work[signal_col]

    target = pd.Series(initial_pos, index=close.index, dtype=float)
    last_rebal_idx = 0
    rebal_dates = close.index[close.index.dayofweek == 2]

    for i, dt in enumerate(close.index):
        if dt not in rebal_dates or i == 0:
            continue
        lookback_ret = close.iloc[i - 1] / close.iloc[last_rebal_idx] - 1
        last_rebal_idx = i

        regime = signal.loc[dt]
        if pd.isna(regime):
            regime = 0
        regime = int(regime)

        pos = target.iloc[i - 1]
        trade_units = min(abs(lookback_ret / 0.01) * 0.1, 1.0)

        if regime == 1:
            pos += trade_units if lookback_ret > 0 else -trade_units
        else:
            pos += -trade_units if lookback_ret > 0 else trade_units

        pos = float(np.clip(pos, 0, 1))
        target.iloc[i:] = pos

    executed = apply_execution_lag(target, initial_pos)
    nav = _apply_costs(executed, close, cost_bp)
    return nav, executed, target


def direct_mapping_strategy(
    df: pd.DataFrame,
    signal_col: str,
    name: str,
    initial_pos: float = 0.5,
    cost_bp: float = config.COST_BP,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """3-state direct mapping: +1→1.0, -1→0.0, 0→0.5; execute next day."""
    work = df.dropna(subset=[signal_col, "close"]).copy().set_index("date")
    close = work["close"]
    signal = work[signal_col]

    target = signal_to_target_position(signal, mode="direct", initial_pos=initial_pos)
    executed = apply_execution_lag(target, initial_pos)
    nav = _apply_costs(executed, close, cost_bp)
    return nav, executed, target


def buy_hold_benchmark(
    df: pd.DataFrame,
    initial_pos: float = 0.5,
    cost_bp: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = df.dropna(subset=["close"]).copy().set_index("date")
    close = work["close"]
    target = pd.Series(initial_pos, index=close.index, dtype=float)
    executed = apply_execution_lag(target, initial_pos)
    nav = _apply_costs(executed, close, cost_bp)
    return nav, executed, target


def pure_momentum_strategy(
    df: pd.DataFrame,
    name: str = "纯动量",
    lookback: int = 20,
    cost_bp: float = config.COST_BP,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = df.dropna(subset=["close"]).copy().set_index("date")
    close = work["close"]
    mom = close.pct_change(lookback)
    target = pd.Series(0.5, index=close.index, dtype=float)
    target[mom > 0] = 1.0
    target[mom < 0] = 0.0
    executed = apply_execution_lag(target, 0.5)
    nav = _apply_costs(executed, close, cost_bp)
    return nav, executed, target


def pure_reversal_strategy(
    df: pd.DataFrame,
    name: str = "纯反转",
    lookback: int = 20,
    cost_bp: float = config.COST_BP,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = df.dropna(subset=["close"]).copy().set_index("date")
    close = work["close"]
    mom = close.pct_change(lookback)
    target = pd.Series(0.5, index=close.index, dtype=float)
    target[mom > 0] = 0.0
    target[mom < 0] = 1.0
    executed = apply_execution_lag(target, 0.5)
    nav = _apply_costs(executed, close, cost_bp)
    return nav, executed, target


def _run_strategy_spec(
    bt: pd.DataFrame,
    bench_nav: pd.Series,
    name: str,
    col: str,
    mode: Literal["replicate", "direct"],
) -> tuple[StrategyMetrics, pd.Series, pd.Series] | None:
    if col not in bt.columns:
        return None
    try:
        if mode == "replicate":
            nav, executed, _ = replicate_strategy(bt, col, name)
        else:
            nav, executed, _ = direct_mapping_strategy(bt, col, name)
        return _compute_metrics(nav, name, bench_nav), nav, executed
    except Exception:
        return None


def run_track_a(
    zigzag_labeled: pd.DataFrame,
    trendscan_labeled: pd.DataFrame,
    zigzag_preds: pd.DataFrame,
    trendscan_preds: pd.DataFrame,
    backtest_start: str = config.BACKTEST_START,
) -> dict:
    """Fair comparison: zigzag vs trend-scanning binary labels, same replicate rules."""
    bt_start = pd.Timestamp(backtest_start)
    results: dict[str, dict[str, StrategyMetrics]] = {"zigzag": {}, "trendscan": {}}
    navs: dict[str, pd.Series] = {}
    positions: dict[str, pd.Series] = {}

    bench_nav, _, _ = buy_hold_benchmark(
        trendscan_labeled[trendscan_labeled["date"] >= bt_start]
    )
    results["zigzag"]["基准(0.5买入持有)"] = _compute_metrics(
        bench_nav, "基准(0.5买入持有)", bench_nav
    )
    results["trendscan"]["基准(0.5买入持有)"] = results["zigzag"]["基准(0.5买入持有)"]
    navs["基准(0.5买入持有)"] = bench_nav

    specs = [
        ("完美标签", "label_binary", "replicate"),
        ("等权模型", "pred_eq_binary", "replicate"),
        ("等权模型(平滑)", "pred_eq_smooth_binary", "replicate"),
        ("逻辑回归", "pred_lr_binary", "replicate"),
        ("逻辑回归(平滑)", "pred_lr_smooth_binary", "replicate"),
        ("决策树", "pred_dt_binary", "replicate"),
        ("决策树(平滑)", "pred_dt_smooth_binary", "replicate"),
    ]

    for method, labeled, preds in [
        ("zigzag", zigzag_labeled, zigzag_preds),
        ("trendscan", trendscan_labeled, trendscan_preds),
    ]:
        bt = preds[preds["date"] >= bt_start].copy()
        lbl = labeled[labeled["date"] >= bt_start][["date", "label_binary"]]
        if "label_binary" not in bt.columns:
            bt = bt.merge(lbl, on="date", how="left", suffixes=("", "_lbl"))
            if "label_binary_lbl" in bt.columns:
                bt["label_binary"] = bt["label_binary"].fillna(bt["label_binary_lbl"])
                bt = bt.drop(columns=["label_binary_lbl"])

        for strat_name, col, mode in specs:
            full_name = f"{method}_{strat_name}"
            out = _run_strategy_spec(bt, bench_nav, full_name, col, mode)
            if out is None:
                continue
            metrics, nav, pos = out
            results[method][strat_name] = metrics
            navs[full_name] = nav
            positions[full_name] = pos

        bt_base = labeled[labeled["date"] >= bt_start].copy()
        for baseline_name, fn in [
            ("纯动量", pure_momentum_strategy),
            ("纯反转", pure_reversal_strategy),
        ]:
            nav, pos, _ = fn(bt_base, name=f"{method}_{baseline_name}")
            full_name = f"{method}_{baseline_name}"
            results[method][baseline_name] = _compute_metrics(nav, full_name, bench_nav)
            navs[full_name] = nav
            positions[full_name] = pos

    return {"metrics": results, "navs": navs, "positions": positions, "benchmark": bench_nav}


def run_track_b(
    pred_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    backtest_start: str = config.BACKTEST_START,
) -> dict:
    """3-state system: directional features + direct mapping."""
    bt_start = pd.Timestamp(backtest_start)
    bt = pred_df[pred_df["date"] >= bt_start].copy()
    lbl = labeled_df[labeled_df["date"] >= bt_start]

    bench_nav, _, _ = buy_hold_benchmark(bt)
    results: dict[str, StrategyMetrics] = {}
    navs: dict[str, pd.Series] = {}
    positions: dict[str, pd.Series] = {}

    results["基准(0.5买入持有)"] = _compute_metrics(bench_nav, "基准(0.5买入持有)", bench_nav)
    navs["基准(0.5买入持有)"] = bench_nav

    if "label" not in bt.columns:
        bt = bt.merge(lbl[["date", "label", "label_perfect"]], on="date", how="left")
    elif "label_perfect" not in bt.columns:
        bt = bt.merge(lbl[["date", "label_perfect"]], on="date", how="left")

    specs = [
        ("完美标签(三态直接映射)", "label_perfect", "direct"),
        ("等权(三态直接映射)", "pred_eq", "direct"),
        ("等权(三态平滑)", "pred_eq_smooth", "direct"),
        ("逻辑回归(三态直接映射)", "pred_lr", "direct"),
        ("逻辑回归(三态平滑)", "pred_lr_smooth", "direct"),
        ("决策树(三态直接映射)", "pred_dt", "direct"),
        ("决策树(三态平滑)", "pred_dt_smooth", "direct"),
    ]

    for strat_name, col, mode in specs:
        out = _run_strategy_spec(bt, bench_nav, strat_name, col, mode)
        if out is None:
            continue
        metrics, nav, pos = out
        results[strat_name] = metrics
        navs[strat_name] = nav
        positions[strat_name] = pos

    for baseline_name, fn in [("纯动量", pure_momentum_strategy), ("纯反转", pure_reversal_strategy)]:
        nav, pos, _ = fn(bt, name=baseline_name)
        results[baseline_name] = _compute_metrics(nav, baseline_name, bench_nav)
        navs[baseline_name] = nav
        positions[baseline_name] = pos

    return {"metrics": results, "navs": navs, "positions": positions, "benchmark": bench_nav}
