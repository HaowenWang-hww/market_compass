"""Self-check assertions run at end of pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from features_directional import DIRECTIONAL_FEATURE_COLS, FEATURE_COLS_3STATE, build_features_3state
from models import ModelResult, SplitResult, temporal_split_with_purge
from strategy import StrategyMetrics, direct_mapping_strategy


class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        return f"[{icon}] [{self.name}] {self.message}"


def check_perfect_label_dominance(
    track_b_metrics: dict[str, StrategyMetrics],
    benchmark_name: str = "基准(0.5买入持有)",
    perfect_name: str = "完美标签(三态直接映射)",
    sharpe_margin: float = 0.1,
) -> ValidationResult:
    bench = track_b_metrics.get(benchmark_name)
    perfect = track_b_metrics.get(perfect_name)
    if bench is None or perfect is None:
        return ValidationResult(
            "完美标签支配基准",
            False,
            "缺少基准或完美标签策略指标",
        )

    mdd_ok = abs(perfect.max_drawdown) < abs(bench.max_drawdown)
    sharpe_ok = perfect.sharpe > bench.sharpe + sharpe_margin
    passed = mdd_ok and sharpe_ok

    msg = (
        f"完美 MDD={perfect.max_drawdown*100:.2f}% vs 基准 {bench.max_drawdown*100:.2f}% | "
        f"夏普 {perfect.sharpe:.2f} vs 基准 {bench.sharpe:.2f} (需 > {bench.sharpe + sharpe_margin:.2f})"
    )
    if not passed:
        msg = "WARN 完美标签未支配基准，仓位映射或对齐存在 bug — " + msg
    return ValidationResult("完美标签支配基准", passed, msg)


def check_recall_down(model_results: dict) -> ValidationResult:
    recalls = []
    for key in ["logistic", "decision_tree", "equal_weight"]:
        r: ModelResult = model_results.get(key)
        if r and r.task == "3state":
            recalls.append(r.recall_per_class.get(-1, 0.0))
    best = max(recalls) if recalls else 0.0
    passed = best > 0.05
    return ValidationResult(
        "三态 recall_下降",
        passed,
        f"最佳 recall_下降={best:.4f}（需 > 0）",
    )


def check_feature_causality(df: pd.DataFrame, n_samples: int = 5) -> ValidationResult:
    """Spot-check that features at t do not change when future prices change."""
    feat = build_features_3state(df)
    rng = np.random.default_rng(config.RANDOM_STATE)
    valid_idx = feat.dropna(subset=FEATURE_COLS_3STATE).index.to_numpy()
    if len(valid_idx) < n_samples + 10:
        return ValidationResult("特征因果性", False, "样本不足")

    sample_idx = rng.choice(valid_idx[60:-10], size=n_samples, replace=False)
    violations = 0
    for idx in sample_idx:
        row = feat.loc[idx, FEATURE_COLS_3STATE].astype(float)
        if row.isna().any():
            continue
        truncated = df.iloc[: idx + 1]
        refeat = build_features_3state(truncated)
        if idx not in refeat.index:
            continue
        new_row = refeat.loc[idx, FEATURE_COLS_3STATE].astype(float)
        if not np.allclose(row.values, new_row.values, equal_nan=True, rtol=1e-5, atol=1e-6):
            violations += 1

    passed = violations == 0
    return ValidationResult(
        "特征因果性",
        passed,
        f"抽查 {n_samples} 点，违规 {violations} 处",
    )


def check_embargo(split: SplitResult) -> ValidationResult:
    embargo = config.MAX_H
    passed = True
    msg_parts = [f"embargo={embargo}"]
    if len(split.train) == 0 or len(split.test) == 0:
        return ValidationResult("Train/Test Embargo", False, "训练或测试集为空")

    gap_days = (split.test_start - split.train_end).days
    if gap_days < embargo:
        passed = False
        msg_parts.append(f"间隔仅 {gap_days} 天 (< {embargo})")
    else:
        msg_parts.append(f"purge+embargo 间隔 {gap_days} 天 OK")
    return ValidationResult("Train/Test Embargo", passed, " | ".join(msg_parts))


def check_same_backtest_universe(
    zigzag_labeled: pd.DataFrame,
    trendscan_labeled: pd.DataFrame,
    backtest_start: str = config.BACKTEST_START,
) -> ValidationResult:
    bt = pd.Timestamp(backtest_start)
    zz = zigzag_labeled[zigzag_labeled["date"] >= bt][["date", "close", "amt"]]
    ts = trendscan_labeled[trendscan_labeled["date"] >= bt][["date", "close", "amt"]]
    common = zz.merge(ts, on="date", suffixes=("_zz", "_ts"))
    same_close = np.allclose(common["close_zz"].values, common["close_ts"].values, rtol=1e-9)
    passed = len(common) > 0 and same_close
    msg = (
        f"共同交易日={len(common)}, zigzag={len(zz)}, trendscan={len(ts)}, "
        f"收盘价一致={same_close}, 区间={bt.date()}~{common['date'].max().date() if len(common) else 'N/A'}, "
        f"成本={config.COST_BP}bp"
    )
    return ValidationResult("同数据同区间对照", passed, msg)


def plot_perfect_label_diagnostic(
    labeled_df: pd.DataFrame,
    output_dir: Path,
    backtest_start: str = config.BACKTEST_START,
) -> None:
    import matplotlib.pyplot as plt

    bt = labeled_df[labeled_df["date"] >= pd.Timestamp(backtest_start)].copy()
    nav, executed, _ = direct_mapping_strategy(bt, "label_perfect", "完美标签诊断")
    work = bt.set_index("date")

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(work.index, work["close"], color="black", alpha=0.7, label="收盘价")
    ax1.set_ylabel("收盘价")
    ax2 = ax1.twinx()
    ax2.fill_between(executed.index, 0, executed.values, alpha=0.35, color="steelblue")
    ax2.set_ylabel("执行仓位")
    ax2.set_ylim(-0.05, 1.05)
    ax1.set_title("完美三态标签：执行仓位 vs 收盘价（自检图）")
    for yr in [2022, 2024]:
        ax1.axvspan(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-06-30"), alpha=0.08, color="red")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_perfect_label_position.png", dpi=config.FIG_DPI)
    plt.close(fig)


def run_all_validations(
    track_b_metrics: dict[str, StrategyMetrics],
    model_results_3state: dict,
    split_3state: SplitResult,
    zigzag_labeled: pd.DataFrame,
    trendscan_labeled: pd.DataFrame,
    df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    output_dir: Path,
) -> list[ValidationResult]:
    plot_perfect_label_diagnostic(labeled_df, output_dir)
    results = [
        check_perfect_label_dominance(track_b_metrics),
        check_recall_down(model_results_3state),
        check_feature_causality(df),
        check_embargo(split_3state),
        check_same_backtest_universe(zigzag_labeled, trendscan_labeled),
    ]
    lines = ["# 自检断言", ""]
    for r in results:
        icon = "✅" if r.passed else "❌"
        line = f"{icon} [{r.name}] {r.message}"
        print(str(r))
        lines.append(line)
    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return results
