"""Generate figures, tables, and markdown report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from models import ModelResult
from strategy import StrategyMetrics
from validate import ValidationResult

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_num(v: float) -> str:
    return f"{v:.2f}"


def plot_confusion_matrix(result: ModelResult, output_dir: Path, suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if result.task == "binary":
        labels = ["震荡(0)", "趋势(1)"]
    else:
        labels = ["下降(-1)", "震荡(0)", "上升(+1)"]
    sns.heatmap(
        result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title(f"{result.name} 混淆矩阵{suffix}")
    fig.tight_layout()
    safe_name = result.name.replace("/", "_")
    fig.savefig(output_dir / f"fig_cm_{safe_name}{suffix}.png", dpi=config.FIG_DPI)
    plt.close(fig)


def plot_nav_curves(navs: dict[str, pd.Series], output_dir: Path, filename: str = "fig_nav_curves.png") -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, nav in navs.items():
        ax.plot(nav.index, nav.values, label=name, linewidth=1.2)
    ax.set_title("策略净值曲线")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=config.FIG_DPI)
    plt.close(fig)


def strategy_summary_table(metrics: dict[str, StrategyMetrics]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append({
            "策略": name,
            "总收益": m.total_return,
            "年化收益": m.annual_return,
            "年化波动": m.annual_vol,
            "最大回撤": m.max_drawdown,
            "夏普": m.sharpe,
            "超额收益": m.excess_return,
            "超额夏普": m.excess_sharpe,
        })
    return pd.DataFrame(rows)


def fair_comparison_table(track_a: dict) -> pd.DataFrame:
    """Side-by-side zigzag vs trendscan on same strategies."""
    rows = []
    zigzag = track_a["metrics"]["zigzag"]
    trendscan = track_a["metrics"]["trendscan"]
    all_names = sorted(set(zigzag) | set(trendscan))
    for name in all_names:
        z = zigzag.get(name)
        t = trendscan.get(name)
        rows.append({
            "策略": name,
            "zigzag_总收益": z.total_return if z else np.nan,
            "zigzag_夏普": z.sharpe if z else np.nan,
            "zigzag_回撤": z.max_drawdown if z else np.nan,
            "trendscan_总收益": t.total_return if t else np.nan,
            "trendscan_夏普": t.sharpe if t else np.nan,
            "trendscan_回撤": t.max_drawdown if t else np.nan,
        })
    return pd.DataFrame(rows)


def model_summary_table(model_results: dict, task: str = "3state") -> pd.DataFrame:
    rows = []
    for key in ["equal_weight", "logistic", "decision_tree"]:
        r: ModelResult = model_results[key]
        for smooth, preds in [("原始", r.y_pred), ("平滑", r.y_pred_smooth)]:
            from sklearn.metrics import accuracy_score, f1_score, recall_score

            if task == "binary":
                labels = [0, 1]
                recalls = recall_score(y_true=r.y_true, y_pred=preds, labels=labels, average=None, zero_division=0)
                row = {
                    "模型": r.name,
                    "平滑": smooth,
                    "准确率": accuracy_score(r.y_true, preds),
                    "macro_F1": f1_score(r.y_true, preds, average="macro", labels=labels, zero_division=0),
                    "recall_震荡": recalls[0],
                    "recall_趋势": recalls[1],
                }
            else:
                labels = [-1, 0, 1]
                recalls = recall_score(y_true=r.y_true, y_pred=preds, labels=labels, average=None, zero_division=0)
                row = {
                    "模型": r.name,
                    "平滑": smooth,
                    "准确率": accuracy_score(r.y_true, preds),
                    "macro_F1": f1_score(r.y_true, preds, average="macro", labels=labels, zero_division=0),
                    "recall_下降": recalls[0],
                    "recall_震荡": recalls[1],
                    "recall_上升": recalls[2],
                }
            rows.append(row)
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame, path: Path, pct_cols: list[str] | None = None) -> None:
    pct_cols = pct_cols or []
    display = df.copy()
    for col in display.columns:
        if col in pct_cols:
            display[col] = display[col].apply(lambda x: f"{x*100:.2f}%")
        elif display[col].dtype in [np.float64, float]:
            display[col] = display[col].apply(lambda x: f"{x:.2f}")
    path.write_text(display.to_markdown(index=False), encoding="utf-8")


def save_fair_comparison(track_a: dict, output_dir: Path) -> pd.DataFrame:
    df = fair_comparison_table(track_a)
    df.to_csv(output_dir / "table_fair_comparison.csv", index=False, float_format="%.4f")
    pct_cols = [c for c in df.columns if "收益" in c or "回撤" in c]
    _df_to_md(df, output_dir / "table_fair_comparison.md", pct_cols=pct_cols)
    return df


def save_track_b(track_b: dict, output_dir: Path) -> pd.DataFrame:
    df = strategy_summary_table(track_b["metrics"])
    df.to_csv(output_dir / "table_track_B.csv", index=False, float_format="%.4f")
    _df_to_md(
        df,
        output_dir / "table_track_B.md",
        pct_cols=["总收益", "年化收益", "年化波动", "最大回撤", "超额收益"],
    )
    return df


def save_model_tables(
    model_binary: dict,
    model_3state: dict,
    output_dir: Path,
) -> None:
    for name, results, task in [
        ("binary", model_binary, "binary"),
        ("3state", model_3state, "3state"),
    ]:
        df = model_summary_table(results, task=task)
        df.to_csv(output_dir / f"table_model_{name}.csv", index=False, float_format="%.4f")


def generate_report(
    tau: float,
    label_dist: pd.DataFrame,
    health: dict,
    fair_df: pd.DataFrame,
    track_b_df: pd.DataFrame,
    model_3state: dict,
    validations: list[ValidationResult],
    output_dir: Path,
) -> None:
    ranging_share = label_dist[label_dist["label"] == 0]["share"].values
    ranging_pct = ranging_share[0] * 100 if len(ranging_share) else 0
    ceiling = health.get("ceiling", {})
    perfect_3 = ceiling.get("完美三态直接映射")
    model_df = model_summary_table(model_3state, task="3state")

    val_lines = [str(v) for v in validations]
    all_passed = all(v.passed for v in validations)

    lines = [
        "# Market Compass — 修订版研究报告",
        "",
        "> **评估准则：策略夏普比率为最终判据。** 对照均在**同一份万得全A、同一回测区间、同一成本**上完成。",
        "",
        "## 1. 方法概述",
        "",
        f"- Trend-Scanning：`min_h={config.MIN_H}`, `max_h={config.MAX_H}`, `tau={tau:.2f}`",
        f"- Zigzag+Binseg：`reversal={config.ZIGZAG_REVERSAL:.0%}`, "
        f"`min_ann_ret={config.ZIGZAG_MIN_ANNUALIZED_RETURN:.0%}`, "
        f"`min_duration={config.ZIGZAG_MIN_DURATION}`",
        f"- 震荡占比：**{ranging_pct:.2f}%**",
        f"- 回测区间：{config.BACKTEST_START} 至数据末尾，成本 {config.COST_BP}bp",
        "",
        "### 伪回归说明",
        "",
        "tau 是相对趋势强度分，非统计显著性水平；调参仅基于标签分布。",
        "",
        "## 2. 对照 A — 标签方法之争（同条件）",
        "",
        "固定：原 6 特征 + 研报策略规则 + 万得全A + 区间 + 成本。",
        "变量：仅标签方法（zigzag vs trend-scanning 二态）。",
        "",
        fair_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "详见 `table_fair_comparison.csv` / `fig_regime_overlay_zigzag.png` / `fig_regime_overlay_trendscan.png`",
        "",
        "## 3. 对照 B — 三态完整系统",
        "",
        "Trend-scanning 三态 + 方向特征 + 直接映射。",
        "",
        track_b_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. 模型（三态，测试集）",
        "",
        model_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. 完美标签天花板",
        "",
    ]

    if perfect_3:
        lines.append(
            f"- 三态直接映射：总收益 {_fmt_pct(perfect_3.total_return)}，"
            f"夏普 {_fmt_num(perfect_3.sharpe)}，回撤 {_fmt_pct(perfect_3.max_drawdown)}"
        )
        lines.append("- 自检图：`fig_perfect_label_position.png`")

    lines += [
        "",
        "## 6. 自检断言",
        "",
        f"**总体：{'全部通过 ✅' if all_passed else '存在失败项 ❌'}**",
        "",
        "\n".join(f"- {v}" for v in val_lines),
        "",
        "## 7. 产出文件",
        "",
        "- `table_fair_comparison.*` — 对照 A",
        "- `table_track_B.*` — 对照 B",
        "- `fig_perfect_label_position.png` — 完美标签仓位自检",
        "- `fig_regime_overlay_zigzag.png` / `fig_regime_overlay_trendscan.png`",
        "- `validation_report.md` — 断言明细",
    ]

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_all_figures(
    model_3state: dict,
    track_b: dict,
    close: pd.Series,
    output_dir: Path,
) -> None:
    for key in ["equal_weight", "logistic", "decision_tree"]:
        plot_confusion_matrix(model_3state[key], output_dir)
    plot_nav_curves(track_b["navs"], output_dir, "fig_nav_track_B.png")
