"""Label quality checks before model training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import config
from labeling import _apply_tau, _compute_raw_t_values, tune_tau
from strategy import direct_mapping_strategy, buy_hold_benchmark, _compute_metrics, replicate_strategy

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def stability_check(
    close: pd.Series,
    baseline_params: tuple[int, int, float],
    output_dir: Path,
) -> pd.DataFrame:
    """Perturb (min_h, max_h, tau) and compute Cohen's kappa vs baseline."""
    min_h, max_h, tau = baseline_params
    base_raw = _compute_raw_t_values(close, min_h, max_h)
    base = _apply_tau(base_raw, tau)
    base_labels = base["label"]

    rows = []
    # Tau sensitivity (fast: reuse raw t-values)
    for t in config.SENSITIVITY_TAU:
        alt = _apply_tau(base_raw, t)
        common = base_labels.index.intersection(alt.index)
        if len(common) < 100:
            continue
        kappa = cohen_kappa_score(
            base_labels.loc[common].values,
            alt.loc[common, "label"].values,
            labels=[-1, 0, 1],
        )
        disagree = (base_labels.loc[common] != alt.loc[common, "label"]).mean()
        rows.append({
            "min_h": min_h, "max_h": max_h, "tau": t,
            "cohens_kappa": kappa, "disagree_rate": disagree,
            "is_baseline": t == tau,
        })

    # Span sensitivity (recompute raw for different min_h / max_h)
    for mh in config.SENSITIVITY_MIN_H:
        for xh in config.SENSITIVITY_MAX_H:
            if mh == min_h and xh == max_h:
                continue
            alt_raw = _compute_raw_t_values(close, mh, xh)
            alt_tau = tune_tau(close, mh, xh, raw=alt_raw)
            alt = _apply_tau(alt_raw, alt_tau)
            common = base_labels.index.intersection(alt.index)
            if len(common) < 100:
                continue
            kappa = cohen_kappa_score(
                base_labels.loc[common].values,
                alt.loc[common, "label"].values,
                labels=[-1, 0, 1],
            )
            disagree = (base_labels.loc[common] != alt.loc[common, "label"]).mean()
            rows.append({
                "min_h": mh, "max_h": xh, "tau": alt_tau,
                "cohens_kappa": kappa, "disagree_rate": disagree,
                "is_baseline": False,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "health_stability.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(index="tau", columns="min_h", values="cohens_kappa", aggfunc="mean")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{t:.1f}" for t in pivot.index])
    ax.set_xlabel("min_h")
    ax.set_ylabel("tau")
    ax.set_title("标签稳定性：Cohen's κ 敏感性")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_stability_kappa.png", dpi=config.FIG_DPI)
    plt.close(fig)
    return df


def regime_overlay_plot(
    df: pd.DataFrame,
    output_dir: Path,
    label_col: str = "label",
    filename: str = "fig_regime_overlay.png",
    title: str = "万得全A 价格与三态标注",
) -> None:
    """Full-history price + three-state coloring."""
    work = df.dropna(subset=[label_col]).copy()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(work["date"], work["close"], color="black", linewidth=0.8, alpha=0.7)

    if label_col == "label":
        for lbl in [-1, 0, 1]:
            mask = work[label_col] == lbl
            ax.scatter(
                work.loc[mask, "date"],
                work.loc[mask, "close"],
                c=config.LABEL_COLORS[lbl],
                s=3,
                alpha=0.6,
                label=config.LABEL_NAMES[lbl],
            )
    else:
        for lbl, color, name in [(1, "#d62728", "趋势"), (0, "#7f7f7f", "震荡")]:
            mask = work[label_col] == lbl
            ax.scatter(
                work.loc[mask, "date"],
                work.loc[mask, "close"],
                c=color,
                s=3,
                alpha=0.6,
                label=name,
            )

    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=config.FIG_DPI)
    plt.close(fig)


def discrimination_check(
    df: pd.DataFrame,
    output_dir: Path,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Future N-day returns and volatility by label group."""
    horizons = horizons or config.FUTURE_HORIZONS
    work = df.dropna(subset=["label", "close"]).copy()
    work = work.set_index("date")

    rows = []
    for h in horizons:
        fwd_ret = work["close"].pct_change(h).shift(-h)
        fwd_vol = work["close"].pct_change().rolling(h).std().shift(-h) * np.sqrt(252)
        for lbl in [-1, 0, 1]:
            mask = work["label"] == lbl
            rets = fwd_ret[mask].dropna()
            vols = fwd_vol[mask].dropna()
            rows.append({
                "horizon": h,
                "label": lbl,
                "label_name": config.LABEL_NAMES[lbl],
                "mean_return": rets.mean(),
                "median_return": rets.median(),
                "mean_vol": vols.mean(),
                "n": len(rets),
            })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "health_discrimination.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        sub = result[result["horizon"] == h]
        data = []
        labels_plot = []
        for lbl in [-1, 0, 1]:
            mask = work["label"] == lbl
            fwd = work["close"].pct_change(h).shift(-h)[mask].dropna()
            data.append(fwd.values)
            labels_plot.append(config.LABEL_NAMES[lbl])
        ax.boxplot(data, labels=labels_plot)
        ax.set_title(f"未来{h}日收益")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    fig.suptitle("三态标签区分度：未来收益分布")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_discrimination_boxplot.png", dpi=config.FIG_DPI)
    plt.close(fig)
    return result


def economic_ceiling(
    df: pd.DataFrame,
    output_dir: Path,
    backtest_start: str = config.BACKTEST_START,
) -> dict:
    """Perfect foresight strategy with transaction costs."""
    bt = df[df["date"] >= pd.Timestamp(backtest_start)].copy()
    bench_nav, _, _ = buy_hold_benchmark(bt)

    results = {}
    for name, col, mode in [
        ("完美三态直接映射", "label_perfect", True),
        ("完美二态研报策略", "label_binary", False),
    ]:
        if mode:
            nav, _, _ = direct_mapping_strategy(bt, col, name)
        else:
            nav, _, _ = replicate_strategy(bt, col, name)
        m = _compute_metrics(nav, name, bench_nav)
        results[name] = m

    rows = [{
        "strategy": k,
        "total_return": v.total_return,
        "sharpe": v.sharpe,
        "max_drawdown": v.max_drawdown,
    } for k, v in results.items()]
    pd.DataFrame(rows).to_csv(output_dir / "health_ceiling.csv", index=False, float_format="%.4f")
    return results


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of label shares."""
    counts = df["label"].value_counts().sort_index()
    total = len(df.dropna(subset=["label"]))
    rows = []
    for lbl in [-1, 0, 1]:
        n = counts.get(lbl, 0)
        rows.append({
            "label": lbl,
            "name": config.LABEL_NAMES[lbl],
            "count": n,
            "share": n / total if total else 0,
        })
    return pd.DataFrame(rows)


def run_health_checks(
    df: pd.DataFrame,
    close: pd.Series,
    params: tuple[int, int, float],
    output_dir: Path,
) -> dict:
    """Run all four health-check items."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stability = stability_check(close, params, output_dir)
    regime_overlay_plot(df, output_dir)
    discrimination = discrimination_check(df, output_dir)
    ceiling = economic_ceiling(df, output_dir)
    distribution = label_distribution(df)
    distribution.to_csv(output_dir / "health_distribution.csv", index=False, float_format="%.4f")

    return {
        "stability": stability,
        "discrimination": discrimination,
        "ceiling": ceiling,
        "distribution": distribution,
    }
