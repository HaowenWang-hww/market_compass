"""Market Compass — Trend-Scanning pipeline entry point."""

from __future__ import annotations

import time

import pandas as pd

import config
from data import load_data
from features import build_features, FEATURE_COLS
from features_directional import FEATURE_COLS_3STATE, build_features_3state
from health_check import run_health_checks, label_distribution, regime_overlay_plot
from labeling import build_labeled_dataframe
from models import (
    temporal_split_with_purge,
    train_and_predict_binary,
    train_and_predict_3state,
    predict_full_series_binary,
    predict_full_series_3state,
)
from report import (
    generate_report,
    generate_all_figures,
    save_fair_comparison,
    save_track_b,
    save_model_tables,
)
from strategy import run_track_a, run_track_b
from validate import run_all_validations
from zigzag_labeling import build_zigzag_dataframe


def main() -> None:
    t0 = time.time()
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Market Compass — Revised Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/9] Loading data...")
    df = load_data()
    print(f"  Rows: {len(df)}, range: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # ── 2. Labeling (trend-scan + zigzag) ─────────────────────────
    print("\n[2/9] Labeling...")
    close = df.set_index("date")["close"]
    labeled_ts, tau = build_labeled_dataframe(df)
    labeled_zz = build_zigzag_dataframe(df)
    print(f"  Trend-scan tau={tau:.2f}, rows={len(labeled_ts)}")
    print(f"  Zigzag binary rows={len(labeled_zz)}")

    dist = label_distribution(labeled_ts)
    for _, row in dist.iterrows():
        print(f"  {row['name']}: {row['share']*100:.1f}%")

    labeled_ts.to_csv(output_dir / "labeled_data_trendscan.csv", index=False, float_format="%.6f")
    labeled_zz.to_csv(output_dir / "labeled_data_zigzag.csv", index=False, float_format="%.6f")

    regime_overlay_plot(
        labeled_ts, output_dir, label_col="label",
        filename="fig_regime_overlay_trendscan.png",
        title="万得全A — Trend-Scanning 三态标注",
    )
    regime_overlay_plot(
        labeled_zz, output_dir, label_col="label_binary",
        filename="fig_regime_overlay_zigzag.png",
        title="万得全A — Zigzag+Binseg 二态标注",
    )
    regime_overlay_plot(
        labeled_ts, output_dir, label_col="label",
        filename="fig_regime_overlay.png",
        title="万得全A — Trend-Scanning 三态标注",
    )

    # ── 3. Health checks ──────────────────────────────────────────
    print("\n[3/9] Label health checks...")
    health = run_health_checks(labeled_ts, close, (config.MIN_H, config.MAX_H, tau), output_dir)

    # ── 4. Features ───────────────────────────────────────────────
    print("\n[4/9] Feature engineering...")
    feat_binary = build_features(df)
    feat_3state = build_features_3state(df)
    print(f"  Binary features: {FEATURE_COLS}")
    print(f"  3-state features: +{len(FEATURE_COLS_3STATE) - len(FEATURE_COLS)} directional")

    merged_ts_bin = labeled_ts.merge(feat_binary, on=["date", "close", "amt"], how="inner")
    merged_zz_bin = labeled_zz.merge(feat_binary, on=["date", "close", "amt"], how="inner")
    merged_ts_3 = labeled_ts.merge(feat_3state, on=["date", "close", "amt"], how="inner")

    # ── 5. Track A models (binary, fair comparison) ───────────────
    print("\n[5/9] Track A — binary models (zigzag vs trendscan)...")
    split_zz = temporal_split_with_purge(merged_zz_bin, FEATURE_COLS, label_col="label_binary")
    split_ts_bin = temporal_split_with_purge(merged_ts_bin, FEATURE_COLS, label_col="label_binary")
    print(f"  Zigzag train={len(split_zz.train)}, test={len(split_zz.test)}")
    print(f"  Trendscan train={len(split_ts_bin.train)}, test={len(split_ts_bin.test)}")

    model_zz = train_and_predict_binary(split_zz)
    model_ts_bin = train_and_predict_binary(split_ts_bin)
    pred_zz = predict_full_series_binary(merged_zz_bin, model_zz)
    pred_ts_bin = predict_full_series_binary(merged_ts_bin, model_ts_bin)

    # ── 6. Track B models (3-state + directional) ─────────────────
    print("\n[6/9] Track B — 3-state models with directional features...")
    split_3 = temporal_split_with_purge(merged_ts_3, FEATURE_COLS_3STATE, label_col="label")
    model_3 = train_and_predict_3state(split_3, FEATURE_COLS_3STATE)
    for key in ["equal_weight", "logistic", "decision_tree"]:
        r = model_3[key]
        rd = r.recall_per_class.get(-1, 0)
        print(f"  {r.name}: acc={r.accuracy:.4f}, recall_下降={rd:.4f}")
    pred_3 = predict_full_series_3state(merged_ts_3, model_3, FEATURE_COLS_3STATE)

    # ── 7. Strategy backtests ─────────────────────────────────────
    print("\n[7/9] Strategy backtesting...")
    track_a = run_track_a(labeled_zz, labeled_ts, pred_zz, pred_ts_bin)
    track_b = run_track_b(pred_3, labeled_ts)

    zz_perfect = track_a["metrics"]["zigzag"].get("完美标签")
    ts_perfect = track_a["metrics"]["trendscan"].get("完美标签")
    if zz_perfect and ts_perfect:
        print(f"  Track A 完美标签夏普: zigzag={zz_perfect.sharpe:.2f}, trendscan={ts_perfect.sharpe:.2f}")

    perfect_3 = track_b["metrics"].get("完美标签(三态直接映射)")
    bench = track_b["metrics"].get("基准(0.5买入持有)")
    if perfect_3 and bench:
        print(
            f"  Track B 完美三态: sharpe={perfect_3.sharpe:.2f}, "
            f"mdd={perfect_3.max_drawdown*100:.2f}% | 基准 sharpe={bench.sharpe:.2f}"
        )

    # ── 8. Reports & figures ──────────────────────────────────────
    print("\n[8/9] Generating reports & figures...")
    fair_df = save_fair_comparison(track_a, output_dir)
    track_b_df = save_track_b(track_b, output_dir)
    save_model_tables(model_ts_bin, model_3, output_dir)
    close_bt = merged_ts_3[merged_ts_3["date"] >= config.BACKTEST_START].set_index("date")["close"]
    generate_all_figures(model_3, track_b, close_bt, output_dir)

    # ── 9. Validation ─────────────────────────────────────────────
    print("\n[9/9] Running self-check validations...")
    validations = run_all_validations(
        track_b_metrics=track_b["metrics"],
        model_results_3state=model_3,
        split_3state=split_3,
        zigzag_labeled=labeled_zz,
        trendscan_labeled=labeled_ts,
        df=df,
        labeled_df=labeled_ts,
        output_dir=output_dir,
    )

    generate_report(
        tau=tau,
        label_dist=dist,
        health=health,
        fair_df=fair_df,
        track_b_df=track_b_df,
        model_3state=model_3,
        validations=validations,
        output_dir=output_dir,
    )

    elapsed = time.time() - t0
    passed = sum(1 for v in validations if v.passed)
    print(f"\nDone in {elapsed:.1f}s. Validations: {passed}/{len(validations)} passed.")
    print(f"Outputs: {output_dir}")
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
