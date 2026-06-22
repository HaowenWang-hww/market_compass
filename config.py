"""Central configuration for Market Compass trend-scanning pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "万得全A数据.xlsx"
OUTPUT_DIR = ROOT / "outputs"

# ── Trend-scanning labeling ────────────────────────────────────────────────
MIN_H = 10
MAX_H = 66
TAU = 2.0
TAU_TARGET_RANGE = (0.25, 0.40)  # target share of label==0 (ranging)
USE_CHANGEPOINT = False
CHANGEPOINT_PENALTY = 3.0  # PELT penalty (BIC-like)

# ── Backtest / ML ──────────────────────────────────────────────────────────
BACKTEST_START = "2020-01-01"
TRAIN_RATIO = 0.7
SMOOTH_WINDOW = 20
RANDOM_STATE = 42
COST_BP = 2.0  # per-side transaction cost in basis points

# ── Label health-check ─────────────────────────────────────────────────────
SENSITIVITY_MIN_H = [8, 10, 12]
SENSITIVITY_MAX_H = [60, 66, 72]
SENSITIVITY_TAU = [1.5, 2.0, 2.5, 3.0]
FUTURE_HORIZONS = [5, 10, 20, 40]

# ── Zigzag + Binseg (fair comparison track A) ────────────────────────────
ZIGZAG_REVERSAL = 0.10
ZIGZAG_MIN_ANNUALIZED_RETURN = 0.20
ZIGZAG_MIN_DURATION = 63
ZIGZAG_BINSEG_SLOPE_RATIO = 0.5

# ── Plotting ─────────────────────────────────────────────────────────────
FIG_DPI = 150
LABEL_COLORS = {-1: "#2ca02c", 0: "#7f7f7f", 1: "#d62728"}  # green/down grey/up red
LABEL_NAMES = {-1: "下降", 0: "震荡", 1: "上升"}
