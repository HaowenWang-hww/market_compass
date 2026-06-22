"""Bootstrap synthetic 万得全A xlsx for pipeline testing (replace with real data)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "万得全A数据.xlsx"


def generate_synthetic() -> pd.DataFrame:
    """~4965 business days, 2005-12-30 to 2026-06-12, regime-switching random walk."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2005-12-30", "2026-06-12")
    n = len(dates)

    log_price = np.zeros(n)
    log_amt = np.log(50.0)
    regime = 0
    regime_len = 0

    for i in range(1, n):
        if regime_len <= 0:
            regime = rng.choice([-1, 0, 1], p=[0.3, 0.3, 0.4])
            regime_len = int(rng.integers(40, 120))
        drift = { -1: -0.0003, 0: 0.0, 1: 0.0004 }[regime]
        log_price[i] = log_price[i - 1] + drift + rng.normal(0, 0.012)
        log_amt += rng.normal(0, 0.05) + 0.1 * abs(drift) * 100
        regime_len -= 1

    close = np.exp(log_price) * 1000
    amt = np.exp(log_amt)
    return pd.DataFrame({"date": dates, "close": close, "amt": amt})


def save_with_metadata(df: pd.DataFrame, path: Path) -> None:
    meta = pd.DataFrame([
        ["万得全A指数", "", ""],
        ["数据来源: 合成测试数据", "", ""],
        ["", "", ""],
        ["日期", "收盘价", "成交额(亿元)"],
        ["", "", ""],
        ["", "", ""],
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        meta.to_excel(writer, index=False, header=False, startrow=0)
        df.to_excel(writer, index=False, header=False, startrow=6)


if __name__ == "__main__":
    df = generate_synthetic()
    save_with_metadata(df, OUT)
    print(f"Saved {len(df)} rows to {OUT}")
