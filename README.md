# Market Compass — Trend-Scanning 万得全A择时

用 **Trend-Scanning**（López de Prado）对万得全A生成三态方向标签 `{-1, 0, +1}`，替代研报中的 zigzag+Binseg 标注，并复刻研报测试流程（特征 → 三模型 → 策略回测）。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

将 `万得全A数据.xlsx` 放在项目根目录（前 6 行为元信息，第 7 行起为 日期/收盘价/成交额）。

> **测试用**：若无真实数据，可运行 `python scripts/bootstrap_data.py` 生成合成数据验证流程（结果不可与研报对照）。

## 项目结构

| 模块 | 功能 |
|------|------|
| `config.py` | 集中参数（min_h, max_h, tau, 回测区间等） |
| `data.py` | 数据加载 |
| `labeling.py` | Trend-scanning 三态标注 + 可选 changepoint 变体 |
| `health_check.py` | 标签体检（稳定性/区分度/天花板） |
| `features.py` | 6 个因果价量特征 |
| `models.py` | 等权 / 逻辑回归 / 决策树 + purge/embargo |
| `strategy.py` | 研报策略复刻 + 三态直接映射改进版 |
| `report.py` | 图表、表格、markdown 报告 |
| `main.py` | 一键运行 |

## 核心参数（`config.py`）

- `MIN_H=10, MAX_H=66`：扫描窗口（约两周到一季度），`MAX_H` 同时是 embargo 长度
- `TAU`：趋势强度阈值，自动调整使震荡占比落在 25–40%
- `BACKTEST_START='2020-01-01'`：回测起点
- `COST_BP=2`：双边交易成本（基点）

## 伪回归 Caveat

价格序列近似单位根，对 `log(close)` 做 `logP ~ a + b·时间` 的 OLS 回归时，斜率 t 值会被**夸大**（伪回归/spurious regression）。因此：

- `tau` **不是**真正的统计显著性水平（如 5%）
- t 值仅作为**相对趋势强度分**使用
- 调 `tau` 和 `span` 仅依据标签分布（震荡占比），**绝不**用下游策略收益调参

## 评估准则

**以策略夏普比率为最终判据，而非预测准确率。**

参考研报：等权模型准确率最高但策略夏普仅 0.34；决策树策略夏普 1.12 甚至超过带未来信息的完美标签策略（夏普 0.99）。

## 产出

运行后所有结果写入 `outputs/`：

- `labeled_data.csv` — 标注数据
- `fig_regime_overlay.png` — 全历史三态上色图
- `report.md` — 完整研究报告
- 混淆矩阵、净值曲线、策略指标表等

## 可选：Changepoint 变体

在 `config.py` 设置 `USE_CHANGEPOINT=True` 启用 `ruptures` PELT 分段标注（对照实验）。
