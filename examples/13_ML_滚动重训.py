"""
示例 13：滚动重训（Walk-Forward）对比
========================================

对比三种训练方式，理解为什么实盘要"滚动重训"：
    1. 训练一次       —— 用前半段训练，后半段全用这个老模型
    2. expanding 扩展窗口 —— 每半年用"全部历史"重训
    3. rolling 滚动窗口   —— 每半年用"最近2年"重训（会遗忘旧数据）

观察：
    - walk-forward 的样本外区间更长（从第3年就开始预测）
    - 重训日志展示了透明的"训练→预测→再训练"滚动过程

跑：python examples/13_ML_滚动重训.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.ml.ml_strategy import MLStrategy
from src.ml import make_dataset, information_coefficient
from src.engine.metrics import format_metrics


def run(prices, label, **kwargs):
    """跑一个 MLStrategy 配置，返回 (策略, 回测结果, IC)。"""
    strat = MLStrategy(
        model_name="ridge", task="regression",
        horizon=5, top_pct=0.3, rebalance_days=5, **kwargs
    )
    weights = strat.generate_weights(prices)
    bt = Backtester(commission=0.0003, slippage=0.0001)
    result = bt.run(prices, weights)

    # 用预测和真实未来收益算 IC
    _, y = make_dataset(prices, horizon=5, kind="return")
    ic = information_coefficient(strat.predictions_, y)

    print(f"\n{'='*55}\n【{label}】")
    print(f"  预测样本数: {len(strat.predictions_)}  "
          f"预测起止: {strat.predictions_.index.get_level_values('date').min().date()} "
          f"~ {strat.predictions_.index.get_level_values('date').max().date()}")
    print(f"  IC均值: {ic['IC均值']:.4f}  ICIR: {ic['ICIR']:.3f}")
    print(f"  年化收益: {result.metrics['年化收益']:.2%}  "
          f"夏普: {result.metrics['夏普比率']:.3f}  "
          f"最大回撤: {result.metrics['最大回撤']:.2%}")
    return strat, result, ic


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2015-01-01", "2024-12-31", seed=20)
    print(f"价格数据：{prices.shape[0]} 天 × {prices.shape[1]} 只票")

    # 1. 训练一次
    run(prices, "模式1：训练一次（train_ratio=0.5）", retrain_freq=None, train_ratio=0.5)

    # 2. expanding 扩展窗口，每半年重训
    s2, _, _ = run(
        prices, "模式2：滚动重训 expanding（每126天，初始2年）",
        retrain_freq=126, window="expanding", initial_train_days=504,
    )

    # 3. rolling 滚动窗口，每半年重训，只用最近2年
    run(
        prices, "模式3：滚动重训 rolling（每126天，窗口2年）",
        retrain_freq=126, window="rolling",
        initial_train_days=504, rolling_window_days=504,
    )

    # 展示 expanding 模式的重训日志（透明的滚动过程）
    print(f"\n{'='*55}\n【expanding 重训日志】（训练窗口随时间扩大）")
    print(s2.walk_log_.to_string(index=False))

    print("\n提示：模拟数据规律恒定，三种模式差异不大。")
    print("真实市场非平稳，rolling 能更快适应régime切换，expanding 数据更充分。")


if __name__ == "__main__":
    main()
