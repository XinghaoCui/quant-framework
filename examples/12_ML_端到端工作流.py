"""
示例 12：端到端 ML 量化工作流
==================================

把 ML 预测接入回测引擎，得到真正的策略净值曲线和绩效指标。
这是"从数据到可交易策略"的完整闭环：

    价格 → 特征工程 → 训练模型 → 预测 → 排序选股 → 回测 → 指标/报告

对比：ML 策略 vs 简单基准（等权买入持有）。

跑：python examples/12_ML_端到端工作流.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.ml.ml_strategy import MLStrategy
from src.utils import save_report
from src.engine.metrics import format_metrics, compute_metrics


def main():
    # 1. 数据
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2015-01-01", "2024-12-31", seed=12)
    print(f"价格数据：{prices.shape[0]} 天 × {prices.shape[1]} 只票\n")

    # 2. ML 策略（用梯度提升树预测 5 日收益，做多预测最高 30%）
    strategy = MLStrategy(
        model_name="gbdt",
        task="regression",
        horizon=5,
        top_pct=0.3,
        allow_short=False,
        rebalance_days=5,
        train_ratio=0.5,   # 前 50% 训练，后 50% 回测
    )

    # 3. 生成权重（内部完成训练+预测）
    print("训练模型 + 生成交易信号 ...")
    weights = strategy.generate_weights(prices)

    # 4. 基准：等权买入持有
    benchmark = prices.pct_change().mean(axis=1)

    # 5. 回测
    bt = Backtester(commission=0.0003, slippage=0.0001)
    result = bt.run(prices, weights, benchmark=benchmark)

    # 6. 结果
    print("\n===== ML 策略回测（仅测试期有持仓）=====")
    print(format_metrics(result.metrics))

    print("\n===== 基准（等权买入持有，全期）=====")
    print(format_metrics(compute_metrics(benchmark)))

    # 7. 特征重要性
    print("\n===== 模型特征重要性 =====")
    print(strategy.model.feature_importance().to_string())

    save_report(result, name="12_ml_gbdt", output_dir="reports")


if __name__ == "__main__":
    main()
