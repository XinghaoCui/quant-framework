"""
示例 0：用模拟数据跑通三个策略（无需联网）
===============================================

这个脚本生成几何布朗运动模拟价格，把双均线、动量、价值（反转）三个策略
全部跑一遍，输出指标对照表。

目的：
    在没网 / yfinance 限流 / 代理拦截国内数据源时，也能验证整个框架可用。
    模拟数据下没有真实 alpha，主要看 pipeline 是否跑通、指标计算是否对。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.strategies import SmaCrossStrategy, MomentumStrategy, ValueFactorStrategy
from src.utils import save_report
from src.engine.metrics import format_metrics


def main():
    # 生成 20 只票、7 年的模拟日线
    symbols = [f"STK{i:02d}" for i in range(20)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)
    print(f"模拟数据：{prices.shape[0]} 天 × {prices.shape[1]} 只票")
    print(f"价格区间：{prices.values.min():.2f} ~ {prices.values.max():.2f}")
    print()

    strategies = [
        SmaCrossStrategy(fast=20, slow=60, allow_short=False),
        MomentumStrategy(lookback_days=252, skip_days=21, top_pct=0.3, allow_short=True),
        ValueFactorStrategy(lookback_days=252, top_pct=0.3),
    ]

    bt = Backtester(commission=0.0003, slippage=0.0001)

    all_metrics = {}
    for strat in strategies:
        weights = strat.generate_weights(prices)
        result = bt.run(prices, weights)
        print(f"===== {strat.name} =====")
        print(format_metrics(result.metrics))
        print()
        save_report(result, name=f"00_synthetic_{strat.name}", output_dir="reports")
        all_metrics[strat.name] = result.metrics

    # 对照表
    print("\n策略对照（模拟数据，无真实 alpha）")
    comp = pd.DataFrame(all_metrics).T
    print(comp[["年化收益", "夏普比率", "最大回撤", "Calmar比率"]].to_string())


if __name__ == "__main__":
    main()
