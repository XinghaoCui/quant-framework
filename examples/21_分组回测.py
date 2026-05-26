"""
示例 21: 因子分组回测
=====================

IC 只能告诉你"线性关系强不强", 不能告诉你"实际选股能不能赚".
分组回测把股票按因子值分成 N 组, 看高组真的跑赢低组吗.

要看的两件事:
    1. 单调性: Q5 > Q4 > Q3 > Q2 > Q1 吗? 越单调越好
    2. 多空价差: Q5 - Q1 年化收益和夏普, 这是"纯因子 alpha"

模拟数据没有真实 alpha, 主要看流程跑通; 真实数据上单调性更明显.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.factors import reversal
from src.factors.evaluation import quantile_backtest


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)

    factor = reversal(prices, window=21)

    result = quantile_backtest(factor, prices, n_quantiles=5, horizon=5)

    print("===== 反转因子 (window=21) 分组回测 =====")
    print(f"持有期: 5 天   分组数: 5\n")
    print("各组摘要:")
    print(result.summary.round(4).to_string())

    print("\n[单调性检查]")
    rets = result.summary["年化收益"][["Q1", "Q2", "Q3", "Q4", "Q5"]]
    monotonic = rets.is_monotonic_increasing or rets.is_monotonic_decreasing
    print(f"  Q1→Q5 年化收益: {rets.tolist()}")
    print(f"  严格单调? {monotonic}")
    print(f"  多空价差 (Q5-Q1) 年化: {result.summary.loc['多空(Qn-Q1)', '年化收益']:.4f}")

    print("\n[累计净值 (最后 5 天)]")
    print(result.cum_returns.tail(5).round(4).to_string())


if __name__ == "__main__":
    main()
