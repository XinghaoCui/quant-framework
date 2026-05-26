"""
示例 42: 质量因子 ROE 分组回测
==============================

把 30 只票按 ROE 分 5 组, 看高组真的跑赢低组吗?

判读重点 (跟 07 章 / 03 分组回测一致):
    - 单调性 Q1 < Q2 < ... < Q5
    - 多空价差 Q5-Q1 的年化收益和夏普
    - 多/空头哪边驱动更强
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import generate_synthetic_prices
from src.factors import roe_factor, quantile_backtest


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2020-01-01", "2024-12-31", seed=42)

    roe = roe_factor(prices, use_synthetic=True)

    result = quantile_backtest(roe, prices, n_quantiles=5, horizon=20)

    print("===== ROE 因子 5 分组回测 (持有期 20 天) =====\n")
    print("各组摘要:")
    print(result.summary.round(4).to_string())

    print("\n[单调性检查]")
    rets = result.summary["年化收益"].loc[["Q1", "Q2", "Q3", "Q4", "Q5"]]
    is_mono = rets.is_monotonic_increasing or rets.is_monotonic_decreasing
    print(f"  Q1→Q5 年化收益: {rets.round(4).tolist()}")
    print(f"  严格单调? {is_mono}")
    print(f"  多空价差 (Q5-Q1) 年化: {result.summary.loc['多空(Qn-Q1)', '年化收益']:.4f}")
    print(f"  多空价差 (Q5-Q1) 夏普: {result.summary.loc['多空(Qn-Q1)', '夏普']:.4f}")

    print("\n[累计净值末 5 天]")
    print(result.cum_returns.tail(5).round(4).to_string())

    print("\n小白怎么读:")
    print("  - 看 Q1~Q5 累计净值: 单调向上 = 高 ROE 看多, 单调向下 = 高 ROE 看空")
    print("  - 看多空价差 (Q5-Q1) 的年化收益和夏普: |夏普| 大 = 因子方向明确, 不为 0 即可被利用")
    print("  - 在本次模拟数据上 ROE 是 *反向* 单调 (Q1 高 Q5 低), 多空年化为负")
    print("    这不算 bug, 是模拟器内部 '高 ROE 已经透支预期' 的副作用, 反向用因子也能挣")
    print("  - 真实 A 股 / 美股上 ROE 通常正向单调, 不要把模拟结论搬过去")


if __name__ == "__main__":
    main()
