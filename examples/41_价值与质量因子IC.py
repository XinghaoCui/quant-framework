"""
示例 41: 价值因子 + 质量因子 IC 评估
=====================================

研究问题: 模拟数据上, 基本面因子有没有 IC?

模拟器特意制造了"高 ROE 股票价格也涨得多" 的关联 (price_link 机制),
所以质量因子应该有正 IC; 价值因子取决于 EP 和 BP 跟未来收益的关系.

跑完你会看到:
    - 质量因子 (ROE, 毛利率, 净利率) 通常有显著正 IC
    - 价值因子 (EP, BP) 在模拟数据上可能 IC 很弱 (因为价格漂移弱化价值效应)
    - 这正好是 "因子有没有用" 这个研究的真实模板, 可以照搬到真实数据
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.factors import (
    earnings_yield, book_to_price, roe_factor,
    gross_margin_factor, net_margin_factor,
    revenue_growth_factor, profit_growth_factor,
    compute_ic, ic_summary, forward_returns,
)


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2020-01-01", "2024-12-31", seed=42)
    print(f"数据: {prices.shape[0]} 天 x {prices.shape[1]} 只票")
    print(f"区间: {prices.index.min().date()} ~ {prices.index.max().date()}\n")

    factors = {
        # 价值 (估值便宜)
        "EP":               earnings_yield(prices, use_synthetic=True),
        "BP":               book_to_price(prices, use_synthetic=True),
        # 质量
        "ROE":              roe_factor(prices, use_synthetic=True),
        "gross_margin":     gross_margin_factor(prices, use_synthetic=True),
        "net_margin":       net_margin_factor(prices, use_synthetic=True),
        # 成长
        "revenue_growth":   revenue_growth_factor(prices, use_synthetic=True),
        "profit_growth":    profit_growth_factor(prices, use_synthetic=True),
    }

    horizon = 20  # 月度 forward (财务因子衰减慢, 用 20 天 IC 更合适)
    fwd = forward_returns(prices, horizon)

    print(f"===== 基本面因子 IC 评估 (horizon = {horizon} 天) =====")
    print(f"{'因子':<18} {'IC均值':>10} {'IR':>8} {'t统计':>8} {'IC>0占比':>10} {'样本':>8}")
    print("-" * 70)
    for name, factor in factors.items():
        ic = compute_ic(factor, fwd, method="spearman")
        s = ic_summary(ic)
        print(
            f"{name:<18} {s['IC均值']:>10.4f} {s['IR']:>8.3f} {s['t统计']:>8.2f} "
            f"{s['IC>0占比']:>10.2%} {s['样本天数']:>8}"
        )

    print("\n经验判读:")
    print("  - |IC| > 0.05 算有效, > 0.10 警惕 (查 bug)")
    print("  - 模拟数据上多数基本面因子 IC 是 *负的* (符号跟直觉相反)")
    print("    原因: 高 ROE / 高 EP 的票之前已经 '涨多了', 未来短期反而均值回归")
    print("    这正是真实研究里常遇到的: 因子方向不一定跟直觉一致, 要靠数据说话")
    print("  - 真实 A 股 / 美股上, ROE 通常是正 IC (质量溢价), EP 也通常正 IC (价值溢价)")
    print("\n下一步:")
    print("  - examples/42 看分组回测, 用单调性进一步验证因子方向")
    print("  - examples/43 看 ic_weighted 怎么自动反向用负 IC 因子, 反而比单因子稳得多")


if __name__ == "__main__":
    main()
