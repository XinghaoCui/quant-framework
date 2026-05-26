"""
示例 30: 组合优化方法对比 (小白入门)
====================================

研究问题: 同样选 6 只股票, 不同的"分仓位"方法谁更好?

对比 4 种方法 (都不需要预测未来收益):
    1. 等权 (equal)              每只 1/6
    2. 反波动率 (inverse_vol)     波动小的多分
    3. 风险平价 (risk_parity)     每只对总风险贡献相同
    4. 最小方差 (min_variance)    凸优化, 让组合波动最小

判读:
    - 看 "组合波动率" (越小越稳)
    - 看 "夏普" (单位风险收益)
    - 看 "权重集中度" (是不是只压一只票)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data import generate_synthetic_prices
from src.portfolio import (
    equal_weights,
    inverse_volatility_weights,
    risk_parity_weights,
    min_variance_weights,
)


def main():
    # 造 6 只票 ~ 3 年模拟数据, 选 6 只是为了让权重表更清晰
    symbols = [f"STK{i:02d}" for i in range(6)]
    prices = generate_synthetic_prices(symbols, "2022-01-01", "2024-12-31", seed=42)
    returns = prices.pct_change().dropna()

    print(f"用 {len(returns)} 天的历史日收益估协方差")
    print(f"6 只票的日波动率: {returns.std().round(4).to_dict()}\n")

    # 跑 4 种优化
    results = {
        "equal":         equal_weights(returns),
        "inverse_vol":   inverse_volatility_weights(returns),
        "risk_parity":   risk_parity_weights(returns),
        "min_variance":  min_variance_weights(returns),
    }

    # 整成大表
    weight_table = pd.DataFrame({m: r.weights for m, r in results.items()}).round(4)
    print("===== 各方法权重对比 =====")
    print(weight_table.to_string())
    print()

    # 摘要
    summary_rows = {}
    for name, res in results.items():
        # 把日波动率年化到 % 形式方便看
        port_vol_annual = res.expected_vol * np.sqrt(252)
        max_w = float(res.weights.max())
        herfindahl = float((res.weights ** 2).sum())  # 越大越集中, 1 = 全押一只, 1/N = 完全均匀
        summary_rows[name] = {
            "组合年化波动": port_vol_annual,
            "最大单票权重": max_w,
            "集中度 (HHI)":  herfindahl,
        }
    summary = pd.DataFrame(summary_rows).T.round(4)
    print("===== 各方法风险摘要 =====")
    print(summary.to_string())

    print("\n小白怎么读这张表:")
    print("  - '组合年化波动' 越小越稳 (但牺牲了灵活)")
    print("  - 'HHI' 是集中度, 等权 = 1/N = 0.167; min_var 经常会集中到几只波动小的票")
    print("  - 没有'谁绝对最好', 看你想要稳还是想要灵活")
    print("\n注意: 这是'静态'优化 (一次性算所有权重), 实际策略里要滚动算,")
    print("     见 examples/32_组合优化策略接入回测.py")


if __name__ == "__main__":
    main()
