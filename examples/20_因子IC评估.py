"""
示例 20: 单因子 IC 评估
=======================

研究流程第一步: 拿到任何新因子, 先看它的 IC 怎么样.
不要急着上策略 —— 先确认 "因子排序能不能预测收益排序".

本例对三个经典因子做 IC 评估:
    动量 / 反转 / 低波 (波动率取负)

判读:
    |IC 均值| > 0.02 勉强可用
    |IC 均值| > 0.05 不错
    IR > 0.5  IC 稳定
    t > 2     5% 显著
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.factors import momentum, reversal, volatility
from src.factors.evaluation import compute_ic, ic_summary, forward_returns


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)
    print(f"模拟数据: {prices.shape[0]} 天 x {prices.shape[1]} 只票\n")

    factors = {
        "momentum_252_21": momentum(prices, lookback=252, skip=21),
        "reversal_21":     reversal(prices, window=21),
        "lowvol_60":      -volatility(prices, window=60),  # 低波好, 取负号
    }

    horizon = 5
    fwd = forward_returns(prices, horizon)

    print(f"{'因子':<20} {'IC均值':>8} {'IR':>8} {'t统计':>8} {'IC>0占比':>10} {'样本天数':>8}")
    print("-" * 70)
    for name, factor in factors.items():
        ic = compute_ic(factor, fwd, method="spearman")
        s = ic_summary(ic)
        print(
            f"{name:<20} {s['IC均值']:>8.4f} {s['IR']:>8.3f} "
            f"{s['t统计']:>8.2f} {s['IC>0占比']:>10.2%} {s['样本天数']:>8}"
        )

    print("\n注: 模拟数据是几何布朗运动 + 随机扰动, 真实 alpha 很弱,")
    print("    所以 IC 通常贴近 0. 真实股票池上动量 IC 经验值 0.03~0.05.")


if __name__ == "__main__":
    main()
