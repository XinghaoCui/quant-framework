"""
示例 22: IC 衰减曲线
====================

一个因子在 1 天后 IC 很高, 但 60 天后 IC 是不是还在?
IC 衰减曲线告诉你信号能维持多久, 决定:
    - 衰减快 → 高频换仓 → 交易成本高 → 实盘可能扛不住
    - 衰减慢 → 低频持有 → 适合大资金, 成本低

经验:
    动量因子 IC 衰减很慢, 持有几个月还有效
    反转因子 IC 衰减很快, 通常 1~2 周后就接近 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.factors import momentum, reversal
from src.factors.evaluation import ic_decay


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)

    horizons = [1, 3, 5, 10, 20, 40, 60]

    print("===== 动量因子 IC 衰减 =====")
    mom = momentum(prices, lookback=252, skip=21)
    print(ic_decay(mom, prices, horizons).round(4).to_string())

    print("\n===== 反转因子 IC 衰减 =====")
    rev = reversal(prices, window=21)
    print(ic_decay(rev, prices, horizons).round(4).to_string())

    print("\n经验: 真实股票池上, 动量 IC 通常能维持到 horizon=60+,")
    print("      反转 IC 通常在 horizon=10~20 就接近 0.")


if __name__ == "__main__":
    main()
