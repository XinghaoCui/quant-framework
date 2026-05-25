"""
示例 1：双均线策略 - 美股科技股
================================

本示例展示完整的回测流程：
    1. 加载数据 → 2. 实例化策略 → 3. 生成权重 → 4. 跑回测 → 5. 出报告

可以直接 python examples/01_双均线_美股.py 运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根加入 sys.path，方便相对导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_prices
from src.engine import Backtester
from src.strategies import SmaCrossStrategy
from src.utils import save_report
from src.engine.metrics import format_metrics


def main():
    # ===== 1. 数据 =====
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    prices = load_prices(symbols, "2018-01-01", "2024-12-31", market="us")
    print(f"加载完成：{prices.shape[0]} 个交易日，{prices.shape[1]} 只标的")
    print(prices.tail(3))

    # ===== 2. 策略 =====
    strategy = SmaCrossStrategy(fast=20, slow=60, allow_short=False)
    weights = strategy.generate_weights(prices)

    # ===== 3. 回测 =====
    bt = Backtester(commission=0.0003, slippage=0.0001)
    result = bt.run(prices, weights)

    # ===== 4. 看结果 =====
    print("\n" + "=" * 50)
    print(f"策略：{strategy.name}（fast={strategy.fast}, slow={strategy.slow}）")
    print(format_metrics(result.metrics))

    # ===== 5. 报告 =====
    save_report(result, name="01_sma_cross_us", output_dir="reports")


if __name__ == "__main__":
    main()
