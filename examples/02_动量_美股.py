"""
示例 2：横截面动量 - 美股大盘股
================================

复现经典论文的横截面动量策略：
    买入过去 12 个月（跳过最近 1 个月）涨幅前 30% 的股票，
    卖空跌幅最深的 30%，每月调仓。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_prices
from src.engine import Backtester
from src.strategies import MomentumStrategy
from src.utils import save_report
from src.engine.metrics import format_metrics


def main():
    # 选一个相对宽的样本池（道指成分股的一部分）
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "BAC", "WFC", "GS",
        "XOM", "CVX", "COP",
        "JNJ", "PFE", "MRK",
        "WMT", "HD", "COST", "MCD",
    ]
    prices = load_prices(symbols, "2015-01-01", "2024-12-31", market="us")
    print(f"加载完成：{prices.shape[0]} 个交易日，{prices.shape[1]} 只标的")

    strategy = MomentumStrategy(
        lookback_days=252,
        skip_days=21,
        top_pct=0.3,
        bottom_pct=0.3,
        rebalance_days=21,
        allow_short=True,
    )
    weights = strategy.generate_weights(prices)

    bt = Backtester(commission=0.0003, slippage=0.0001)
    result = bt.run(prices, weights)

    print(f"\n策略：{strategy.name}")
    print(format_metrics(result.metrics))

    save_report(result, name="02_momentum_us", output_dir="reports")


if __name__ == "__main__":
    main()
