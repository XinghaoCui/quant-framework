"""
示例 3：价值因子 - A 股大盘股
================================

简化版价值因子（用 1 年反转代理"便宜程度"）。
A 股数据来自 akshare。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_prices
from src.engine import Backtester
from src.strategies import ValueFactorStrategy
from src.utils import save_report
from src.engine.metrics import format_metrics


def main():
    # A 股几只大盘票（茅台、平安、五粮液、招商、宁德等）
    symbols = ["600519", "601318", "000858", "600036", "300750",
               "000333", "601166", "600276", "600887", "002594"]

    prices = load_prices(symbols, "2018-01-01", "2024-12-31", market="cn")
    print(f"加载完成：{prices.shape[0]} 个交易日，{prices.shape[1]} 只标的")

    strategy = ValueFactorStrategy(
        lookback_days=252,
        top_pct=0.3,
        rebalance_days=21,
    )
    weights = strategy.generate_weights(prices)

    # A 股佣金通常万 2.5 + 印花税千 1（单边），这里给一个保守估计
    bt = Backtester(commission=0.0008, slippage=0.0002)
    result = bt.run(prices, weights)

    print(f"\n策略：{strategy.name}")
    print(format_metrics(result.metrics))

    save_report(result, name="03_value_factor_cn", output_dir="reports")


if __name__ == "__main__":
    main()
