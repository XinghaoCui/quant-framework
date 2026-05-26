"""
示例 23: 多因子合成策略
========================

把动量 + 反转 + 低波三个相关性低的因子合成一个综合分,
然后选 top 30% 等权持有, 月度调仓.

对比:
    1. 单因子: 三个因子各自跑一个策略
    2. equal 等权合成
    3. ic_weighted 滚动 IC 加权合成
    4. 基准: 等权买入持有 (永远是 sanity check)

模拟数据上四种策略差距不会很大 (真实 alpha 弱), 真实数据上
多因子合成通常比单因子稳一截, ic_weighted 不一定打得过 equal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.engine.metrics import format_metrics
from src.factors import momentum, reversal, volatility
from src.strategies import MultiFactorStrategy


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.DataFrame:
    """每天等权持有所有票, 作为最朴素的基准."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)
    print(f"数据: {prices.shape[0]} 天 x {prices.shape[1]} 只票\n")

    factors = {
        "momentum":  momentum(prices, lookback=252, skip=21),
        "reversal":  reversal(prices, window=21),
        "lowvol":   -volatility(prices, window=60),
    }

    bt = Backtester(commission=0.0003, slippage=0.0001)
    all_metrics = {}

    # 1. 单因子策略 (每个因子单独 top 30% 等权)
    for name, factor in factors.items():
        strat = MultiFactorStrategy(
            factors={name: factor},
            combine="equal",
            top_pct=0.3,
            rebalance_days=21,
        )
        weights = strat.generate_weights(prices)
        result = bt.run(prices, weights)
        all_metrics[f"single_{name}"] = result.metrics

    # 2. equal 合成
    strat_eq = MultiFactorStrategy(factors, combine="equal", top_pct=0.3, rebalance_days=21)
    weights_eq = strat_eq.generate_weights(prices)
    all_metrics["combo_equal"] = bt.run(prices, weights_eq).metrics

    # 3. ic_weighted 合成
    strat_ic = MultiFactorStrategy(
        factors, combine="ic_weighted",
        ic_lookback=60, ic_horizon=5,
        top_pct=0.3, rebalance_days=21,
    )
    weights_ic = strat_ic.generate_weights(prices)
    all_metrics["combo_ic_weighted"] = bt.run(prices, weights_ic).metrics

    # 4. 等权买入持有基准
    bench_w = equal_weight_benchmark(prices)
    all_metrics["benchmark_equal_hold"] = bt.run(prices, bench_w).metrics

    print("===== 策略对照 =====")
    comp = pd.DataFrame(all_metrics).T
    cols = ["年化收益", "夏普比率", "最大回撤", "Calmar比率", "平均换手率(单边)"]
    print(comp[cols].round(4).to_string())

    print("\n[ic_weighted 模式下因子权重 (末 5 个调仓日)]")
    if strat_ic.weight_log_ is not None:
        print(strat_ic.weight_log_.dropna().tail(5).round(3).to_string())

    print("\n核心结论 (放到真实数据上才更有意义):")
    print("  - 多因子合成的目标不是'最高夏普', 而是'更稳' (低 IC 相关 → 抵消单因子失效)")
    print("  - equal 是非常硬的基准, ic_weighted 没明显赢就不要用它")
    print("  - 永远和等权买入持有对比, 跑不过它就说明因子+组合都没增值")


if __name__ == "__main__":
    main()
