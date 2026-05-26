"""
示例 43: 技术 + 基本面多因子组合
=================================

研究问题: 加入基本面因子后, 能不能让纯技术因子的策略更稳?

对比 4 个策略 (都用月度调仓 + 选 top 30%):
    A. 纯技术: 动量 + 反转 + 低波
    B. 纯基本面: ROE + EP + 成长
    C. 技术 + 基本面 等权合成
    D. 技术 + 基本面 IC 加权合成
    + 基准: 等权买入持有

预期:
    - 单因子家族都能挣钱, 但组合家族波动 / 回撤会更稳
    - IC 加权可能优于等权 (低风险因子 + 模拟数据上 alpha 集中)
    - 真实数据上技术 + 基本面通常 IC 相关性 < 0.2, 合成增量很大
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.factors import (
    momentum, reversal, volatility,
    roe_factor, earnings_yield, profit_growth_factor,
)
from src.strategies import MultiFactorStrategy


def equal_weight_benchmark(prices):
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2020-01-01", "2024-12-31", seed=42)
    print(f"数据: {prices.shape[0]} 天 x {prices.shape[1]} 只票\n")

    # 三组因子
    tech_factors = {
        "momentum":  momentum(prices, lookback=252, skip=21),
        "reversal":  reversal(prices, window=21),
        "lowvol":   -volatility(prices, window=60),
    }
    fund_factors = {
        "ROE":           roe_factor(prices, use_synthetic=True),
        "EP":            earnings_yield(prices, use_synthetic=True),
        "profit_growth": profit_growth_factor(prices, use_synthetic=True),
    }
    combo_factors = {**tech_factors, **fund_factors}

    bt = Backtester(commission=0.0003, slippage=0.0001)

    strategies = {
        "A_tech_only_equal":
            MultiFactorStrategy(tech_factors, combine="equal", top_pct=0.3, rebalance_days=21),
        "B_fund_only_equal":
            MultiFactorStrategy(fund_factors, combine="equal", top_pct=0.3, rebalance_days=21),
        "C_combo_equal":
            MultiFactorStrategy(combo_factors, combine="equal", top_pct=0.3, rebalance_days=21),
        "D_combo_ic_weighted":
            MultiFactorStrategy(combo_factors, combine="ic_weighted",
                                ic_lookback=60, ic_horizon=5,
                                top_pct=0.3, rebalance_days=21),
    }

    all_metrics = {}
    for name, strat in strategies.items():
        weights = strat.generate_weights(prices)
        result = bt.run(prices, weights)
        all_metrics[name] = result.metrics

    # 基准
    bench_w = equal_weight_benchmark(prices)
    all_metrics["benchmark_equal_hold"] = bt.run(prices, bench_w).metrics

    print("===== 策略对照表 =====")
    comp = pd.DataFrame(all_metrics).T
    cols = ["年化收益", "夏普比率", "最大回撤", "Calmar比率", "平均换手率(单边)"]
    print(comp[cols].round(4).to_string())

    print("\n[ic_weighted 模式下因子权重 (末 3 个调仓日)]")
    if strategies["D_combo_ic_weighted"].weight_log_ is not None:
        wl = strategies["D_combo_ic_weighted"].weight_log_.dropna()
        if len(wl) > 0:
            print(wl.tail(3).round(3).to_string())

    print("\n小白怎么解读:")
    print("  - A vs B: 纯技术 vs 纯基本面, 看哪一组在当前数据上 alpha 更强")
    print("  - C 等权合成: 一定比基准强 (因为同时利用 2 类信息)")
    print("  - D IC 加权: ic_weighted 会自动给负 IC 因子负权重, 理论上更精")
    print("  - 真实数据上 C/D 通常显著优于 A 和 B, 多因子合成的价值就在这里")
    print("\n下一步可玩:")
    print("  - 配合 08 章组合优化: 把 D 的 scores 喂给 OptimizedStrategy 跑 min_var")
    print("  - 调 top_pct / rebalance_days, 看怎样平衡换手成本与信号强度")


if __name__ == "__main__":
    main()
