"""
示例 32: 把组合优化接入完整策略回测
====================================

研究问题: 用多因子打分选股 + 不同方式分仓位, 哪种回测表现最好?

对比 5 种策略:
    A. 单因子动量 (基准)
    B. 多因子合成 + 等权              (MultiFactorStrategy)
    C. 多因子合成 + 反波动率分仓       (OptimizedStrategy inverse_vol)
    D. 多因子合成 + 风险平价分仓       (OptimizedStrategy risk_parity)
    E. 多因子合成 + 最小方差分仓       (OptimizedStrategy min_variance)

回答的问题: 在打分相同的情况下, "怎么分钱"能不能让回测更稳?

预期:
    - 模拟数据噪声大, 差距不会很惊艳
    - 但你能看到 min_variance 通常波动最小, 夏普可能更高
    - 等权依然是难打的基准
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.engine import Backtester
from src.factors import momentum, reversal, volatility
from src.strategies import MultiFactorStrategy, OptimizedStrategy
from src.strategies.multi_factor import cross_section_zscore


def make_composite_score(prices: pd.DataFrame) -> pd.DataFrame:
    """构造跟 MultiFactorStrategy(equal) 同口径的综合打分, 让对比公平."""
    factors = {
        "momentum":  momentum(prices, lookback=252, skip=21),
        "reversal":  reversal(prices, window=21),
        "lowvol":   -volatility(prices, window=60),
    }
    zs = [cross_section_zscore(f.reindex_like(prices)) for f in factors.values()]
    # 等权合成 (NaN 自动跳过)
    return sum(zs) / len(zs)


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2024-12-31", seed=42)
    print(f"数据: {prices.shape[0]} 天 x {prices.shape[1]} 只票\n")

    # 共用的多因子综合打分
    composite = make_composite_score(prices)

    bt = Backtester(commission=0.0003, slippage=0.0001)
    factors = {
        "momentum":  momentum(prices, lookback=252, skip=21),
        "reversal":  reversal(prices, window=21),
        "lowvol":   -volatility(prices, window=60),
    }

    strategies = {
        "single_momentum":
            MultiFactorStrategy({"momentum": factors["momentum"]}, combine="equal",
                                top_pct=0.3, rebalance_days=21),
        "multi_equal":
            MultiFactorStrategy(factors, combine="equal", top_pct=0.3, rebalance_days=21),
        "multi_inv_vol":
            OptimizedStrategy(scores=composite, optimizer="inverse_vol",
                              top_pct=0.3, rebalance_days=21, lookback_days=60),
        "multi_risk_parity":
            OptimizedStrategy(scores=composite, optimizer="risk_parity",
                              top_pct=0.3, rebalance_days=21, lookback_days=60),
        "multi_min_var":
            OptimizedStrategy(scores=composite, optimizer="min_variance",
                              top_pct=0.3, rebalance_days=21, lookback_days=60,
                              max_weight=0.4),  # 防止集中
    }

    all_metrics = {}
    for name, strat in strategies.items():
        weights = strat.generate_weights(prices)
        result = bt.run(prices, weights)
        all_metrics[name] = result.metrics

    print("===== 策略对照表 =====")
    comp = pd.DataFrame(all_metrics).T
    cols = ["年化收益", "夏普比率", "最大回撤", "Calmar比率", "平均换手率(单边)"]
    print(comp[cols].round(4).to_string())

    print("\n小白怎么解读:")
    print("  1. 比 'multi_equal' 和 'multi_min_var' 的 [波动 = 1/夏普×收益]:")
    print("     min_var 通常波动更小, 但收益也可能小一点")
    print("  2. 看 '最大回撤': 优化方法的回撤往往比等权更可控")
    print("  3. 看 '换手率': 优化方法每次重算权重, 换手率可能比等权高")
    print("  4. 没有绝对赢家. 实战要看你的目标 (求稳 / 求高 / 求大容量)")


if __name__ == "__main__":
    main()
