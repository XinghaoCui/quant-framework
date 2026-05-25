"""
横截面动量策略（Cross-Sectional Momentum）。

思路:
    在某个时间点，把全体股票按"过去 N 个月的收益率"排序，
    买入收益最高的一组（赢家），卖空收益最低的一组（输家），
    持有 M 个月后调仓。

学术起源:
    Jegadeesh & Titman (1993) 的著名论文 "Returns to Buying Winners
    and Selling Losers" 首次系统证明：过去 3-12 个月的赢家在未来 3-12 个月
    继续跑赢输家。这个现象至今仍是因子投资的核心实证之一。

经典参数:
    形成期 (lookback) = 12 个月（但跳过最近 1 个月，避免短期反转）
    持有期 (holding) = 1 个月
    赢家组 = 收益率前 30%，输家组 = 后 30%

为什么跳过最近 1 个月（skip）?
    短期（1 个月内）存在"反转效应"——刚涨多了的股票短期会回调。
    所以"过去 12 个月动量"通常指 t-12 到 t-1 个月，不含最近 1 个月。
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class MomentumStrategy(Strategy):
    """
    横截面动量策略。

    参数:
        lookback_days: 形成期长度（交易日），默认 252（约 12 个月）
        skip_days: 跳过最近 N 天，默认 21（约 1 个月）
        top_pct: 买入前 top_pct 的赢家，默认 0.3（30%）
        bottom_pct: 卖空后 bottom_pct 的输家，默认 0.3
        rebalance_days: 调仓周期，默认 21（约 1 个月）
        allow_short: 是否做空输家组
    """

    name = "Momentum"

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        top_pct: float = 0.3,
        bottom_pct: float = 0.3,
        rebalance_days: int = 21,
        allow_short: bool = True,
    ):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.rebalance_days = rebalance_days
        self.allow_short = allow_short

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # 形成期收益 = (t - skip) 时刻的价格 / (t - skip - lookback) 时刻的价格 - 1
        # 用 shift 实现"跳过最近 skip 天"
        past_prices = prices.shift(self.skip_days)
        formation_return = past_prices / past_prices.shift(self.lookback_days) - 1.0

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # 在调仓日才计算新权重，其他日子保持上次的权重
        rebalance_dates = prices.index[::self.rebalance_days]

        for date in rebalance_dates:
            if date not in formation_return.index:
                continue
            scores = formation_return.loc[date].dropna()
            if len(scores) < 4:  # 标的太少，没法排序
                continue

            n = len(scores)
            n_top = max(1, int(n * self.top_pct))
            n_bot = max(1, int(n * self.bottom_pct))

            winners = scores.nlargest(n_top).index
            losers = scores.nsmallest(n_bot).index

            row = pd.Series(0.0, index=prices.columns)
            row[winners] = 1.0 / n_top  # 赢家组等权多头
            if self.allow_short:
                row[losers] = -1.0 / n_bot  # 输家组等权空头

            weights.loc[date] = row

        # 前向填充：调仓日之间保持上次仓位
        weights = weights.replace(0, pd.NA).ffill().fillna(0.0).infer_objects(copy=False)
        return weights
