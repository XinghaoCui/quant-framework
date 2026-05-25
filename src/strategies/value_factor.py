"""
价值因子策略（Value Factor）。

经典价值因子:
    PB (市净率)、PE (市盈率)、PS (市销率)、EV/EBITDA 等。
    思路：买入"便宜"的股票（低 PB/PE），卖出"贵"的股票。

学术起源:
    Fama-French (1992) 三因子模型把价值因子（HML, High Minus Low）
    确立为驱动股票收益的核心系统性因子之一。

简化实现:
    完整的价值策略需要基本面数据（财报）。本框架定位是"通用回测引擎"，
    数据层只有价格。所以这里用一个**价格代理因子**：
        反转因子 = -过去 N 日累计收益
    意思是"过去跌得多的现在便宜"，作为"价值"的非常粗糙的代理。

    真正的价值策略请用真实 PB/PE 数据。这里主要展示因子框架的写法。
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class ValueFactorStrategy(Strategy):
    """
    价值因子策略（简化版：用反转作为价值代理）。

    参数:
        lookback_days: 反转期长度，默认 252（约 1 年）
        top_pct: 买入"便宜"组比例（过去跌最多的）
        rebalance_days: 调仓周期
    """

    name = "ValueFactor"

    def __init__(
        self,
        lookback_days: int = 252,
        top_pct: float = 0.3,
        rebalance_days: int = 21,
    ):
        self.lookback_days = lookback_days
        self.top_pct = top_pct
        self.rebalance_days = rebalance_days

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # 反转分数 = -过去 N 日收益（跌得多 = 高分 = 便宜）
        past_return = prices / prices.shift(self.lookback_days) - 1.0
        value_score = -past_return

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        rebalance_dates = prices.index[::self.rebalance_days]

        for date in rebalance_dates:
            if date not in value_score.index:
                continue
            scores = value_score.loc[date].dropna()
            if len(scores) < 4:
                continue

            n_top = max(1, int(len(scores) * self.top_pct))
            cheap = scores.nlargest(n_top).index

            row = pd.Series(0.0, index=prices.columns)
            row[cheap] = 1.0 / n_top
            weights.loc[date] = row

        weights = weights.replace(0, pd.NA).ffill().fillna(0.0).infer_objects(copy=False)
        return weights
