"""
双均线策略（SMA Cross / Moving Average Crossover）。

思路:
    短期均线代表"近期趋势"，长期均线代表"长期趋势"。
    短期上穿长期 → 多头趋势确立 → 买入
    短期下穿长期 → 空头趋势确立 → 卖出（或空仓）

为什么有效（理论上）:
    市场存在"趋势"现象，价格会沿某方向延续一段时间（动量效应）。
    均线是噪声平滑后的趋势指示器。

何时失效:
    震荡市（价格上下穿插）会反复触发买卖，被来回打脸 + 交易成本拖累。
    所以双均线在牛熊分明时表现好，震荡市表现差。

这是个"经典中的经典"，几乎每本量化书都讲，但实盘很少单独使用——
通常作为"基础信号"和其他过滤器组合（如波动率过滤、ADX 过滤）。
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class SmaCrossStrategy(Strategy):
    """
    双均线策略。

    参数:
        fast: 短期均线窗口，默认 20 日
        slow: 长期均线窗口，默认 60 日
        allow_short: 是否允许做空。True 时，短下穿长开空仓；False 时只持多或空仓

    输出权重:
        多头信号 → 等权分配在所有标的上
        空头信号且 allow_short=True → 等权做空
        否则 → 空仓
    """

    name = "SmaCross"

    def __init__(self, fast: int = 20, slow: int = 60, allow_short: bool = False):
        if fast >= slow:
            raise ValueError(f"fast({fast}) 必须小于 slow({slow})")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # 计算两条均线
        fast_ma = prices.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = prices.rolling(self.slow, min_periods=self.slow).mean()

        # 信号矩阵：+1 多头 / -1 空头 / 0 空仓
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[fast_ma > slow_ma] = 1.0
        if self.allow_short:
            signal[fast_ma < slow_ma] = -1.0

        # 等权分配：每天有信号的标的数 N，每个权重 = signal / N
        # 防止某天没信号时除零
        n_active = signal.abs().sum(axis=1).replace(0, 1)
        weights = signal.div(n_active, axis=0)

        return weights
