"""
技术因子库。

什么是因子?
    因子（Factor）是"对每只股票的某种数值打分"，
    高分股票理论上未来收益与低分股票有系统性差异。
    例如：
        - 动量因子高 → 未来收益可能更高（动量效应）
        - 波动率因子高 → 未来收益可能更低（低波动异象）

因子的两个用法:
    1. 直接做策略：按因子排序选股
    2. 组合成多因子模型：把多个因子加权打分

下面这些都是"纯价格因子"，不需要基本面数据。
基本面因子（PE/PB/ROE）需要财报数据，将来扩展数据层后再加。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """简单移动平均"""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    指数移动平均。
    EMA 比 SMA 反应更快，因为对近期数据赋予更大权重。
    alpha = 2 / (window + 1) 是 pandas 默认的"span"参数。
    """
    return prices.ewm(span=window, adjust=False).mean()


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index, 相对强弱指数)。
    取值 0-100：>70 通常视为超买，<30 视为超卖。

    计算：
        gain = max(close - close_prev, 0) 的 EMA
        loss = max(close_prev - close, 0) 的 EMA
        RS = gain / loss
        RSI = 100 - 100 / (1 + RS)
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR (Average True Range, 平均真实波幅)。
    衡量价格波动幅度的绝对值（非百分比）。
    常用于止损位设置：止损 = 入场价 - 2 * ATR。

    True Range = max(high-low, |high-close_prev|, |low-close_prev|)
    """
    close_prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=0).groupby(level=0).max()
    return tr.rolling(window, min_periods=window).mean()


def bollinger_bands(
    prices: pd.DataFrame, window: int = 20, n_std: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    布林带。
    中轨 = 20 日 SMA
    上轨 = 中轨 + 2 倍标准差
    下轨 = 中轨 - 2 倍标准差

    用法：价格触上轨 → 超买；触下轨 → 超卖；带宽收缩 → 即将放量突破。
    """
    mid = sma(prices, window)
    std = prices.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    动量因子 = (t-skip) 价格 / (t-skip-lookback) 价格 - 1
    跳过最近 skip 天避免短期反转污染。
    """
    past = prices.shift(skip)
    return past / past.shift(lookback) - 1.0


def volatility(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    波动率因子 = 过去 window 天日收益率的标准差（已年化）。
    低波动股票长期跑赢高波动股票，这叫"低波动异象"。
    """
    rets = prices.pct_change()
    return rets.rolling(window, min_periods=window).std() * np.sqrt(252)


def reversal(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    短期反转因子 = -过去 window 天收益。
    短期（1 个月内）涨多的会回调，跌多的会反弹。
    """
    return -(prices / prices.shift(window) - 1.0)
