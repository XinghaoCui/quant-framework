"""
特征工程与标签构造。

机器学习的核心是 (X, y)：
    X = 特征矩阵，每行一个"样本"，每列一个"特征"
    y = 标签，我们想预测的目标

在量化里：
    一个"样本" = (某只股票, 某个时间点)
    特征 X = 这只股票在这个时间点的各种因子值（动量、波动率、RSI...）
    标签 y = 这只股票未来 N 天的收益率（回归）或涨跌方向（分类）

最关键的一点（新手必踩的坑）：
    特征必须是"t 时刻能知道的信息"，
    标签必须是"t 时刻之后才发生的事"。
    如果特征里混入了未来信息 → 数据泄露 → 回测完美、实盘崩盘。
    详见 思考与学习/06_机器学习入门/03_金融数据的坑.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 复用已有的因子库
from ..factors import momentum, volatility, reversal, rsi, sma, ema


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    把价格宽表转成"长表特征矩阵"。

    输入:
        prices: 宽表，index=date, columns=symbol

    输出:
        长表 DataFrame，MultiIndex=(date, symbol)，每列一个特征。
        每一行就是一个 ML 样本：某天某只票的所有因子值。

    为什么转长表？
        sklearn 的模型接受的是 (n_samples, n_features) 的二维数组。
        宽表是 (date, symbol) 的价格，没法直接喂给模型。
        长表把每个 (date, symbol) 拉成一行，正好是一个样本。
    """
    feats = {}

    # 各种技术因子（都只用历史价格，无未来信息）
    feats["mom_21"] = momentum(prices, lookback=21, skip=0)       # 1 月动量
    feats["mom_63"] = momentum(prices, lookback=63, skip=0)       # 3 月动量
    feats["mom_252_21"] = momentum(prices, lookback=252, skip=21) # 经典 12-1 动量
    feats["vol_21"] = volatility(prices, window=21)               # 1 月波动率
    feats["vol_63"] = volatility(prices, window=63)               # 3 月波动率
    feats["reversal_5"] = reversal(prices, window=5)              # 1 周反转
    feats["rsi_14"] = rsi(prices, window=14)                      # RSI
    # 价格相对均线的位置（>0 在均线上方）
    feats["px_vs_sma20"] = prices / sma(prices, 20) - 1
    feats["px_vs_sma60"] = prices / sma(prices, 60) - 1
    # 均线斜率（趋势方向）
    feats["sma20_slope"] = sma(prices, 20).pct_change(5)

    # 把每个因子（宽表）堆叠成长表，再横向拼成特征矩阵
    long_frames = []
    for name, wide in feats.items():
        s = wide.stack()                  # MultiIndex (date, symbol) -> 一列
        s.name = name
        long_frames.append(s)

    X = pd.concat(long_frames, axis=1)
    X.index.names = ["date", "symbol"]
    return X


def build_label(
    prices: pd.DataFrame,
    horizon: int = 5,
    kind: str = "return",
) -> pd.Series:
    """
    构造标签 y：未来 horizon 天的收益率。

    参数:
        horizon: 预测未来多少天的收益
        kind: "return" 回归（连续收益率） / "binary" 分类（涨=1 跌=0）

    关键：用 shift(-horizon)，即"未来"的价格除以"现在"的价格。
        forward_return(t) = P(t+horizon) / P(t) - 1
    这是 t 时刻之后才知道的，正是我们要预测的目标。
    """
    forward_return = prices.shift(-horizon) / prices - 1.0

    label = forward_return.stack()
    label.index.names = ["date", "symbol"]

    if kind == "binary":
        label = (label > 0).astype(int)
    label.name = f"y_{kind}_{horizon}d"
    return label


def make_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    kind: str = "return",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    一步到位：返回对齐好的 (X, y)，去掉含 NaN 的样本。

    返回:
        X: 特征矩阵 (n_samples, n_features)，MultiIndex=(date, symbol)
        y: 标签 (n_samples,)
    """
    X = build_features(prices)
    y = build_label(prices, horizon=horizon, kind=kind)

    # 对齐索引（取交集）
    df = X.join(y, how="inner")
    df = df.dropna()  # 丢掉任何含 NaN 的样本（前期因子未成形 + 末期标签缺失）

    y_out = df[y.name]
    X_out = df.drop(columns=[y.name])
    return X_out, y_out
