"""
多因子合成策略
==============

单因子能挣钱, 但 IC 通常只有 0.03~0.08, 信号弱且容易失效.
把几个**相关性低**的因子合成一个综合分, 能让信号更强、更稳:

    综合分 = w1 * 标准化(因子1) + w2 * 标准化(因子2) + ...

为什么必须先标准化?
    动量因子可能在 -50% ~ +200% 之间; RSI 在 0~100 之间; 波动率在 0~1.
    量级差异巨大, 不标准化就直接相加, 等于让大量级的因子独占话语权.
    标准化把每个因子拉回同一个尺度 (zscore: 均值 0、标准差 1).

权重怎么定 (两种主流做法):
    1. equal      : 每个因子等权. 简单稳健, 不需要参数调优.
    2. ic_weighted: 用每个因子最近 N 天的 IC 加权, IC 高的权重高.
                    更"聪明"但要小心 IC 估计噪声大、过拟合.

学术与业界都验证: equal 加权在样本外通常打不过, 但也输不太多 (Bayesian
shrinkage 的常识 —— 等权是对所有 IC 估计的极强收缩, 反而避免噪声).
新手强烈建议先 equal, 再试 ic_weighted, 对比看是否真的提升.

详细原理见 `思考与学习/07_因子研究方法论/05_多因子合成.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from ..factors.evaluation import compute_ic, forward_returns


def cross_section_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    """
    每天做一次截面 zscore 标准化: (x - 截面均值) / 截面标准差.

    标准化必须按日做, 不能用整个序列的全局均值/标准差 ——
    那样会引入未来信息 (今天的 zscore 用到了未来才知道的均值).
    """
    mean = factor.mean(axis=1)
    std = factor.std(axis=1)
    # broadcasting: (T, N) - (T,) → 每列减各自日期的均值
    return factor.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


class MultiFactorStrategy(Strategy):
    """
    多因子合成 → 排序选股策略.

    参数:
        factors          : dict[str, DataFrame], 因子名 -> 因子宽表
                           约定 "值越大越看多"; 如果因子是 "值越大越看空"
                           (如波动率), 在传入前自己取负号.
        combine          : "equal" 等权 / "ic_weighted" IC 滚动加权
        ic_lookback      : ic_weighted 模式下, 算 IC 的滚动窗 (默认 60)
        ic_horizon       : ic_weighted 模式下, IC 用的 forward 天数 (默认 5)
        top_pct          : 买入综合分前 top_pct 的股票
        bottom_pct       : 卖空综合分后 bottom_pct 的股票 (allow_short=True 才生效)
        rebalance_days   : 调仓周期
        allow_short      : 是否做空底部组
        min_stocks       : 截面有效股票数下限, 不足则当日不持仓

    用法:
        from src.factors import momentum, reversal, volatility
        factors = {
            "momentum":  momentum(prices, lookback=252, skip=21),
            "reversal":  reversal(prices, window=21),
            "lowvol":   -volatility(prices, window=60),  # 低波动好, 取负号
        }
        strat = MultiFactorStrategy(factors, combine="equal", top_pct=0.3)
        weights = strat.generate_weights(prices)
    """

    name = "MultiFactor"

    def __init__(
        self,
        factors: dict[str, pd.DataFrame],
        combine: str = "equal",
        ic_lookback: int = 60,
        ic_horizon: int = 5,
        top_pct: float = 0.3,
        bottom_pct: float = 0.0,
        rebalance_days: int = 5,
        allow_short: bool = False,
        min_stocks: int = 5,
    ):
        if combine not in {"equal", "ic_weighted"}:
            raise ValueError(f"combine 必须是 'equal' 或 'ic_weighted', 收到 {combine!r}")
        if not factors:
            raise ValueError("factors 不能为空")

        self.factors = factors
        self.combine = combine
        self.ic_lookback = ic_lookback
        self.ic_horizon = ic_horizon
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.rebalance_days = rebalance_days
        self.allow_short = allow_short
        self.min_stocks = min_stocks

        # 训练后可读: 每个调仓日各因子的权重 (ic_weighted 模式才有意义)
        self.weight_log_: pd.DataFrame | None = None

    # ---------- 内部: 合成综合分 ----------

    def _composite_score(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        把所有因子标准化, 按 combine 方法加权合成.
        equal 模式权重恒为 1/k; ic_weighted 模式权重按滚动 IC.
        """
        # 1. 每个因子先做截面 zscore, 拉到同尺度
        zs = {
            name: cross_section_zscore(f.reindex_like(prices))
            for name, f in self.factors.items()
        }

        if self.combine == "equal":
            # 等权: 平均所有 zscore (NaN 自动跳过)
            stacked = pd.concat(zs.values(), axis=0, keys=zs.keys())
            composite = stacked.groupby(level=1).mean()
            composite.index = prices.index
            self.weight_log_ = None
            return composite

        # ic_weighted: 每天用过去 ic_lookback 天的 IC 加权
        return self._ic_weighted_composite(prices, zs)

    def _ic_weighted_composite(
        self, prices: pd.DataFrame, zs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        fwd = forward_returns(prices, self.ic_horizon)

        # 一次性把每个因子的 daily IC 时序算出来 (用 zscore 后的版本, 排序不变)
        ic_series = {name: compute_ic(z, fwd) for name, z in zs.items()}
        ic_df = pd.DataFrame(ic_series).reindex(prices.index)

        # 滚动 IC 均值: 这一天能"知道"的 IC 只到 t-ic_horizon
        # (因为 t-ic_horizon 天的 IC 用到了 t 天的真实收益, 已经发生)
        # 严谨做法是 shift(ic_horizon), 防止用到未来收益
        rolling_ic = ic_df.rolling(self.ic_lookback, min_periods=10).mean().shift(self.ic_horizon)

        # 权重: 按 |IC| 归一化, 保留 IC 符号 (允许负 IC 因子被反向使用)
        abs_ic = rolling_ic.abs()
        denom = abs_ic.sum(axis=1).replace(0, np.nan)
        weights = rolling_ic.div(denom, axis=0).fillna(1.0 / len(zs))

        # 加权合成
        composite = sum(
            zs[name].mul(weights[name], axis=0) for name in zs
        )
        self.weight_log_ = weights
        return composite

    # ---------- Strategy 主接口 ----------

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        composite = self._composite_score(prices)
        # 用 NaN 占位 (不是 0!) —— 这样 ffill 时不会把 "本次没选的票" 误当成
        # "保留之前持仓". 整行更新 + ffill 才是 "调仓日之间保持上次组合" 的正确语义.
        weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

        rebalance_dates = prices.index[::self.rebalance_days]

        for date in rebalance_dates:
            if date not in composite.index:
                continue
            scores = composite.loc[date].dropna()
            if len(scores) < self.min_stocks:
                continue

            n = len(scores)
            n_top = max(1, int(n * self.top_pct))
            n_bot = max(1, int(n * self.bottom_pct)) if self.allow_short else 0

            row = pd.Series(0.0, index=prices.columns)
            winners = scores.nlargest(n_top).index
            row[winners] = 1.0 / n_top
            if n_bot > 0:
                losers = scores.nsmallest(n_bot).index
                row[losers] = -1.0 / n_bot

            weights.loc[date] = row

        # 调仓日之间整行复制上一个调仓日; 开头还没调过仓的日子用 0 (空仓)
        weights = weights.ffill().fillna(0.0)
        return weights
