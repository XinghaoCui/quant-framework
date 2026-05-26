"""
组合优化策略 (OptimizedStrategy).

跟 MultiFactorStrategy 一样, 都是 "打分 -> 选 top N -> 分仓位" 的范式,
区别只在最后一步:
    MultiFactorStrategy  最后一步是 "等权" (每只 1/N)
    OptimizedStrategy    最后一步用组合优化器: 最小方差 / 风险平价 / 反波动率

什么时候用这个?
    - 你已经能产出某种"每只票的打分" (因子合成、ML 预测、自定义信号都行)
    - 你想验证: 不等权分仓能不能让组合更稳

输入:
    scores  DataFrame, index=date, columns=symbol. 每天每只票的打分, 越大越看多.
    其他参数都在 __init__ 时配置.

输出:
    generate_weights(prices) 返回 (date × symbol) 的权重矩阵, 直接丢给 Backtester.

为什么用 scores + prices 两个输入?
    scores 决定 "选哪 N 只", prices 决定 "用过去多少天的收益算协方差".
    两者分开传, 让你可以自由切换打分来源.

小白提示:
    第一次用先选 optimizer="min_variance", 它最稳健, 不需要预测收益.
    跑完和 MultiFactorStrategy(equal) 对比, 看夏普和回撤有没有改善.
    没明显改善就不要用复杂方法 —— 这是组合优化的铁律.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from ..portfolio.optimizer import (
    equal_weights,
    inverse_volatility_weights,
    risk_parity_weights,
    min_variance_weights,
)


# 哪些 optimizer 只看协方差, 不需要 expected_returns
_COV_ONLY_METHODS = {
    "equal": equal_weights,
    "inverse_vol": inverse_volatility_weights,
    "risk_parity": risk_parity_weights,
    "min_variance": min_variance_weights,
}


class OptimizedStrategy(Strategy):
    """
    打分 -> 选 top -> 组合优化分仓.

    参数:
        scores           : DataFrame (date × symbol), 每天打分, 大 = 看多
        optimizer        : "equal" / "inverse_vol" / "risk_parity" / "min_variance"
        top_pct          : 取打分前 top_pct 的股票进入候选
        rebalance_days   : 调仓周期 (天)
        lookback_days    : 算协方差用的历史窗口 (默认 60 天 ≈ 一季度)
        min_stocks       : 候选数下限, 不足则当日空仓
        max_weight       : 单只权重上限 (None = 无)

    用法:
        from src.factors import momentum, reversal
        # 自己造一个综合打分 (可以来自因子、ML 模型, 任何来源)
        score = momentum(prices) + reversal(prices)

        strat = OptimizedStrategy(
            scores=score,
            optimizer="min_variance",
            top_pct=0.3,
            rebalance_days=21,
            lookback_days=60,
        )
        weights = strat.generate_weights(prices)
        result = Backtester().run(prices, weights)
    """

    name = "Optimized"

    def __init__(
        self,
        scores: pd.DataFrame,
        optimizer: str = "min_variance",
        top_pct: float = 0.3,
        rebalance_days: int = 21,
        lookback_days: int = 60,
        min_stocks: int = 5,
        max_weight: float | None = None,
    ):
        if optimizer not in _COV_ONLY_METHODS:
            raise ValueError(
                f"optimizer 必须是 {list(_COV_ONLY_METHODS)} 之一, 收到 {optimizer!r}. "
                "需要 expected_returns 的 max_sharpe / mean_variance 请直接调函数接口."
            )
        if top_pct <= 0 or top_pct > 1:
            raise ValueError(f"top_pct 必须在 (0, 1], 收到 {top_pct}")
        if not isinstance(scores, pd.DataFrame):
            raise TypeError("scores 必须是 DataFrame (index=date, columns=symbol)")

        self.scores = scores
        self.optimizer = optimizer
        self.top_pct = top_pct
        self.rebalance_days = rebalance_days
        self.lookback_days = lookback_days
        self.min_stocks = min_stocks
        self.max_weight = max_weight

        # 训练后可读: 每个调仓日的优化结果摘要 (方便诊断)
        self.opt_log_: list[dict] = []

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # 对齐打分与价格
        scores = self.scores.reindex_like(prices)
        asset_returns = prices.pct_change()

        # 与 MultiFactorStrategy 同样的"整行 NaN ffill"思想, 避免持仓越积越多
        weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        self.opt_log_ = []

        rebalance_dates = prices.index[::self.rebalance_days]
        opt_fn = _COV_ONLY_METHODS[self.optimizer]

        for date in rebalance_dates:
            if date not in scores.index:
                continue
            today_scores = scores.loc[date].dropna()
            if len(today_scores) < self.min_stocks:
                continue

            n = len(today_scores)
            n_top = max(self.min_stocks, int(n * self.top_pct))
            winners = today_scores.nlargest(n_top).index.tolist()

            # 取 lookback 天的历史收益, 算协方差
            hist = asset_returns.loc[:date, winners].tail(self.lookback_days).dropna(how="any")
            if len(hist) < max(20, self.min_stocks):
                # 历史不足, 退化成等权 (小白友好: 不让程序炸, 给个保守 fallback)
                row = pd.Series(0.0, index=prices.columns)
                row[winners] = 1.0 / len(winners)
                weights.loc[date] = row
                self.opt_log_.append({"date": date, "method": "fallback_equal", "n": len(winners)})
                continue

            try:
                res = opt_fn(hist)
            except Exception as e:
                # 数值不稳定时也退化, 不让回测中断
                row = pd.Series(0.0, index=prices.columns)
                row[winners] = 1.0 / len(winners)
                weights.loc[date] = row
                self.opt_log_.append(
                    {"date": date, "method": f"fallback_after_{self.optimizer}_error", "err": str(e)}
                )
                continue

            # 应用 max_weight 截断
            w_series = res.weights.copy()
            if self.max_weight is not None:
                w_series = w_series.clip(upper=self.max_weight)
                w_series = w_series / w_series.sum()  # 归一化

            row = pd.Series(0.0, index=prices.columns)
            row[w_series.index] = w_series.values
            weights.loc[date] = row

            self.opt_log_.append({
                "date": date,
                "method": self.optimizer,
                "n_top": len(winners),
                "expected_vol": res.expected_vol,
                "success": res.success,
            })

        # 调仓日之间整行复制上一个调仓日; 开头还没调过仓的日子用 0 (空仓)
        weights = weights.ffill().fillna(0.0)
        return weights

    def log_dataframe(self) -> pd.DataFrame:
        """把 opt_log_ 转成方便查看的 DataFrame."""
        if not self.opt_log_:
            return pd.DataFrame()
        return pd.DataFrame(self.opt_log_).set_index("date")
