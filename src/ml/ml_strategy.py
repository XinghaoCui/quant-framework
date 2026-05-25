"""
把 ML 模型预测接入策略框架。

设计：MLStrategy 继承 Strategy，实现 generate_weights。
    内部流程：
        1. 用训练期数据训练模型
        2. 在每个交易日，用模型预测各股票的未来收益
        3. 按预测值排序，做多预测最高的一组（可选做空最低组）

关键：训练/预测的时序边界
    - 模型只在 train_end 之前的数据上训练一次（避免每天重训的开销）
    - 之后所有日期都用这个模型预测
    - 更严谨的做法是滚动重训（walk-forward），这里为了清晰先用"训练一次"
      想做滚动重训见 思考与学习/06_机器学习入门/08_完整工作流.md
"""

from __future__ import annotations

import pandas as pd

from ..strategies.base import Strategy
from .features import build_features, make_dataset
from .dataset import time_series_split
from .models import ModelWrapper


class MLStrategy(Strategy):
    """
    机器学习选股策略。

    参数:
        model_name: "ridge" / "rf" / "gbdt" 等，见 models.make_model
        task: "regression"（预测收益） / "classification"（预测涨跌）
        horizon: 预测未来多少天的收益
        top_pct: 做多预测最高的比例
        bottom_pct: 做空预测最低的比例
        allow_short: 是否做空
        rebalance_days: 调仓周期
        train_ratio: 用前多少比例的数据训练（其余用于预测/回测）
    """

    name = "MLStrategy"

    def __init__(
        self,
        model_name: str = "gbdt",
        task: str = "regression",
        horizon: int = 5,
        top_pct: float = 0.3,
        bottom_pct: float = 0.3,
        allow_short: bool = False,
        rebalance_days: int = 5,
        train_ratio: float = 0.5,
    ):
        self.model_name = model_name
        self.task = task
        self.horizon = horizon
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.allow_short = allow_short
        self.rebalance_days = rebalance_days
        self.train_ratio = train_ratio

        self.model: ModelWrapper | None = None
        self.predictions_: pd.Series | None = None

    def fit_predict(self, prices: pd.DataFrame) -> pd.Series:
        """
        训练模型并对全样本预测。返回预测值 Series，MultiIndex=(date, symbol)。
        预测只覆盖测试期（训练期不交易，避免用样本内信号）。
        """
        kind = "binary" if self.task == "classification" else "return"
        X, y = make_dataset(prices, horizon=self.horizon, kind=kind)

        X_train, y_train, X_test, y_test = time_series_split(
            X, y, test_size=(1 - self.train_ratio), embargo_days=self.horizon
        )

        self.model = ModelWrapper(self.model_name, task=self.task)
        self.model.fit(X_train, y_train)

        self.predictions_ = self.model.predict(X_test)
        return self.predictions_

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        # 训练 + 预测（只在测试期产生信号）
        predictions = self.fit_predict(prices)

        # 把长表预测 (date, symbol) 还原成宽表 (date × symbol)
        pred_wide = predictions.unstack("symbol")
        pred_wide = pred_wide.reindex(index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        rebalance_dates = pred_wide.dropna(how="all").index[::self.rebalance_days]

        for date in rebalance_dates:
            scores = pred_wide.loc[date].dropna()
            if len(scores) < 4:
                continue
            n_top = max(1, int(len(scores) * self.top_pct))
            longs = scores.nlargest(n_top).index

            row = pd.Series(0.0, index=prices.columns)
            row[longs] = 1.0 / n_top
            if self.allow_short:
                n_bot = max(1, int(len(scores) * self.bottom_pct))
                shorts = scores.nsmallest(n_bot).index
                row[shorts] = -1.0 / n_bot

            weights.loc[date] = row

        weights = weights.replace(0, pd.NA).ffill().fillna(0.0).infer_objects(copy=False)
        return weights
