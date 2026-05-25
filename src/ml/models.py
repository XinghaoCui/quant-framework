"""
模型封装 —— 用统一接口包住 sklearn 的各种模型。

为什么要封装？
    不同模型（线性回归、随机森林、梯度提升）的超参数完全不同，
    但在量化里我们关心的接口是一样的：fit / predict。
    封装后，策略代码里换模型只要改一个名字字符串，其他不动。

提供的模型（从简单到复杂）：
    "linear"  : 普通线性回归（最简单，可解释，速度快）
    "ridge"   : 岭回归（加 L2 正则，防过拟合）
    "lasso"   : Lasso（加 L1 正则，自动做特征选择）
    "logistic": 逻辑回归（分类：预测涨跌方向）
    "rf"      : 随机森林（非线性，抗过拟合，特征重要性）
    "gbdt"    : 梯度提升树（通常最强，类似 XGBoost/LightGBM）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_model(name: str = "ridge", task: str = "regression", **kwargs):
    """
    工厂函数：按名字创建一个 sklearn 模型（含标准化预处理）。

    参数:
        name: 模型名（见上方文档）
        task: "regression" 回归 / "classification" 分类
        kwargs: 透传给模型的超参数

    返回:
        sklearn Pipeline（标准化 + 模型）
    """
    name = name.lower()

    # 线性类模型对特征量纲敏感，必须先标准化；树模型不需要但加了也无害
    need_scale = name in ("linear", "ridge", "lasso", "logistic")

    # 注意：必须按需构造（if/elif），不能用字典一次性建全部模型，
    # 否则 kwargs（如 alpha）会被传给不接受它的模型而报错。
    if task == "regression":
        if name == "linear":
            estimator = LinearRegression(**kwargs)
        elif name == "ridge":
            estimator = Ridge(alpha=kwargs.pop("alpha", 1.0), **kwargs)
        elif name == "lasso":
            estimator = Lasso(alpha=kwargs.pop("alpha", 0.001), **kwargs)
        elif name == "rf":
            estimator = RandomForestRegressor(
                n_estimators=kwargs.pop("n_estimators", 200),
                max_depth=kwargs.pop("max_depth", 5),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 50),
                n_jobs=-1, random_state=42, **kwargs,
            )
        elif name == "gbdt":
            estimator = GradientBoostingRegressor(
                n_estimators=kwargs.pop("n_estimators", 200),
                max_depth=kwargs.pop("max_depth", 3),
                learning_rate=kwargs.pop("learning_rate", 0.05),
                subsample=kwargs.pop("subsample", 0.8),
                random_state=42, **kwargs,
            )
        else:
            raise ValueError(f"未知回归模型 '{name}'。可选: linear/ridge/lasso/rf/gbdt")
    elif task == "classification":
        if name == "logistic":
            estimator = LogisticRegression(max_iter=1000, **kwargs)
        elif name == "rf":
            estimator = RandomForestClassifier(
                n_estimators=kwargs.pop("n_estimators", 200),
                max_depth=kwargs.pop("max_depth", 5),
                min_samples_leaf=kwargs.pop("min_samples_leaf", 50),
                n_jobs=-1, random_state=42, **kwargs,
            )
        else:
            raise ValueError(f"未知分类模型 '{name}'。可选: logistic/rf")
    else:
        raise ValueError(f"未知 task '{task}'，应为 regression 或 classification")

    if need_scale:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator


class ModelWrapper:
    """
    把模型 + 训练 + 预测包成一个对象，方便接入策略。

    用法:
        wrapper = ModelWrapper("gbdt", task="regression")
        wrapper.fit(X_train, y_train)
        pred = wrapper.predict(X_test)          # 返回带索引的预测值 Series
        importance = wrapper.feature_importance()  # 特征重要性（如果模型支持）
    """

    def __init__(self, name: str = "ridge", task: str = "regression", **kwargs):
        self.name = name
        self.task = task
        self.model = make_model(name, task, **kwargs)
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelWrapper":
        self.feature_names_ = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """返回预测值，保留 X 的 (date, symbol) 索引。"""
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            # 分类时返回"上涨概率"，更适合排序
            pred = self.model.predict_proba(X.values)[:, 1]
        else:
            pred = self.model.predict(X.values)
        return pd.Series(pred, index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series | None:
        """返回特征重要性（树模型）或系数绝对值（线性模型）。"""
        est = self.model
        if isinstance(est, Pipeline):
            est = est.named_steps["model"]

        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
        elif hasattr(est, "coef_"):
            imp = np.abs(np.ravel(est.coef_))
        else:
            return None
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)
