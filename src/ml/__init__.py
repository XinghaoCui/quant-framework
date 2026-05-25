"""
机器学习模块 - 把 ML 接入量化策略框架。

子模块：
    features    : 特征工程（把价格/因子变成特征矩阵 X）和标签构造（y）
    dataset     : 防数据泄露的时序数据划分（train/test split, purged CV）
    models      : 统一接口封装 sklearn 模型
    evaluation  : IC、分组回测、模型评估
    ml_strategy : 把 ML 预测接入 Strategy 框架
"""

from .features import build_features, build_label, make_dataset
from .dataset import time_series_split, purged_kfold
from .models import make_model, ModelWrapper
from .evaluation import information_coefficient, quantile_backtest

__all__ = [
    "build_features",
    "build_label",
    "make_dataset",
    "time_series_split",
    "purged_kfold",
    "make_model",
    "ModelWrapper",
    "information_coefficient",
    "quantile_backtest",
]
