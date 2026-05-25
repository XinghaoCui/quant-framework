"""
时序数据划分 —— 机器学习在量化里最容易出错、也最重要的一环。

普通 ML 用随机划分 (train_test_split shuffle=True)，但金融数据**绝对不能随机划分**！

为什么？
    随机划分会把"未来的样本"放进训练集，"过去的样本"放进测试集。
    模型在训练时"见过未来"，测试分数虚高，实盘必然崩盘。
    这是数据泄露 (data leakage) 的一种。

正确做法：
    1. 严格按时间切：训练集都在测试集之前
    2. 训练集和测试集之间留一段"隔离带"(embargo/purge)，
       因为标签是 horizon 天后的收益，最后几天的训练样本的标签
       会和测试集开头重叠，造成泄露。

详见 思考与学习/06_机器学习入门/03_金融数据的坑.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    embargo_days: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    按时间把数据切成训练集 + 测试集，中间留 embargo 隔离带。

    参数:
        X, y: make_dataset 产出的特征和标签，MultiIndex=(date, symbol)
        test_size: 测试集占比（按时间，取最后这部分）
        embargo_days: 训练集末尾和测试集开头之间隔离的天数
                      应该 >= 标签的 horizon，防止标签重叠泄露

    返回:
        X_train, y_train, X_test, y_test
    """
    # 取出所有唯一日期并排序
    dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)

    split_idx = int(n_dates * (1 - test_size))
    train_end_date = dates[split_idx]
    # 隔离带：测试集从 train_end + embargo 之后开始
    test_start_idx = min(split_idx + embargo_days, n_dates - 1)
    test_start_date = dates[test_start_idx]

    train_mask = X.index.get_level_values("date") <= train_end_date
    test_mask = X.index.get_level_values("date") >= test_start_date

    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
    )


def purged_kfold(
    X: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 5,
):
    """
    时序友好的 K 折交叉验证（Purged Walk-Forward CV）。

    与普通 KFold 的区别：
        - 每一折的训练集都在验证集之前（walk-forward）
        - 训练集和验证集之间有 embargo 隔离带

    用法（生成器，产出每折的训练/验证日期索引位置）:
        for train_idx, val_idx in purged_kfold(X, n_splits=5):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    这是 Marcos López de Prado《Advances in Financial ML》里推荐的方法的简化版。
    """
    dates = X.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_splits + 1)

    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        val_start = min(train_end + embargo_days, n_dates - 1)
        val_end = min(val_start + fold_size, n_dates)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[val_start:val_end]

        train_idx = np.where(dates.isin(train_dates))[0]
        val_idx = np.where(dates.isin(val_dates))[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx
