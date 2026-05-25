"""
示例 10：用线性模型预测未来收益（机器学习入门第一步）
=========================================================

最简单的 ML 量化流程：
    1. 把价格变成特征矩阵 X（动量、波动率、RSI...）
    2. 标签 y = 未来 5 天收益
    3. 按时间切训练/测试集（防数据泄露）
    4. 训练岭回归
    5. 评估：IC + 分组回测（量化的方式，不是看准确率）

跑：python examples/10_ML_线性模型预测.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.ml import make_dataset, time_series_split, ModelWrapper
from src.ml import information_coefficient, quantile_backtest


def main():
    # 1. 数据（模拟数据，无需联网）
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2015-01-01", "2024-12-31", seed=7)
    print(f"价格数据：{prices.shape[0]} 天 × {prices.shape[1]} 只票\n")

    # 2. 构造 (X, y)
    X, y = make_dataset(prices, horizon=5, kind="return")
    print(f"特征矩阵 X：{X.shape}，标签 y：{y.shape}")
    print(f"特征列：{list(X.columns)}\n")

    # 3. 按时间切（训练在前，测试在后，留 5 天隔离带）
    X_tr, y_tr, X_te, y_te = time_series_split(X, y, test_size=0.3, embargo_days=5)
    print(f"训练集 {X_tr.shape[0]} 样本，测试集 {X_te.shape[0]} 样本\n")

    # 4. 训练岭回归
    model = ModelWrapper("ridge", task="regression", alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    # 5. 评估（量化方式）
    print("===== IC 评估（测试集）=====")
    ic = information_coefficient(pred, y_te)
    for k, v in ic.items():
        print(f"  {k}: {v:.4f}")

    print("\n===== 分组回测（5 组）=====")
    qb = quantile_backtest(pred, y_te, n_quantiles=5)
    print(qb.to_string())
    print(f"\n多空价差（日均）: {qb.attrs['多空价差_日均']:.5f}")

    print("\n===== 特征重要性（线性系数绝对值）=====")
    print(model.feature_importance().to_string())


if __name__ == "__main__":
    main()
