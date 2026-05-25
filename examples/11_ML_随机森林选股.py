"""
示例 11：随机森林 vs 线性模型（非线性 ML）
=============================================

对比线性模型和树模型，理解：
    - 树模型能捕捉非线性关系和特征交互
    - 但更容易过拟合，需要更小心地调参和验证
    - 量化里"更复杂的模型不一定更好"，要看样本外 IC

跑：python examples/11_ML_随机森林选股.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.ml import make_dataset, time_series_split, ModelWrapper
from src.ml import information_coefficient, quantile_backtest


def evaluate(model_name, X_tr, y_tr, X_te, y_te, **kwargs):
    model = ModelWrapper(model_name, task="regression", **kwargs)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    ic = information_coefficient(pred, y_te)
    qb = quantile_backtest(pred, y_te, n_quantiles=5)
    return model, ic, qb


def main():
    symbols = [f"STK{i:02d}" for i in range(30)]
    prices = generate_synthetic_prices(symbols, "2015-01-01", "2024-12-31", seed=11)

    X, y = make_dataset(prices, horizon=5, kind="return")
    X_tr, y_tr, X_te, y_te = time_series_split(X, y, test_size=0.3, embargo_days=5)
    print(f"训练 {X_tr.shape[0]} / 测试 {X_te.shape[0]} 样本\n")

    results = {}
    for name in ["ridge", "rf", "gbdt"]:
        print(f"训练 {name} ...")
        model, ic, qb = evaluate(name, X_tr, y_tr, X_te, y_te)
        results[name] = {
            "IC均值": ic["IC均值"],
            "ICIR": ic["ICIR"],
            "多空价差日均": qb.attrs["多空价差_日均"],
        }

    print("\n===== 模型对比（测试集）=====")
    comp = pd.DataFrame(results).T
    print(comp.to_string())

    # 看随机森林的特征重要性
    rf, _, _ = evaluate("rf", X_tr, y_tr, X_te, y_te)
    print("\n===== 随机森林特征重要性 =====")
    print(rf.feature_importance().to_string())

    print("\n提示：模拟数据下各模型 IC 接近，因为信号是线性的（GBM drift）。")
    print("真实市场里树模型通常因能捕捉非线性而略胜，但也更容易过拟合。")


if __name__ == "__main__":
    main()
