"""
ML 模块的单元测试，重点测"防数据泄露"是否正确。
跑：python tests/test_ml.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import generate_synthetic_prices
from src.ml import make_dataset, time_series_split, ModelWrapper
from src.ml import information_coefficient, quantile_backtest


def _make_data():
    symbols = [f"STK{i:02d}" for i in range(15)]
    prices = generate_synthetic_prices(symbols, "2018-01-01", "2023-12-31", seed=1)
    return prices


def test_make_dataset_shape():
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    # X 和 y 行数一致，且无 NaN
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert X.shape[1] == 10  # 10 个特征


def test_time_split_no_leakage():
    # 核心测试：训练集的所有日期必须严格早于测试集
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    X_tr, y_tr, X_te, y_te = time_series_split(X, y, test_size=0.3, embargo_days=5)

    train_dates = X_tr.index.get_level_values("date")
    test_dates = X_te.index.get_level_values("date")
    # 训练集最晚的一天 < 测试集最早的一天（含 embargo 间隔）
    assert train_dates.max() < test_dates.min()


def test_label_is_future():
    # 标签必须是未来收益：手工核对一个值
    prices = _make_data()
    from src.ml.features import build_label
    label = build_label(prices, horizon=5, kind="return")
    # 取某个 (date, symbol)，验证 = P(t+5)/P(t) - 1
    sym = prices.columns[0]
    t = prices.index[10]
    t5 = prices.index[15]
    expected = prices.loc[t5, sym] / prices.loc[t, sym] - 1
    actual = label.loc[(t, sym)]
    assert abs(expected - actual) < 1e-9


def test_model_fit_predict():
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    X_tr, y_tr, X_te, y_te = time_series_split(X, y)
    model = ModelWrapper("ridge").fit(X_tr, y_tr)
    pred = model.predict(X_te)
    # 预测长度对、索引对、无 NaN
    assert len(pred) == len(X_te)
    assert pred.index.equals(X_te.index)
    assert not pred.isna().any()


def test_feature_importance_not_none():
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    X_tr, y_tr, _, _ = time_series_split(X, y)
    for name in ["ridge", "rf", "gbdt"]:
        model = ModelWrapper(name).fit(X_tr, y_tr)
        imp = model.feature_importance()
        assert imp is not None
        assert len(imp) == X.shape[1]


def test_ic_perfect_prediction():
    # 如果预测=真实，IC 应该接近 1
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    ic = information_coefficient(y, y)  # 预测就是真实
    assert ic["IC均值"] > 0.99


def test_quantile_monotonic_with_perfect_pred():
    # 完美预测下，分组应严格单调（Q5 > Q1）
    prices = _make_data()
    X, y = make_dataset(prices, horizon=5, kind="return")
    qb = quantile_backtest(y, y, n_quantiles=5)
    assert qb["平均收益"].iloc[-1] > qb["平均收益"].iloc[0]


def run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"  PASS  {f.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {f.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(funcs)} 测试通过")
    return passed == len(funcs)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
