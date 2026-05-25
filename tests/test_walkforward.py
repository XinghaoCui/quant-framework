"""
walk-forward 滚动重训的单元测试。
重点验证：无数据泄露、预测无重复、两种窗口都能跑、接入策略正常。
跑：python tests/test_walkforward.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import generate_synthetic_prices
from src.ml import walk_forward_predict
from src.ml.ml_strategy import MLStrategy


def _make_data():
    symbols = [f"STK{i:02d}" for i in range(15)]
    # 4 年数据，够多次重训
    return generate_synthetic_prices(symbols, "2018-01-01", "2021-12-31", seed=3)


def test_expanding_runs():
    prices = _make_data()
    pred, log = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=252, retrain_freq=63, window="expanding",
    )
    assert len(pred) > 0
    assert len(log) > 0
    # 重训应该发生多次
    assert len(log) >= 2


def test_rolling_runs():
    prices = _make_data()
    pred, log = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=252, retrain_freq=63,
        window="rolling", rolling_window_days=252,
    )
    assert len(pred) > 0
    assert len(log) >= 2


def test_no_duplicate_predictions():
    # 预测段无缝拼接，不应有重复的 (date, symbol)
    prices = _make_data()
    pred, _ = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=252, retrain_freq=63,
    )
    assert pred.index.is_unique


def test_predictions_after_initial_train():
    # 核心防泄露：所有预测日期必须在初始训练窗口之后
    prices = _make_data()
    X_dates = pd.to_datetime(prices.index).sort_values()
    initial_train = 252
    embargo = 5
    pred, _ = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=initial_train, retrain_freq=63, embargo_days=embargo,
    )
    pred_min_date = pred.index.get_level_values("date").min()
    # 预测最早日期应 >= 初始训练止 + embargo 对应的日期
    # 用数据集自身的交易日（含因子 NaN 裁剪后），保守地检查不早于初始训练期
    assert pred_min_date >= X_dates[initial_train]


def test_log_train_before_predict():
    # 每一轮：训练止 < 预测起（含隔离带）
    prices = _make_data()
    _, log = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=252, retrain_freq=63,
    )
    for _, row in log.iterrows():
        assert row["训练止"] < row["预测起"]


def test_expanding_window_grows():
    # expanding 模式：训练样本数应随轮次单调不减
    prices = _make_data()
    _, log = walk_forward_predict(
        prices, model_name="ridge", horizon=5,
        initial_train_days=252, retrain_freq=63, window="expanding",
    )
    sizes = log["训练样本"].tolist()
    assert all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1))


def test_mlstrategy_retrain_mode():
    # MLStrategy 滚动重训模式能产出权重，且 walk_log_ 被填充
    prices = _make_data()
    strat = MLStrategy(
        model_name="ridge", horizon=5, top_pct=0.3,
        retrain_freq=63, window="expanding", initial_train_days=252,
    )
    weights = strat.generate_weights(prices)
    assert weights.shape == prices.shape
    assert strat.walk_log_ is not None
    assert len(strat.walk_log_) >= 2


def test_mlstrategy_train_once_still_works():
    # 回归测试：retrain_freq=None 时仍是训练一次，self.model 有效
    prices = _make_data()
    strat = MLStrategy(model_name="ridge", horizon=5, retrain_freq=None, train_ratio=0.5)
    weights = strat.generate_weights(prices)
    assert weights.shape == prices.shape
    assert strat.model is not None


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
