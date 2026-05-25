"""
指标计算的单元测试。
用 pytest 跑：pytest tests/ -v
或直接 python tests/test_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.engine.metrics import (
    annual_return, max_drawdown, sharpe_ratio, win_rate, cum_returns,
)


def test_cum_returns_simple():
    # 第一天 +10%，第二天 -10%，累计应为 -1%
    r = pd.Series([0.1, -0.1])
    cum = cum_returns(r)
    assert abs(cum.iloc[-1] - (-0.01)) < 1e-9


def test_max_drawdown_known():
    # 构造：涨到 110，跌到 88（相对峰值 110 跌 20%）
    # 收益序列：+10%, -20%
    r = pd.Series([0.10, -0.20])
    mdd = max_drawdown(r)
    # net: 1.1, 0.88; peak: 1.1, 1.1; dd: 0, -0.2
    assert abs(mdd - (-0.20)) < 1e-9


def test_max_drawdown_no_loss():
    # 一直涨，回撤应为 0
    r = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert max_drawdown(r) == 0.0 or max_drawdown(r) > -1e-9


def test_sharpe_zero_vol():
    # 波动率为 0 时夏普应返回 0（避免除零）
    r = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive():
    # 稳定正收益应有正夏普
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    assert sharpe_ratio(r) > 0


def test_win_rate():
    r = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1])  # 3 胜 2 负
    assert abs(win_rate(r) - 0.6) < 1e-9


def test_annual_return_sign():
    # 整体亏损应得负年化
    r = pd.Series([-0.01] * 252)
    assert annual_return(r) < 0


def run_all():
    funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for f in funcs:
        try:
            f()
            print(f"  PASS  {f.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {f.__name__}: {e}")
    print(f"\n{passed}/{len(funcs)} 测试通过")
    return passed == len(funcs)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
