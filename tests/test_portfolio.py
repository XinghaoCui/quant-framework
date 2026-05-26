"""
组合优化模块单元测试.

为什么需要这些测试?
    组合优化的代码很容易"看着对、其实错": 权重和不为 1、最小方差权重比等权方差还大、
    风险平价没真平价... 用"构造已知答案的极端情形"来验证, 比凭直觉读公式更可靠.

跑法:
    python tests/test_portfolio.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.portfolio.optimizer import (
    equal_weights,
    inverse_volatility_weights,
    risk_parity_weights,
    min_variance_weights,
    mean_variance_weights,
    max_sharpe_weights,
    efficient_frontier,
)
from src.strategies.optimized import OptimizedStrategy


# ---------- 工具 ----------

def _toy_returns(n_days: int = 252, n_stocks: int = 5, seed: int = 0) -> pd.DataFrame:
    """构造可复现的随机日收益."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(n_days, n_stocks)),
        index=pd.bdate_range("2022-01-01", periods=n_days),
        columns=[f"S{i}" for i in range(n_stocks)],
    )


# ---------- 基础不变量 ----------

def test_equal_weights_sums_to_one():
    r = _toy_returns()
    w = equal_weights(r).weights
    assert abs(w.sum() - 1.0) < 1e-10
    # 所有权重一致
    assert (w.std() < 1e-10)


def test_inverse_vol_sums_to_one_and_inverse():
    r = _toy_returns()
    w = inverse_volatility_weights(r).weights
    assert abs(w.sum() - 1.0) < 1e-10
    # 波动率高的票应该权重低 → 验证排序
    vols = r.std()
    correlation = w.corr(1 / vols)
    assert correlation > 0.99  # 几乎完美相关


def test_risk_parity_sums_to_one_and_no_short():
    r = _toy_returns()
    w = risk_parity_weights(r).weights
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()


def test_risk_parity_actually_parity():
    """每只票的边际风险贡献应该大致相等."""
    r = _toy_returns(n_days=500, n_stocks=4, seed=3)
    w = risk_parity_weights(r).weights.values
    cov = r.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    rc = w * (cov @ w) / port_vol
    # 相对差异 < 1%
    assert (rc.max() - rc.min()) / rc.mean() < 0.01


def test_min_variance_sums_to_one_and_beats_equal():
    """最小方差组合的方差不应大于等权."""
    r = _toy_returns(n_days=500, n_stocks=8, seed=7)
    cov = r.cov().values
    n = cov.shape[0]
    eq = np.full(n, 1.0 / n)
    eq_var = eq @ cov @ eq

    res = min_variance_weights(r)
    w = res.weights.values
    assert abs(w.sum() - 1.0) < 1e-6
    mv_var = w @ cov @ w
    assert mv_var <= eq_var + 1e-10  # 浮点容差


def test_min_variance_allow_short_can_be_smaller():
    """允许做空时, 方差应该 <= 不允许做空."""
    r = _toy_returns(n_days=500, n_stocks=6, seed=9)
    cov = r.cov().values
    w_long = min_variance_weights(r).weights.values
    w_any = min_variance_weights(r, allow_short=True).weights.values
    assert (w_any @ cov @ w_any) <= (w_long @ cov @ w_long) + 1e-8


# ---------- 均值-方差 ----------

def test_mean_variance_runs_and_normalizes():
    r = _toy_returns()
    mu = r.mean()
    res = mean_variance_weights(mu, r, risk_aversion=1.0)
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert res.expected_return is not None


def test_mean_variance_high_aversion_approaches_min_var():
    """极高风险厌恶下, 均值方差应接近最小方差."""
    r = _toy_returns(n_days=500, n_stocks=6, seed=11)
    mu = r.mean()
    cov = r.cov().values
    w_mv = mean_variance_weights(mu, r, risk_aversion=1e8).weights.values
    w_min = min_variance_weights(r).weights.values
    # 两者方差应非常接近
    var_mv = w_mv @ cov @ w_mv
    var_min = w_min @ cov @ w_min
    assert abs(var_mv - var_min) / var_min < 0.05


# ---------- 最大夏普 ----------

def test_max_sharpe_runs():
    r = _toy_returns(n_days=500, n_stocks=6, seed=13)
    mu = r.mean()
    res = max_sharpe_weights(mu, r, risk_free=0.0)
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert res.expected_sharpe is not None


# ---------- 有效前沿 ----------

def test_efficient_frontier_returns_dataframe():
    r = _toy_returns(n_days=400, n_stocks=5, seed=17)
    mu = r.mean()
    front = efficient_frontier(mu, r, n_points=10)
    assert len(front) > 0
    assert {"期望收益", "期望波动率", "夏普"}.issubset(front.columns)


def test_efficient_frontier_monotone():
    """前沿上, 期望收益升高时, 波动率也应单调升高 (大致)."""
    r = _toy_returns(n_days=400, n_stocks=5, seed=19)
    mu = r.mean()
    front = efficient_frontier(mu, r, n_points=15).sort_values("期望收益").reset_index(drop=True)
    # 取前沿"右上半段" (期望收益 >= 最小方差点的期望收益), 此段单调
    if len(front) >= 3:
        idx_min = front["期望波动率"].idxmin()
        upper = front.iloc[idx_min:]
        vols = upper["期望波动率"].values
        assert all(vols[i] <= vols[i + 1] + 1e-6 for i in range(len(vols) - 1))


# ---------- OptimizedStrategy 接入 ----------

def test_optimized_strategy_min_variance_runs():
    """端到端: 给打分 + 价格, 跑出合法权重矩阵."""
    rng = np.random.default_rng(23)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=(400, 10)), axis=0)),
        index=pd.bdate_range("2022-01-01", periods=400),
        columns=[f"S{i}" for i in range(10)],
    )
    scores = pd.DataFrame(
        rng.normal(0, 1, size=prices.shape),
        index=prices.index, columns=prices.columns,
    )
    strat = OptimizedStrategy(
        scores=scores, optimizer="min_variance",
        top_pct=0.5, rebalance_days=20, lookback_days=60,
    )
    w = strat.generate_weights(prices)
    assert w.shape == prices.shape
    # 权重和 ≤ 1 (浮点容差)
    assert w.sum(axis=1).max() <= 1.0 + 1e-6
    # 没有负权重 (默认不做空)
    assert (w.values >= -1e-9).all()
    # 训练日志非空
    assert len(strat.opt_log_) > 0


def test_optimized_strategy_rejects_unknown_optimizer():
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(np.ones((50, 5)), columns=list("ABCDE"))
    scores = pd.DataFrame(rng.normal(0, 1, size=prices.shape), columns=prices.columns)
    try:
        OptimizedStrategy(scores=scores, optimizer="bogus")
    except ValueError:
        return
    raise AssertionError("optimizer='bogus' 应该抛 ValueError")


# ---------- runner ----------

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
        except Exception as e:
            print(f"  ERR   {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(funcs)} 测试通过")
    return passed == len(funcs)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
