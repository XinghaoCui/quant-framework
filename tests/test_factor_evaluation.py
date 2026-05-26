"""
因子评估模块的单元测试.

为什么需要这些测试?
    IC / 分组回测的数学不复杂, 但容易出错的地方非常隐蔽:
    - 截面 vs 时序 axis 搞反
    - 未来收益 shift 方向反了
    - 分组标签 0~n-1 vs 1~n
    所以用"构造一个已知答案的因子"来验证, 比凭直觉读代码更可靠.

跑法:
    python tests/test_factor_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.factors.evaluation import (
    forward_returns,
    compute_ic,
    ic_summary,
    quantile_labels,
    quantile_returns,
    quantile_backtest,
    ic_decay,
    factor_report,
)
from src.strategies.multi_factor import cross_section_zscore, MultiFactorStrategy


# ---------- 构造确定性数据 ----------

def _toy_prices(n_days: int = 200, n_stocks: int = 10, seed: int = 0) -> pd.DataFrame:
    """构造可复现的随机游走价格表."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, size=(n_days, n_stocks))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=n_days),
        columns=[f"S{i}" for i in range(n_stocks)],
    )
    prices.index.name = "date"
    return prices


# ---------- forward_returns ----------

def test_forward_returns_basic():
    """horizon=1 时 forward_returns[t] = prices[t+1]/prices[t] - 1."""
    prices = _toy_prices(50)
    fwd = forward_returns(prices, 1)
    # 手算第一行
    expected = prices.iloc[1] / prices.iloc[0] - 1
    np.testing.assert_allclose(fwd.iloc[0].values, expected.values, rtol=1e-12)


def test_forward_returns_last_row_nan():
    """最后 horizon 行应该是 NaN (没有未来数据)."""
    prices = _toy_prices(50)
    fwd = forward_returns(prices, 5)
    assert fwd.iloc[-5:].isna().all().all()


# ---------- IC ----------

def test_ic_perfect_signal():
    """因子 = 真实未来收益 → IC 应该接近 1."""
    prices = _toy_prices(100)
    fwd = forward_returns(prices, 1)
    # 直接用未来收益作为因子 (作弊用作单元测试)
    ic = compute_ic(fwd, fwd, method="spearman")
    assert ic.dropna().mean() > 0.99


def test_ic_random_signal_near_zero():
    """随机因子 IC 应该接近 0."""
    prices = _toy_prices(500, n_stocks=20, seed=1)
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(
        rng.normal(0, 1, size=prices.shape),
        index=prices.index, columns=prices.columns,
    )
    fwd = forward_returns(prices, 5)
    ic = compute_ic(noise, fwd)
    # 长样本随机, |均值| 应该 < 0.05
    assert abs(ic.mean()) < 0.05


def test_ic_summary_keys():
    """ic_summary 必有的几个键."""
    s = pd.Series([0.1, -0.05, 0.08, 0.12, -0.02])
    out = ic_summary(s)
    for k in ["IC均值", "IC标准差", "IR", "t统计", "IC>0占比", "样本天数"]:
        assert k in out


def test_ic_summary_ir_formula():
    """IR = 均值 / 标准差."""
    s = pd.Series([0.1, 0.2, -0.05, 0.15, -0.1])
    out = ic_summary(s)
    expected_ir = float(s.mean() / s.std())
    assert abs(out["IR"] - expected_ir) < 1e-12


# ---------- 分组 ----------

def test_quantile_labels_range():
    """标签应该在 0..n_quantiles-1."""
    prices = _toy_prices(100)
    labels = quantile_labels(prices, n_quantiles=5)
    valid = labels.stack(future_stack=True).dropna()
    assert valid.min() == 0
    assert valid.max() == 4


def test_quantile_returns_monotone_for_perfect_factor():
    """因子=未来收益时 Q1<Q2<...<Q5, 严格单调."""
    prices = _toy_prices(200, n_stocks=30, seed=7)
    fwd = forward_returns(prices, 5)
    qrets = quantile_returns(fwd, fwd, n_quantiles=5)
    avg = qrets.mean()
    # 完美信号下应该严格单调
    assert all(avg.iloc[i] < avg.iloc[i + 1] for i in range(len(avg) - 1))


def test_quantile_backtest_long_short_positive_for_perfect_factor():
    """完美因子下多空价差应该明显为正."""
    prices = _toy_prices(200, n_stocks=30, seed=11)
    fwd = forward_returns(prices, 5)
    result = quantile_backtest(fwd, prices, n_quantiles=5, horizon=5)
    assert result.summary.loc["多空(Qn-Q1)", "年化收益"] > 0


# ---------- IC 衰减 ----------

def test_ic_decay_horizons_match():
    """ic_decay 的 index 应该等于传入的 horizons."""
    prices = _toy_prices(300, n_stocks=20)
    rng = np.random.default_rng(3)
    factor = pd.DataFrame(
        rng.normal(0, 1, size=prices.shape),
        index=prices.index, columns=prices.columns,
    )
    horizons = [1, 5, 20]
    out = ic_decay(factor, prices, horizons)
    assert list(out.index) == horizons


# ---------- factor_report ----------

def test_factor_report_runs_end_to_end():
    """综合报告完整流程能跑, 各字段都非空."""
    prices = _toy_prices(300, n_stocks=20)
    rng = np.random.default_rng(5)
    factor = pd.DataFrame(
        rng.normal(0, 1, size=prices.shape),
        index=prices.index, columns=prices.columns,
    )
    report = factor_report(factor, prices, name="noise", horizon=5)
    assert report.name == "noise"
    assert len(report.ic_series) > 0
    assert len(report.decay) > 0
    assert len(report.quantile.summary) > 0
    # to_text 不应抛错
    text = report.to_text()
    assert "noise" in text


# ---------- 多因子合成 ----------

def test_cross_section_zscore_mean_zero():
    """每天的截面 zscore 均值应该是 0 (有效股票数 >= 2)."""
    prices = _toy_prices(100, n_stocks=10)
    z = cross_section_zscore(prices)
    daily_mean = z.mean(axis=1)
    assert daily_mean.abs().max() < 1e-10


def test_multi_factor_equal_runs():
    """MultiFactorStrategy(equal) 能产出合法权重."""
    prices = _toy_prices(300, n_stocks=20)
    rng = np.random.default_rng(7)
    factors = {
        "f1": pd.DataFrame(rng.normal(0, 1, size=prices.shape),
                           index=prices.index, columns=prices.columns),
        "f2": pd.DataFrame(rng.normal(0, 1, size=prices.shape),
                           index=prices.index, columns=prices.columns),
    }
    strat = MultiFactorStrategy(factors, combine="equal", top_pct=0.3, rebalance_days=5)
    w = strat.generate_weights(prices)
    assert w.shape == prices.shape
    # 权重和不超过 1 (允许浮点误差)
    assert w.sum(axis=1).max() <= 1.0 + 1e-9


def test_multi_factor_ic_weighted_logs_weights():
    """ic_weighted 模式应该记录 weight_log_."""
    prices = _toy_prices(400, n_stocks=20)
    rng = np.random.default_rng(13)
    factors = {
        "f1": pd.DataFrame(rng.normal(0, 1, size=prices.shape),
                           index=prices.index, columns=prices.columns),
        "f2": pd.DataFrame(rng.normal(0, 1, size=prices.shape),
                           index=prices.index, columns=prices.columns),
    }
    strat = MultiFactorStrategy(
        factors, combine="ic_weighted", ic_lookback=60, ic_horizon=5, top_pct=0.3,
    )
    strat.generate_weights(prices)
    assert strat.weight_log_ is not None
    assert set(strat.weight_log_.columns) == {"f1", "f2"}


def test_multi_factor_invalid_combine_raises():
    prices = _toy_prices(50)
    try:
        MultiFactorStrategy({"f": prices}, combine="bogus")
    except ValueError:
        return
    raise AssertionError("combine='bogus' 应该抛 ValueError")


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
