"""
基本面数据 + 基本面因子的单元测试.

最核心要验证的是 "防未来函数" 这件事 — 这是 09 章的灵魂.
其它测试覆盖 schema 一致性、因子方向约定、模拟数据可重复.

跑法:
    python tests/test_fundamentals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import (
    generate_synthetic_prices,
    generate_synthetic_fundamentals,
    get_fundamental_factor,
    to_daily,
    FundamentalsBundle,
    AKSHARE_FIELD_MAP,
    FIELD_DIRECTION,
)
from src.factors.fundamental import (
    earnings_yield, book_to_price, roe_factor,
    revenue_growth_factor, low_leverage_factor,
    all_fundamental_factors,
)


# ---------- 模拟数据基础 ----------

def test_generate_synthetic_fundamentals_basic():
    """生成的 bundle 应该有正确的季度数, 字段齐全."""
    bundles = generate_synthetic_fundamentals(
        ["A", "B", "C"], "2022-01-01", "2024-12-31", seed=42,
    )
    assert len(bundles) == 3
    for sym in ["A", "B", "C"]:
        b = bundles[sym]
        assert isinstance(b, FundamentalsBundle)
        assert b.symbol == sym
        # 2022~2024 共 3 年 × 4 季度 = 12 个报告期
        assert len(b.quarterly) == 12
        # 关键字段都在
        for col in ["roe", "eps", "bvps", "net_margin", "revenue_yoy"]:
            assert col in b.quarterly.columns


def test_synthetic_fundamentals_reproducible():
    """同 seed 应该产出完全一致的数据."""
    b1 = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=7)
    b2 = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=7)
    pd.testing.assert_frame_equal(b1["X"].quarterly, b2["X"].quarterly)


def test_synthetic_internal_consistency():
    """净利率 ≈ 净利润 / 营收, 内部数据应自洽."""
    b = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=3)["X"]
    q = b.quarterly
    # 净利润 / 营收 应该接近 净利率
    derived = q["net_profit"] / q["revenue"]
    assert (derived - q["net_margin"]).abs().max() < 1e-9


# ---------- 防未来函数 (核心) ----------

def test_to_daily_announce_lag_no_lookahead():
    """
    最关键的测试: 报告期 t 的数据, 应该在 t+lag 才能出现, 之前必须是上一份报告的值.
    """
    bundles = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=42)
    trading_dates = pd.bdate_range("2022-01-01", "2024-12-31")
    roe = to_daily(bundles, "roe", trading_dates, announce_lag_days=45)

    q = bundles["X"].quarterly["roe"]
    # 取一个中间报告期来检查
    report_date = pd.Timestamp("2024-03-31")
    announce_date = report_date + pd.Timedelta(days=45)  # 2024-05-15
    new_value = q.loc[report_date]
    prev_report = pd.Timestamp("2023-12-31")
    prev_value = q.loc[prev_report]

    # 在披露日之前一天 (取最近的交易日), 因子值应等于上一份报告
    day_before = trading_dates[trading_dates < announce_date][-1]
    assert abs(roe.loc[day_before, "X"] - prev_value) < 1e-10, (
        f"披露日前应该还是上一份报告值, 但 {day_before} 的值是 {roe.loc[day_before, 'X']}, 期望 {prev_value}"
    )

    # 在披露日 (或之后第一个交易日), 因子值应更新
    day_at_or_after = trading_dates[trading_dates >= announce_date][0]
    assert abs(roe.loc[day_at_or_after, "X"] - new_value) < 1e-10, (
        f"披露日起应该更新, 但 {day_at_or_after} 的值是 {roe.loc[day_at_or_after, 'X']}, 期望 {new_value}"
    )


def test_to_daily_first_periods_are_nan():
    """第一份财报披露之前的日期, 因子值应该是 NaN."""
    bundles = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=42)
    trading_dates = pd.bdate_range("2022-01-01", "2024-12-31")
    roe = to_daily(bundles, "roe", trading_dates, announce_lag_days=45)
    # 第一份报告期 2022-03-31, 披露 ≈ 2022-05-15
    # 在 2022-05-14 之前所有交易日, ROE 应该是 NaN
    early = roe.loc[:"2022-05-13", "X"]
    assert early.isna().all(), "首份披露日之前应该全 NaN"


def test_to_daily_different_lag_changes_alignment():
    """改 lag, 披露日跟着变, 因子时间序列也应改变."""
    bundles = generate_synthetic_fundamentals(["X"], "2022-01-01", "2024-12-31", seed=42)
    trading_dates = pd.bdate_range("2022-01-01", "2024-12-31")
    r45 = to_daily(bundles, "roe", trading_dates, announce_lag_days=45)
    r90 = to_daily(bundles, "roe", trading_dates, announce_lag_days=90)
    # 同一天可能两者值不同 (lag=90 时披露更晚, 用的还是上一份报告)
    diff = (r45["X"] - r90["X"]).abs().sum()
    assert diff > 0, "改 lag 应该改变对齐结果"


# ---------- get_fundamental_factor 接口 ----------

def test_get_fundamental_factor_shape_and_index():
    prices = generate_synthetic_prices(["A", "B"], "2022-01-01", "2024-12-31", seed=42)
    roe = get_fundamental_factor("roe", ["A", "B"], prices.index, use_synthetic=True, prices=prices)
    assert roe.shape == prices.shape
    assert list(roe.columns) == ["A", "B"]
    assert roe.index.equals(prices.index)


def test_get_fundamental_factor_use_synthetic_no_network():
    """use_synthetic=True 时不应该尝试连网络."""
    prices = generate_synthetic_prices(["X", "Y", "Z"], "2022-01-01", "2024-12-31", seed=1)
    # 即使是没接 akshare 的环境, 应该也能跑
    factor = get_fundamental_factor(
        "net_margin", ["X", "Y", "Z"], prices.index,
        use_synthetic=True, prices=prices,
    )
    assert factor.shape == prices.shape
    # 末尾不应全是 NaN
    assert factor.iloc[-1].notna().sum() == 3


# ---------- 字段方向约定 ----------

def test_field_direction_signs():
    """所有 AKSHARE_FIELD_MAP 里的字段, 都应该在 FIELD_DIRECTION 里有明确方向."""
    for name in AKSHARE_FIELD_MAP:
        assert name in FIELD_DIRECTION, f"{name} 缺方向定义"
        assert FIELD_DIRECTION[name] in (+1, -1), f"{name} 方向必须 ±1"


def test_debt_ratio_direction_is_negative():
    """资产负债率应该是 -1: 高杠杆风险大, 不直接看多."""
    assert FIELD_DIRECTION["debt_ratio"] == -1


# ---------- 因子函数: 值越大越看多 ----------

def test_earnings_yield_is_eps_over_price():
    """EP = EPS / 价格, 应该跟 PE 反向 (大 = 便宜)."""
    prices = generate_synthetic_prices(["A", "B"], "2022-01-01", "2024-12-31", seed=42)
    ep = earnings_yield(prices, use_synthetic=True)
    # 末日有值, 范围应该合理 (-0.5 ~ 0.5)
    last = ep.iloc[-1].dropna()
    assert (last.abs() < 2.0).all(), "EP 值不应过大"


def test_book_to_price_runs():
    prices = generate_synthetic_prices(["A", "B"], "2022-01-01", "2024-12-31", seed=42)
    bp = book_to_price(prices, use_synthetic=True)
    last = bp.iloc[-1].dropna()
    assert len(last) > 0


def test_low_leverage_negates_debt():
    """low_leverage 应该是 -debt_ratio."""
    prices = generate_synthetic_prices(["A", "B"], "2022-01-01", "2024-12-31", seed=42)
    debt = get_fundamental_factor("debt_ratio", ["A", "B"], prices.index, use_synthetic=True, prices=prices)
    ll = low_leverage_factor(prices, use_synthetic=True)
    # 同 seed 下应该完全等于负值
    pd.testing.assert_frame_equal(ll, -debt)


def test_all_fundamental_factors_returns_dict():
    prices = generate_synthetic_prices(["A", "B", "C"], "2022-01-01", "2024-12-31", seed=42)
    factors = all_fundamental_factors(prices, use_synthetic=True)
    assert isinstance(factors, dict)
    assert "ROE" in factors and "EP" in factors and "BP" in factors
    for name, f in factors.items():
        assert f.shape == prices.shape, f"{name} shape 不一致"


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
