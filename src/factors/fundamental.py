"""
基本面因子库
============

把基本面字段加工成可用因子. 跟技术因子的接口完全一致:
    输入: 价格宽表 + 基本面 bundle 或 get_fundamental_factor 返回的日频字段
    输出: DataFrame, index=date, columns=symbol, 值越大越看多

关键约定:
    所有函数返回的因子都满足"值越大越看多". 估值因子 (PE/PB) 天然是"低估好",
    所以这里返回的是 PE 的倒数 (EP), 这样大值 = 便宜 = 看多, 跟其他因子统一方向.

防未来函数:
    因子计算只能用 "今天之前已披露" 的财务数据.
    本模块依赖 src.data.fundamentals.get_fundamental_factor, 它内部已做披露日对齐.
    上层调用者不需要再担心.

价值因子 vs 质量因子 (经典区分):
    价值: 当前估值便宜不便宜 (PE/PB/PS), 答 "现在值不值"
    质量: 公司本身好不好 (ROE/ROA/毛利率), 答 "未来能不能持续"
    成长: 业绩涨不涨 (营收/净利 同比), 答 "未来增速预期"
    安全: 财务稳不稳 (负债率/流动比), 答 "会不会暴雷"

经典因子论文:
    Fama-French HML (价值): 高 BM 跑赢低 BM
    Novy-Marx GP (质量): 高毛利率跑赢低毛利率
    Asness QMJ (质量): 综合质量因子
"""

from __future__ import annotations

import pandas as pd

from ..data.fundamentals import get_fundamental_factor


# ---------- 价值因子 ----------

def earnings_yield(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    EP (Earnings Yield, 盈利收益率) = EPS / 股价 = 1 / PE_ttm.

    为什么不直接用 PE?
        我们约定"值越大越看多". PE 越低越看多, 直接用 PE 方向反了.
        EP = 1/PE, EP 大 = PE 小 = 便宜 = 看多, 方向统一.

    传统意义上这就是"价值因子 (Value)" 的代表.
    """
    syms = symbols or list(prices.columns)
    eps = get_fundamental_factor("eps", syms, prices.index, use_synthetic=use_synthetic, prices=prices)
    return (eps / prices).replace([float("inf"), -float("inf")], pd.NA)


def book_to_price(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    BP (Book-to-Price) = 每股净资产 / 股价 = 1 / PB.

    Fama-French 三因子里的 HML 就是按 BP 排序构造的, 学界最经典的"价值"因子.
    """
    syms = symbols or list(prices.columns)
    bvps = get_fundamental_factor("bvps", syms, prices.index, use_synthetic=use_synthetic, prices=prices)
    return (bvps / prices).replace([float("inf"), -float("inf")], pd.NA)


# ---------- 质量因子 ----------

def roe_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    ROE (净资产收益率) 因子. 高 ROE = 高质量.
    本框架的 ROE 是季频"年化值" (来自 akshare).
    """
    syms = symbols or list(prices.columns)
    return get_fundamental_factor("roe", syms, prices.index, use_synthetic=use_synthetic, prices=prices)


def gross_margin_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """毛利率因子. Novy-Marx (2013) 论文里"质量因子"的代表."""
    syms = symbols or list(prices.columns)
    return get_fundamental_factor("gross_margin", syms, prices.index, use_synthetic=use_synthetic, prices=prices)


def net_margin_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """销售净利率因子. 跟 ROE 类似但角度不同 (营收效率 vs 资本效率)."""
    syms = symbols or list(prices.columns)
    return get_fundamental_factor("net_margin", syms, prices.index, use_synthetic=use_synthetic, prices=prices)


# ---------- 成长因子 ----------

def revenue_growth_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """营收同比增速因子. 增长越快越看多 (在合理范围内)."""
    syms = symbols or list(prices.columns)
    return get_fundamental_factor("revenue_yoy", syms, prices.index, use_synthetic=use_synthetic, prices=prices)


def profit_growth_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """归母净利同比增速因子."""
    syms = symbols or list(prices.columns)
    return get_fundamental_factor("profit_yoy", syms, prices.index, use_synthetic=use_synthetic, prices=prices)


# ---------- 安全因子 ----------

def low_leverage_factor(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """
    低杠杆 = 高安全. 取负的资产负债率, 让"值越大越看多" 约定成立.
    """
    syms = symbols or list(prices.columns)
    debt = get_fundamental_factor("debt_ratio", syms, prices.index, use_synthetic=use_synthetic, prices=prices)
    return -debt


# ---------- 一键拉全套基本面因子 ----------

def all_fundamental_factors(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    use_synthetic: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    一次性把常用基本面因子全算出来, 返回 dict, 方便丢给 MultiFactorStrategy.

    返回的因子全都是"值越大越看多", 可以直接合成.
    """
    return {
        "EP":           earnings_yield(prices, symbols, use_synthetic),
        "BP":           book_to_price(prices, symbols, use_synthetic),
        "ROE":          roe_factor(prices, symbols, use_synthetic),
        "gross_margin": gross_margin_factor(prices, symbols, use_synthetic),
        "net_margin":   net_margin_factor(prices, symbols, use_synthetic),
        "rev_growth":   revenue_growth_factor(prices, symbols, use_synthetic),
        "profit_growth": profit_growth_factor(prices, symbols, use_synthetic),
        "low_leverage": low_leverage_factor(prices, symbols, use_synthetic),
    }
