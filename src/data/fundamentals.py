"""
基本面数据层
============

读价 (price) 与读财报 (fundamentals) 是两套独立的数据流, 工程上完全不同:

    | 维度       | 价格              | 财报               |
    | -------- | --------------- | ---------------- |
    | 频率       | 日频              | 季频 (上市公司一年报告 4 次) |
    | 时间戳      | 交易日             | "报告期" + "披露日"   |
    | 防未来函数风险 | 低 (今日收盘对应今日)    | 高 (披露有滞后, 报告期≠可用日) |
    | 接口稳定性    | 不错              | 一般 (字段名经常变)      |

本模块负责把财报数据按"披露日"对齐到日频, 让上层因子代码可以像用价格一样用财报.

防未来函数 (这是 09 章核心教学点):
    报告期 = 报告统计的截止日 (Q1: 03-31, Q2: 06-30, Q3: 09-30, 年报: 12-31)
    披露日 = 报告实际公布的日子 (通常比报告期晚 1~3 个月)
    在第 t 天用因子时, 只能看 "披露日 ≤ t" 的财报, 否则就是用未来数据.

    本框架用简化近似: 披露日 = 报告期 + announce_lag_days (默认 45 天).
    实战中更严谨的做法是去 stock_yjbb_em (业绩报告) 接口拿真实披露日.

数据源:
    1. akshare 的 `stock_financial_abstract` — 主要源, 含 80+ 财务指标
    2. 模拟兜底 `generate_synthetic_fundamentals` — 跟 generate_synthetic_prices 配对

详见 `思考与学习/09_基本面数据/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable, Literal

import numpy as np
import pandas as pd


# 我们关心的财务字段及在 akshare 摘要表里的中文指标名
# 设计: 用 dict 隔离"我们用什么名字" 和 "akshare 用什么名字", 接口名变了只改这里
AKSHARE_FIELD_MAP: dict[str, list[str]] = {
    # value (字段名 -> 在 akshare "指标" 列里的中文名候选, 取第一个能匹配上的)
    "eps":           ["基本每股收益", "每股收益(摊薄)", "每股收益"],
    "bvps":          ["每股净资产"],
    "net_profit":    ["归母净利润", "净利润"],
    "revenue":       ["营业总收入", "营业收入"],
    # quality
    "roe":           ["净资产收益率(ROE)", "ROE", "净资产收益率"],
    "roa":           ["总资产报酬率(ROA)", "ROA", "总资产报酬率"],
    "gross_margin":  ["毛利率"],
    "net_margin":    ["销售净利率", "净利率"],
    # growth (yoy = 同比增长)
    "revenue_yoy":   ["营业总收入同比增长", "营业收入同比增长"],
    "profit_yoy":    ["归母净利润同比增长", "净利润同比增长"],
    # safety
    "debt_ratio":    ["资产负债率"],
}

# 字段的"看多方向": +1 = 值越大越看多, -1 = 值越小越看多
FIELD_DIRECTION: dict[str, int] = {
    "eps": +1, "bvps": +1, "net_profit": +1, "revenue": +1,
    "roe": +1, "roa": +1, "gross_margin": +1, "net_margin": +1,
    "revenue_yoy": +1, "profit_yoy": +1,
    "debt_ratio": -1,  # 杠杆越高风险越大, 取负
    # 估值因子是衍生的 (price / fundamental), 在 factors/fundamental.py 算
}


# ---------- 数据容器 ----------

@dataclass
class FundamentalsBundle:
    """
    单只票的全部财务数据 (季频).

    quarterly: DataFrame
        index = report_date (季末日期, 如 2024-03-31)
        columns = AKSHARE_FIELD_MAP 里的字段
        values = 该报告期的财务数值
    """
    symbol: str
    quarterly: pd.DataFrame = dc_field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        if self.quarterly.empty:
            return f"<FundamentalsBundle {self.symbol} empty>"
        return (
            f"<FundamentalsBundle {self.symbol} "
            f"{len(self.quarterly)} 个报告期 "
            f"{self.quarterly.index.min().date()}~{self.quarterly.index.max().date()}>"
        )


# ---------- 模拟数据 (兜底, 保证示例 100% 可跑) ----------

def generate_synthetic_fundamentals(
    symbols: list[str],
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    prices: pd.DataFrame | None = None,
) -> dict[str, FundamentalsBundle]:
    """
    生成跟价格配套的模拟季频财报数据.

    模型思路:
        每只票有一个 "基础质量" 系数, 决定它的 ROE 大致水平 (3%~25%).
        每个季度的净利润 = 基础 × (1 + 噪声) × 时间增长趋势.
        营收、净利润、ROE、毛利率等都从这个基础推导, 保证内部一致 (净利率 = 净利润/营收).

    如果传入 prices, 模拟出来的"高质量股票" 会跟价格涨幅大致正相关 (制造 alpha),
    便于教学时能看到价值/质量因子有 IC.
    """
    rng = np.random.default_rng(seed)
    report_dates = _quarterly_dates(start, end)

    bundles = {}
    for i, sym in enumerate(symbols):
        # 基础质量 (决定 ROE 中枢): 0.03~0.25
        base_quality = 0.03 + 0.22 * rng.random()
        # 营收基数 (亿元量级)
        base_revenue = 10 * np.exp(rng.normal(0, 1))
        # 杠杆 / 资产负债率: 0.2~0.7
        base_debt = 0.2 + 0.5 * rng.random()
        # 价格关联系数: 让"高质量" 的票价格也涨得多 (制造 alpha)
        price_link = 0.0
        if prices is not None and sym in prices.columns:
            ret = prices[sym].pct_change().mean() * 252
            price_link = (ret - 0.10)  # 跟年化收益跟基线 10% 的差

        rows = []
        for q_idx, rdate in enumerate(report_dates):
            # 时间趋势: 营收每季度大约 +2% 增长
            growth = 1.0 + 0.02 * q_idx + 0.005 * price_link * q_idx
            noise = 1.0 + 0.15 * rng.standard_normal()  # ±15% 季度波动
            revenue = base_revenue * growth * noise

            quality_noise = 1.0 + 0.20 * rng.standard_normal()
            # 净利率 = 基础质量 × 噪声, 限制在 0.5%~40%
            net_margin = float(np.clip(base_quality * quality_noise, 0.005, 0.40))
            net_profit = revenue * net_margin
            gross_margin = float(np.clip(net_margin + 0.10 + 0.05 * rng.standard_normal(), 0.05, 0.70))

            equity = revenue * 5 * (1 - base_debt)  # 简化: 净资产 ~ 5 倍营收 × 权益占比
            roe = float(np.clip(net_profit * 4 / equity, -0.10, 0.50))  # 年化 ROE
            roa = float(roe * (1 - base_debt))

            shares = 1e8 * (1 + 0.3 * rng.random())  # 总股本 1~1.3 亿
            eps = net_profit * 1e8 / shares  # 元
            bvps = equity * 1e8 / shares

            row = {
                "eps": eps,
                "bvps": bvps,
                "net_profit": net_profit,
                "revenue": revenue,
                "roe": roe,
                "roa": roa,
                "gross_margin": gross_margin,
                "net_margin": net_margin,
                "debt_ratio": float(np.clip(base_debt + 0.05 * rng.standard_normal(), 0.05, 0.95)),
            }
            rows.append(row)

        df = pd.DataFrame(rows, index=pd.DatetimeIndex(report_dates, name="report_date"))
        # 同比增长: 跟 4 个季度前比 (年报 vs 上年年报)
        df["revenue_yoy"] = df["revenue"].pct_change(periods=4)
        df["profit_yoy"] = df["net_profit"].pct_change(periods=4)
        bundles[sym] = FundamentalsBundle(symbol=sym, quarterly=df)

    return bundles


def _quarterly_dates(start: str, end: str) -> list[pd.Timestamp]:
    """生成区间内所有季末日期 (3-31, 6-30, 9-30, 12-31)."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    # 从 start 所在年份的第一个季末开始
    for year in range(s.year, e.year + 1):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            d = pd.Timestamp(year=year, month=month, day=day)
            if s <= d <= e:
                out.append(d)
    return out


# ---------- 真实数据 (akshare) ----------

def load_fundamentals_real(symbols: list[str], market: str = "cn") -> dict[str, FundamentalsBundle]:
    """
    从 akshare 拉真实财报数据.

    返回: dict[symbol -> FundamentalsBundle]
    失败的 symbol 不抛, 跳过 (打印警告), 让批量调用尽量成功.
    本机有 Clash 全局代理 / akshare 服务不通时, 应自动 fallback 到 generate_synthetic_fundamentals.
    """
    if market != "cn":
        raise NotImplementedError("目前只支持 A 股 (akshare). 美股财报可加 yfinance.Ticker.financials")

    try:
        import akshare as ak
    except ImportError:
        raise ImportError("没装 akshare. pip install akshare")

    bundles = {}
    for sym in symbols:
        code = sym.split(".")[0]  # 去掉 .SH/.SZ 后缀
        try:
            raw = ak.stock_financial_abstract(symbol=code)
        except Exception as e:
            print(f"[fundamentals] {sym} akshare 拉取失败: {e}")
            bundles[sym] = FundamentalsBundle(symbol=sym)
            continue

        if raw is None or raw.empty:
            bundles[sym] = FundamentalsBundle(symbol=sym)
            continue

        bundles[sym] = _parse_akshare_abstract(sym, raw)

    return bundles


def _parse_akshare_abstract(symbol: str, raw: pd.DataFrame) -> FundamentalsBundle:
    """
    把 akshare 的"宽表" (80 指标 × 100+ 季度) 转成我们的"长表" (季度 × 字段).

    akshare schema:
        列 = ['选项', '指标', '20240331', '20231231', ...]
        每行 = 一个财务指标在所有季度的值
    """
    if "指标" not in raw.columns:
        return FundamentalsBundle(symbol=symbol)

    date_cols = [c for c in raw.columns if isinstance(c, str) and len(c) == 8 and c.isdigit()]
    if not date_cols:
        return FundamentalsBundle(symbol=symbol)

    # 对每个我们要的字段, 找到 akshare 里第一个能匹配上的指标行
    result = {}
    for our_name, candidates in AKSHARE_FIELD_MAP.items():
        for cand in candidates:
            matched = raw[raw["指标"] == cand]
            if not matched.empty:
                row = matched.iloc[0][date_cols]
                result[our_name] = pd.to_numeric(row, errors="coerce")
                break

    if not result:
        return FundamentalsBundle(symbol=symbol)

    df = pd.DataFrame(result)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "report_date"
    df = df.sort_index()
    # akshare 的 yoy 字段可能已有, 若缺则自己算
    if "revenue_yoy" not in df.columns and "revenue" in df.columns:
        df["revenue_yoy"] = df["revenue"].pct_change(periods=4)
    if "profit_yoy" not in df.columns and "net_profit" in df.columns:
        df["profit_yoy"] = df["net_profit"].pct_change(periods=4)

    return FundamentalsBundle(symbol=symbol, quarterly=df)


# ---------- 关键: 防未来函数 + 对齐到日频 ----------

def to_daily(
    bundles: dict[str, FundamentalsBundle],
    field: str,
    trading_dates: pd.DatetimeIndex,
    announce_lag_days: int = 45,
) -> pd.DataFrame:
    """
    把季频财报字段按"披露日" 对齐到日频, 形成 (date × symbol) 因子矩阵.

    防未来函数的核心:
        披露日 ≈ 报告期 + announce_lag_days (默认 45 天近似)
        在第 t 天, 因子值 = 最近一次"披露日 ≤ t" 的报告对应字段
        在第一次披露之前的日期, 因子值 = NaN

    示例:
        报告期 2024-03-31, 披露日近似 = 2024-05-15
        → 2024-05-14 时因子值还是 2023Q4 的
        → 2024-05-15 起因子值更新为 2024Q1 的

    参数:
        bundles: load_fundamentals_real 或 generate_synthetic_fundamentals 的返回
        field: 字段名 (如 "roe", "net_margin"); 必须在 AKSHARE_FIELD_MAP 里
        trading_dates: 目标交易日索引 (通常用 prices.index)
        announce_lag_days: 披露日 = 报告期 + 这么多天 (默认 45)

    返回:
        DataFrame, index=trading_dates, columns=symbols, 值 = 已披露的最新字段值
    """
    cols = {}
    for sym, bundle in bundles.items():
        if bundle.quarterly.empty or field not in bundle.quarterly.columns:
            cols[sym] = pd.Series(np.nan, index=trading_dates)
            continue

        q = bundle.quarterly[field].dropna()
        if q.empty:
            cols[sym] = pd.Series(np.nan, index=trading_dates)
            continue

        # 每条季报的可用日 = 报告期 + lag
        available = pd.Series(
            q.values,
            index=q.index + pd.Timedelta(days=announce_lag_days),
        ).sort_index()

        # 把可用值映射到 trading_dates: 用 reindex(method='ffill') 即可
        # — 关键: 第 t 天只能取 available.index <= t 的最近一条
        aligned = available.reindex(
            available.index.union(trading_dates).sort_values(),
            method=None,
        ).ffill().reindex(trading_dates)

        cols[sym] = aligned

    out = pd.DataFrame(cols).reindex(trading_dates)
    out.index.name = "date"
    return out


# ---------- 便捷入口 ----------

def get_fundamental_factor(
    field: str,
    symbols: list[str],
    trading_dates: pd.DatetimeIndex,
    market: str = "cn",
    use_synthetic: bool = False,
    prices: pd.DataFrame | None = None,
    announce_lag_days: int = 45,
    seed: int = 42,
) -> pd.DataFrame:
    """
    一键拉一个基本面因子, 已对齐到日频且防未来函数.

    参数:
        field: 字段名 (eps / bvps / roe / net_margin / revenue_yoy 等)
        symbols: 股票列表
        trading_dates: 目标交易日索引 (通常 prices.index)
        market: "cn" (目前仅支持)
        use_synthetic: True 时直接用模拟数据, 避免依赖网络
        prices: 给模拟数据用, 让"高质量"票跟价格涨幅正相关 (制造教学 alpha)
        announce_lag_days: 披露日 = 报告期 + 这么多天 (默认 45)

    返回:
        DataFrame index=trading_dates, columns=symbols
    """
    start_str = trading_dates.min().strftime("%Y-%m-%d")
    end_str = trading_dates.max().strftime("%Y-%m-%d")

    if use_synthetic:
        bundles = generate_synthetic_fundamentals(symbols, start_str, end_str, seed=seed, prices=prices)
    else:
        try:
            bundles = load_fundamentals_real(symbols, market=market)
            # 如果全军覆没, fallback
            if all(b.quarterly.empty for b in bundles.values()):
                print("[fundamentals] 真实数据全空, fallback 到模拟数据")
                bundles = generate_synthetic_fundamentals(symbols, start_str, end_str, seed=seed, prices=prices)
        except Exception as e:
            print(f"[fundamentals] 真实数据拉取异常, fallback 到模拟数据: {e}")
            bundles = generate_synthetic_fundamentals(symbols, start_str, end_str, seed=seed, prices=prices)

    return to_daily(bundles, field, trading_dates, announce_lag_days=announce_lag_days)
