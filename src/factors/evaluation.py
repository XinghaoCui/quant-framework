"""
因子评估工具
============

研究一个因子, 量化研究员每天都在问的四个问题:
    1. 有用吗?           → IC / Rank IC (今日因子值能不能预测未来收益)
    2. 稳不稳?           → IR / IC>0 占比 / t 统计 (IC 是不是稳定为正)
    3. 单调吗?           → 分组回测 (高分组真的跑赢低分组吗)
    4. 信号能维持多久?    → IC 衰减曲线 (1 天 IC vs 60 天 IC)

本模块把这四件事打包成独立函数. 输入约定:
    factor : 因子宽表, index=date, columns=symbol, 值越大越看多
    prices : 价格宽表, 同形状

这种宽表和 src/strategies/ 里因子的输出形式一致, 拿来即用.

为什么不直接复用 src/ml/evaluation.py?
    ML 模块里的 IC 是 "长表" 版 (MultiIndex (date, symbol) 的 Series),
    那是为模型预测设计的. 因子研究用宽表更自然 —— 一行就是一个截面,
    pandas 的 corrwith(axis=1) 一行算出所有日期的 IC, 又快又直观.

判读经验 (A 股 + 美股股票池):
    |IC 均值| > 0.02   勉强可用
    |IC 均值| > 0.05   不错
    |IC 均值| > 0.10   非常强 (要警惕过拟合)
    |IR|     > 0.5    IC 稳定
    |IR|     > 1.0    顶级因子
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------- 工具函数 ----------

def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    计算未来 horizon 天的收益率.

    用法: forward_returns(prices, 5)[t] = prices[t+5] / prices[t] - 1
    最后 horizon 行会是 NaN (没有未来数据).
    """
    return prices.shift(-horizon) / prices - 1.0


# ---------- IC / Rank IC ----------

def compute_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    每天算一次截面 IC, 返回 IC 时间序列.

    截面 IC 的含义: 在 t 这一天, 把所有股票的因子值和它们 horizon 天后
    的真实收益做相关, 得到一个数. 这个数衡量 "因子排序能不能预测收益排序".

    method:
        spearman → Rank IC, 只看排序, 抗异常值 (推荐, 业界默认)
        pearson  → 普通 IC, 假设线性, 对异常值敏感
    """
    factor, forward_ret = factor.align(forward_ret, join="inner")
    # corrwith(axis=1) 是 "逐行算相关": 一行 = 一天 = 一个截面
    return factor.corrwith(forward_ret, axis=1, method=method).dropna()


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """
    把 IC 时序汇总成几个判读指标.

    指标含义:
        IC均值     主信号, 看正负和量级
        IC标准差   IC 的波动
        IR        = IC均值 / IC标准差, 信息比率, 衡量稳定性
        t统计     = IR * sqrt(N), 统计显著性 (|t|>2 大致 5% 显著)
        IC>0占比  IC 为正的天数占比, 直观看稳定度
        正IC均值/负IC均值  上下两侧的平均大小
    """
    s = ic_series.dropna()
    if len(s) == 0:
        return {k: 0.0 for k in ["IC均值", "IC标准差", "IR", "t统计", "IC>0占比", "样本天数"]}

    mean = float(s.mean())
    std = float(s.std())
    ir = mean / std if std > 0 else 0.0
    n = len(s)
    t_stat = ir * np.sqrt(n)

    return {
        "IC均值": mean,
        "IC标准差": std,
        "IR": float(ir),
        "t统计": float(t_stat),
        "IC>0占比": float((s > 0).mean()),
        "样本天数": int(n),
    }


# ---------- 分组回测 ----------

def quantile_labels(factor: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    把因子值按日期分位数打标签: 0=最低组(Q1), n-1=最高组(Qn).

    每天独立分组. 标的太少的日期 (<n_quantiles) 返回 NaN.
    """
    def _per_row(row: pd.Series) -> pd.Series:
        clean = row.dropna()
        if len(clean) < n_quantiles:
            return pd.Series(np.nan, index=row.index)
        # qcut 把 [min,max] 切成 n 个等频区间, labels=False 返回 0..n-1
        labels = pd.qcut(clean, n_quantiles, labels=False, duplicates="drop")
        return labels.reindex(row.index)

    return factor.apply(_per_row, axis=1)


def quantile_returns(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    每天把股票按因子值分 n 组, 返回 (date × Qi) 的等权平均未来收益.

    输出列名: Q1 (最低) ... Qn (最高). 列里的值是 "若 t 日持有这一组,
    horizon 天后的等权平均收益". 这是观察因子单调性的核心数据.
    """
    factor, forward_ret = factor.align(forward_ret, join="inner")
    labels = quantile_labels(factor, n_quantiles)

    # 把 (factor 标签, forward 收益) 拉长成长表, 然后 groupby 求均值
    long = pd.DataFrame({
        "label": labels.stack(future_stack=True),
        "ret": forward_ret.stack(future_stack=True),
    }).dropna()

    grouped = long.groupby([long.index.get_level_values(0), "label"])["ret"].mean()
    wide = grouped.unstack("label")
    wide.columns = [f"Q{int(c) + 1}" for c in wide.columns]
    return wide.sort_index()


@dataclass
class QuantileBacktestResult:
    """分组回测的完整结果."""
    daily_returns: pd.DataFrame      # 每组每日收益 (Q1..Qn)
    cum_returns: pd.DataFrame        # 每组累计净值曲线 (起点 1.0)
    long_short: pd.Series            # 多空价差 Qn - Q1 的日收益
    long_short_equity: pd.Series     # 多空价差累计净值
    summary: pd.DataFrame            # 各组的年化收益/夏普/胜率


def quantile_backtest(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    n_quantiles: int = 5,
    horizon: int = 1,
) -> QuantileBacktestResult:
    """
    把分组的"未来 horizon 天收益"折算成日频净值曲线, 顺带算多空价差.

    注意:
        horizon=1 (持有 1 天) 时, 这就是真实可执行策略的近似;
        horizon>1 时, 等价于 "每天用因子分组, 持有 horizon 天再换",
        相当于很多条 horizon 天持仓的滚动平均, 用于研究信号衰减,
        不直接对应实盘策略 (实盘要看 src/strategies/multi_factor.py).
    """
    fwd = forward_returns(prices, horizon)
    qrets = quantile_returns(factor, fwd, n_quantiles)

    # 折算到 "等价日频": horizon 天收益 / horizon (近似, 用于跨 horizon 横比)
    daily = qrets / horizon
    cum = (1.0 + daily.fillna(0.0)).cumprod()

    long_short = daily.iloc[:, -1] - daily.iloc[:, 0]  # Qn - Q1
    ls_equity = (1.0 + long_short.fillna(0.0)).cumprod()

    # 各组摘要
    summary_rows = {}
    for col in daily.columns:
        s = daily[col].dropna()
        if len(s) == 0:
            continue
        ann_ret = float((1 + s.mean()) ** 252 - 1)
        ann_vol = float(s.std() * np.sqrt(252))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        summary_rows[col] = {
            "年化收益": ann_ret,
            "年化波动": ann_vol,
            "夏普": sharpe,
            "胜率": float((s > 0).mean()),
        }
    # 多空一行
    s = long_short.dropna()
    ann_ret = float((1 + s.mean()) ** 252 - 1) if len(s) else 0.0
    ann_vol = float(s.std() * np.sqrt(252)) if len(s) else 0.0
    summary_rows["多空(Qn-Q1)"] = {
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "胜率": float((s > 0).mean()) if len(s) else 0.0,
    }
    summary = pd.DataFrame(summary_rows).T

    return QuantileBacktestResult(
        daily_returns=daily,
        cum_returns=cum,
        long_short=long_short,
        long_short_equity=ls_equity,
        summary=summary,
    )


# ---------- IC 衰减 ----------

def ic_decay(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: list[int] | tuple[int, ...] = (1, 3, 5, 10, 20, 40, 60),
    method: str = "spearman",
) -> pd.DataFrame:
    """
    扫一遍多个 horizon, 看 IC 怎么衰减.

    一个 "持仓 1 天" IC 高但 "持仓 20 天" IC 跌到 0 的因子, 说明信号短命,
    实盘要高频换仓, 交易成本会吃掉大半收益. 反过来 IC 持续到 60 天的因子
    更稳, 适合低频策略.

    返回: DataFrame, index=horizon, columns=[IC均值, IR, t统计, IC>0占比, 样本天数]
    """
    rows = {}
    for h in horizons:
        fwd = forward_returns(prices, h)
        ic = compute_ic(factor, fwd, method=method)
        rows[h] = ic_summary(ic)
    out = pd.DataFrame(rows).T
    out.index.name = "horizon"
    return out


# ---------- 综合报告 ----------

@dataclass
class FactorReport:
    """单因子综合研究报告."""
    name: str
    ic_series: pd.Series                    # IC 时序
    ic_stats: dict[str, float]              # 主 IC 摘要 (默认 horizon)
    decay: pd.DataFrame                     # IC 衰减表
    quantile: QuantileBacktestResult        # 分组回测

    def to_text(self) -> str:
        lines = [f"========== 因子研究报告: {self.name} ==========", ""]
        lines.append(f"[IC 摘要 (horizon=主)]")
        for k, v in self.ic_stats.items():
            lines.append(f"  {k:<10} {v:.4f}" if isinstance(v, float) else f"  {k:<10} {v}")
        lines += ["", "[IC 衰减]"]
        lines.append(self.decay.round(4).to_string())
        lines += ["", "[分组回测摘要]"]
        lines.append(self.quantile.summary.round(4).to_string())
        return "\n".join(lines)


def factor_report(
    factor: pd.DataFrame,
    prices: pd.DataFrame,
    name: str = "factor",
    horizon: int = 5,
    n_quantiles: int = 5,
    decay_horizons: list[int] | tuple[int, ...] = (1, 3, 5, 10, 20, 40, 60),
    method: str = "spearman",
) -> FactorReport:
    """
    一次性把单因子的 "有用吗 / 稳不稳 / 单调吗 / 信号多长" 全跑一遍.

    给入门者的建议: 拿到任何新因子, 先 factor_report 跑一份, 不要急着上策略.
    """
    fwd = forward_returns(prices, horizon)
    ic = compute_ic(factor, fwd, method=method)
    stats = ic_summary(ic)
    decay = ic_decay(factor, prices, decay_horizons, method=method)
    qbt = quantile_backtest(factor, prices, n_quantiles=n_quantiles, horizon=horizon)
    return FactorReport(
        name=name,
        ic_series=ic,
        ic_stats=stats,
        decay=decay,
        quantile=qbt,
    )
