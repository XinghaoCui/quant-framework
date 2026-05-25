"""
绩效指标计算。

所有指标的输入都是"日度收益率序列"（pd.Series），不是价格。
日度收益率 r_t = P_t / P_{t-1} - 1。

为什么用日收益率？
    因为大多数指标在数学定义上就是基于收益率，
    且日收益率分布近似独立同分布，便于统计推断（如年化、计算波动率）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252  # 美股一年约 252 个交易日；A 股约 242，但学术界惯用 252


def cum_returns(returns: pd.Series) -> pd.Series:
    """累计收益曲线：(1+r1)(1+r2)... - 1。如果策略不亏不赚，曲线水平。"""
    return (1.0 + returns).cumprod() - 1.0


def annual_return(returns: pd.Series) -> float:
    """
    年化收益率。
    用几何平均：(1 + total_return) ** (252 / N) - 1
    其中 N 是样本天数，total_return 是累计收益。

    为什么不用算术平均？
        算术平均会高估真实收益。
        例：第一天 +50%，第二天 -50%，算术平均 = 0%，但实际累计 = -25%。
        几何平均才是"等价的稳定增长率"。
    """
    if len(returns) == 0:
        return 0.0
    total = (1.0 + returns).prod()
    return float(total ** (TRADING_DAYS / len(returns)) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """
    年化波动率 = 日波动率 × √252。
    √252 来自"独立同分布下方差可加"的假设：
        Var(年化收益) = 252 × Var(日收益)，所以 Std 是 √252 × 日 Std。
    """
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    夏普比率：(年化超额收益) / 年化波动率。
    衡量"每承担一单位波动率，能获得多少超额收益"。

    经验解读：
        Sharpe < 1：策略一般
        1 ~ 2：可投资
        > 2：优秀（但要警惕过拟合）
        > 3：极少见，几乎可以确定是回测过拟合或数据错误
    """
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """
    Sortino 比率：只把"下行波动"算进分母。
    思路是：上行波动是"赚钱波动"，不该被惩罚，只有亏钱的波动才是真正的风险。
    分母 = sqrt(mean(min(r-rf, 0)^2)) * sqrt(252)。
    """
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    return float(excess.mean() * TRADING_DAYS / downside_std)


def max_drawdown(returns: pd.Series) -> float:
    """
    最大回撤：从历史最高净值到后续最低净值的跌幅（负数）。
    这是衡量"持有这个策略时心理上能承受多大痛苦"的指标。

    计算：
        net = (1+r).cumprod()
        peak = net.cummax()  # 截止到当下的历史最高点
        drawdown = net / peak - 1  # 当前相对历史高点的跌幅
        max_dd = drawdown.min()  # 最深那一刻

    回撤 -50% 比 -20% 严重得多：跌 50% 之后需要 +100% 才能回本。
    """
    if len(returns) == 0:
        return 0.0
    net = (1.0 + returns).cumprod()
    peak = net.cummax()
    drawdown = net / peak - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar 比率 = 年化收益 / |最大回撤|。比较"收益相对最痛点"的性价比。"""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annual_return(returns) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """胜率：盈利天数 / 总天数。注意胜率高不代表赚钱（可能小赢多次大亏一次）。"""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def profit_loss_ratio(returns: pd.Series) -> float:
    """盈亏比：平均盈利 / 平均亏损（取绝对值）。和胜率配合看才有意义。"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return float("inf")
    return float(wins.mean() / abs(losses.mean()))


def compute_metrics(returns: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """
    一次性计算所有常用指标。返回字典方便打印或存表。
    """
    return {
        "累计收益": float((1.0 + returns).prod() - 1.0),
        "年化收益": annual_return(returns),
        "年化波动率": annual_volatility(returns),
        "夏普比率": sharpe_ratio(returns, risk_free),
        "Sortino比率": sortino_ratio(returns, risk_free),
        "最大回撤": max_drawdown(returns),
        "Calmar比率": calmar_ratio(returns),
        "胜率": win_rate(returns),
        "盈亏比": profit_loss_ratio(returns),
        "交易天数": len(returns),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """把指标字典格式化成漂亮的字符串，方便打印。"""
    lines = ["=" * 40]
    for k, v in metrics.items():
        if k in ("累计收益", "年化收益", "年化波动率", "最大回撤", "胜率"):
            lines.append(f"  {k:12s}: {v:>10.2%}")
        elif k == "交易天数":
            lines.append(f"  {k:12s}: {v:>10.0f}")
        else:
            lines.append(f"  {k:12s}: {v:>10.3f}")
    lines.append("=" * 40)
    return "\n".join(lines)
