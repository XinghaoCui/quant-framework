"""
组合优化器: 几种主流"怎么分配权重"的方法.

什么是组合优化?
    选完股票后, 不是一律平均分钱. 高波动的票少分一点、低相关的票多分一点,
    能让组合的"收益/风险" 比简单等权更好. 这件事叫组合优化.

本模块实现 6 种方法, 全部基于 scipy.optimize, 不引入 cvxpy 等额外依赖:

    1. equal_weights              等权: 每只 1/N. 极简, 难打的基准.
    2. inverse_volatility_weights  反波动率: 波动越大权重越小. 风险平价的简化版.
    3. risk_parity_weights         风险平价: 每只票对组合总风险贡献相同.
                                   桥水"全天候"基金的核心思想.
    4. min_variance_weights        最小方差: 不在乎收益, 只追求波动最小.
                                   理论上最稳, 适合不会预测收益的人 (大多数人).
    5. mean_variance_weights       均值-方差 (Markowitz): 经典.
                                   在"期望收益 ≥ 某个值"约束下找方差最小的权重.
    6. max_sharpe_weights          最大夏普: 找夏普最高的组合, 也叫切点组合.

判读经验:
    - 没法预测收益时 → 用 min_variance 或 risk_parity
    - 有信号 (因子打分) → mean_variance 或 max_sharpe
    - 第一次做组合优化, 永远先跟 equal_weights 比, 没明显赢就别用复杂的
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------- 工具 ----------

def annualize_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    把日收益的协方差矩阵年化 → 用于年化夏普等指标.
    协方差年化系数 = periods_per_year (不是 sqrt!).
    """
    return returns.cov() * periods_per_year


def _to_array_cov(returns_or_cov: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str] | None]:
    """
    输入既支持收益 DataFrame, 也支持已算好的协方差矩阵.
    自动判断: 方阵 → 协方差, 否则按收益处理 (调 .cov()).
    """
    if isinstance(returns_or_cov, pd.DataFrame):
        if returns_or_cov.shape[0] == returns_or_cov.shape[1] and (
            returns_or_cov.index.equals(returns_or_cov.columns)
        ):
            return returns_or_cov.values, list(returns_or_cov.columns)
        cov = returns_or_cov.cov().values
        return cov, list(returns_or_cov.columns)
    return np.asarray(returns_or_cov), None


def _safe_cov(cov: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """给协方差矩阵对角加微小扰动, 避免数值奇异 (实战标配)."""
    n = cov.shape[0]
    return cov + jitter * np.eye(n)


@dataclass
class OptimizerResult:
    """优化结果."""
    weights: pd.Series                 # 权重 (index 是股票名)
    method: str                        # 用的哪种优化
    success: bool                      # 是否收敛
    expected_return: float | None = None
    expected_vol: float | None = None
    expected_sharpe: float | None = None
    extras: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        line = f"OptimizerResult(method={self.method}, success={self.success}"
        if self.expected_return is not None:
            line += f", return={self.expected_return:.2%}"
        if self.expected_vol is not None:
            line += f", vol={self.expected_vol:.2%}"
        if self.expected_sharpe is not None:
            line += f", sharpe={self.expected_sharpe:.2f}"
        return line + ")"


def _wrap(weights: np.ndarray, method: str, cols: list[str] | None,
          cov: np.ndarray, expected_returns: np.ndarray | None = None,
          success: bool = True) -> OptimizerResult:
    """统一把数组转成带摘要的 OptimizerResult."""
    if cols is None:
        cols = [f"asset_{i}" for i in range(len(weights))]
    vol = float(np.sqrt(weights @ cov @ weights))
    er = None
    sharpe = None
    if expected_returns is not None:
        er = float(weights @ expected_returns)
        sharpe = er / vol if vol > 0 else 0.0
    return OptimizerResult(
        weights=pd.Series(weights, index=cols),
        method=method,
        success=success,
        expected_return=er,
        expected_vol=vol,
        expected_sharpe=sharpe,
    )


# ---------- 1. 等权 ----------

def equal_weights(returns_or_cov: pd.DataFrame | np.ndarray) -> OptimizerResult:
    """
    每只票 1/N, 没有任何优化.
    永远作为基准存在: 你的复杂方法跑不过它, 复杂方法就没意义.
    """
    cov, cols = _to_array_cov(returns_or_cov)
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    return _wrap(w, "equal", cols, cov)


# ---------- 2. 反波动率 (Inverse Volatility) ----------

def inverse_volatility_weights(returns_or_cov: pd.DataFrame | np.ndarray) -> OptimizerResult:
    """
    权重 ∝ 1 / 波动率. 波动大的票分得少, 波动小的分得多.

    本质: 让每只票贡献的"独立风险" (假设零相关) 大致相等.
    这是真"风险平价" 的一阶近似, 计算非常便宜, 在大部分场景下接近 risk_parity.

    适用: 不知道相关性、只想快速做一个比等权稳的组合.
    """
    cov, cols = _to_array_cov(returns_or_cov)
    vols = np.sqrt(np.diag(cov))
    inv = 1.0 / vols
    w = inv / inv.sum()
    return _wrap(w, "inverse_vol", cols, cov)


# ---------- 3. 风险平价 (Risk Parity) ----------

def risk_parity_weights(
    returns_or_cov: pd.DataFrame | np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 5000,
) -> OptimizerResult:
    """
    严格风险平价: 让每只票对组合总风险的"边际贡献" 相等.

    数学:
        组合方差     σ² = w' Σ w
        票 i 的风险贡献 RC_i = w_i × (Σ w)_i / σ
        目标          RC_i 都相等 = σ / N

    用 SLSQP 最小化 (RC_i - mean(RC))² 之和, 满足 sum(w)=1, w>=0.
    这是 Bridgewater "All Weather" 基金的内核思想.
    """
    cov_raw, cols = _to_array_cov(returns_or_cov)
    cov = _safe_cov(cov_raw)
    n = cov.shape[0]

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        port_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w / port_vol
        return w * marginal

    def _objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w)
        return float(np.sum((rc - rc.mean()) ** 2))

    w0 = np.full(n, 1.0 / n)
    bounds = [(1e-8, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(_objective, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": max_iter, "ftol": tol})
    return _wrap(res.x, "risk_parity", cols, cov_raw, success=bool(res.success))


# ---------- 4. 最小方差 ----------

def min_variance_weights(
    returns_or_cov: pd.DataFrame | np.ndarray,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> OptimizerResult:
    """
    在 sum(w)=1 (可选 w>=0) 下, 找让组合方差最小的 w.

    解析解 (无约束): w ∝ Σ⁻¹ 1, 然后归一化.
    带"不卖空"约束就没法解析解, 用 SLSQP.

    适用: 你完全不会预测收益时, 用最小方差是最理性的选择 ("贝叶斯先验认为
    所有票期望收益一样, 那就只剩"哪个组合最不波动" 这一个目标").
    """
    cov_raw, cols = _to_array_cov(returns_or_cov)
    cov = _safe_cov(cov_raw)
    n = cov.shape[0]

    def _objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    w0 = np.full(n, 1.0 / n)
    bounds = [((-1.0 if allow_short else 0.0), (max_weight if max_weight else 1.0))] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(_objective, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    return _wrap(res.x, "min_variance", cols, cov_raw, success=bool(res.success))


# ---------- 5. 均值-方差 (Markowitz) ----------

def mean_variance_weights(
    expected_returns: pd.Series | np.ndarray,
    returns_or_cov: pd.DataFrame | np.ndarray,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> OptimizerResult:
    """
    Markowitz 均值-方差: 最大化  μ'w - λ/2 × w'Σw.

    risk_aversion (λ) 控制"想要多稳":
        λ 小 → 追求高收益, 接受高波动
        λ 大 → 追求低波动, 接受低收益

    实战警告: 这玩意对 expected_returns 的输入**极度敏感**, 微小误差就让
    权重剧烈变化, 甚至全压一只票. 直接用历史均值当 expected_returns 是
    经典灾难案例. 工程上通常用 shrinkage (Ledoit-Wolf) 或 Black-Litterman
    先把 μ 收缩. 这一点 08 章 / 05 约束与稳健性.md 会展开.
    """
    cov_raw, cols = _to_array_cov(returns_or_cov)
    cov = _safe_cov(cov_raw)
    n = cov.shape[0]
    mu = np.asarray(expected_returns, dtype=float)
    if len(mu) != n:
        raise ValueError(f"expected_returns 长度 {len(mu)} 与协方差维度 {n} 不一致")

    def _neg_utility(w: np.ndarray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

    w0 = np.full(n, 1.0 / n)
    bounds = [((-1.0 if allow_short else 0.0), (max_weight if max_weight else 1.0))] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(_neg_utility, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    return _wrap(res.x, "mean_variance", cols, cov_raw,
                 expected_returns=mu, success=bool(res.success))


# ---------- 6. 最大夏普 (切点组合) ----------

def max_sharpe_weights(
    expected_returns: pd.Series | np.ndarray,
    returns_or_cov: pd.DataFrame | np.ndarray,
    risk_free: float = 0.0,
    allow_short: bool = False,
    max_weight: float | None = None,
) -> OptimizerResult:
    """
    最大化 (μ - r_f)'w / sqrt(w'Σw), 也就是组合夏普.

    几何上, 这是有效前沿上"过无风险点切线" 的切点 → 也叫切点组合 (tangency portfolio).
    经典金融理论 (CAPM) 认为, 全市场最优组合就是这个切点组合 + 一份无风险资产.

    实战警告: 同 mean_variance, 对 expected_returns 极度敏感.
    """
    cov_raw, cols = _to_array_cov(returns_or_cov)
    cov = _safe_cov(cov_raw)
    n = cov.shape[0]
    mu = np.asarray(expected_returns, dtype=float)

    def _neg_sharpe(w: np.ndarray) -> float:
        port_ret = (mu - risk_free) @ w
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return float(-port_ret / port_vol)

    w0 = np.full(n, 1.0 / n)
    bounds = [((-1.0 if allow_short else 0.0), (max_weight if max_weight else 1.0))] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(_neg_sharpe, w0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
    return _wrap(res.x, "max_sharpe", cols, cov_raw,
                 expected_returns=mu, success=bool(res.success))


# ---------- 有效前沿 (画图用) ----------

def efficient_frontier(
    expected_returns: pd.Series | np.ndarray,
    returns_or_cov: pd.DataFrame | np.ndarray,
    n_points: int = 30,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    画 Markowitz 有效前沿用. 在期望收益区间内取 n_points 个目标收益,
    对每个目标算"达到该收益时方差最小的权重" → 得到前沿上一系列点.

    返回: DataFrame, index=点编号, columns=[期望收益, 期望波动率, 夏普]
    """
    cov_raw, cols = _to_array_cov(returns_or_cov)
    cov = _safe_cov(cov_raw)
    n = cov.shape[0]
    mu = np.asarray(expected_returns, dtype=float)

    targets = np.linspace(mu.min(), mu.max(), n_points)
    rows = []

    bounds = [((-1.0 if allow_short else 0.0), 1.0)] * n
    for target in targets:
        def _obj(w):
            return float(w @ cov @ w)
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=target: mu @ w - t},
        ]
        w0 = np.full(n, 1.0 / n)
        res = minimize(_obj, w0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
        if not res.success:
            continue
        w = res.x
        vol = float(np.sqrt(w @ cov @ w))
        ret = float(mu @ w)
        rows.append({"期望收益": ret, "期望波动率": vol, "夏普": ret / vol if vol > 0 else 0.0})

    return pd.DataFrame(rows)
