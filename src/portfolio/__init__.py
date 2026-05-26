"""
组合优化模块.

把"选哪些股票" (因子打分 / ML 预测的工作) 和
"每只票分多少仓位" (组合优化的工作) 分开.

本模块只负责后者: 给定一组候选股票和它们的历史收益, 算出
"该给每只票多少权重" —— 等权 / 最小方差 / 风险平价 / 均值方差 / 最大夏普.

详见 `思考与学习/08_组合优化/`.
"""

from .optimizer import (
    equal_weights,
    inverse_volatility_weights,
    risk_parity_weights,
    min_variance_weights,
    max_sharpe_weights,
    mean_variance_weights,
    efficient_frontier,
    annualize_cov,
    OptimizerResult,
)

__all__ = [
    "equal_weights",
    "inverse_volatility_weights",
    "risk_parity_weights",
    "min_variance_weights",
    "max_sharpe_weights",
    "mean_variance_weights",
    "efficient_frontier",
    "annualize_cov",
    "OptimizerResult",
]
