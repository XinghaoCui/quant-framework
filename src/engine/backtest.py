"""
向量化回测引擎。

核心思想：
    给定"目标仓位矩阵 weights"（每天每只票应该持有的权重），
    用矩阵乘法一次性算出整个回测期的组合收益。
    不做"逐 bar 撮合"，而是直接对收益率做加权求和。

为什么是向量化？
    1. 速度快：10 年日线 × 500 只票 → 几毫秒
    2. 代码短：策略只需要输出"weights"这个 DataFrame，其他都是引擎的事
    3. 易于做大规模因子测试

局限：
    不适合"事件驱动"型策略（如限价单、止损止盈、订单簿撮合）。
    那种场景应该用 backtrader 等事件驱动引擎。

防止未来函数（look-ahead bias）：
    在 t 日盘后计算的信号，只能在 t+1 日开盘后才能交易。
    所以引擎默认会把 weights shift(1)：今天的仓位 = 昨天计算出的目标仓位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestResult:
    """回测结果容器。"""
    returns: pd.Series              # 策略日收益率序列
    equity_curve: pd.Series         # 净值曲线（起点为 1.0）
    positions: pd.DataFrame         # 实际持仓权重（已 shift）
    trades: pd.DataFrame            # 每日换手率
    metrics: dict[str, float] = field(default_factory=dict)
    benchmark_returns: pd.Series | None = None
    benchmark_equity: pd.Series | None = None

    def summary(self) -> str:
        """打印一份可读的摘要。"""
        from .metrics import format_metrics
        out = ["策略回测结果"]
        out.append(format_metrics(self.metrics))
        if self.benchmark_returns is not None:
            from .metrics import compute_metrics as _cm
            bench_metrics = _cm(self.benchmark_returns)
            out.append("基准对照")
            out.append(format_metrics(bench_metrics))
        return "\n".join(out)


class Backtester:
    """
    向量化回测器。

    用法：
        bt = Backtester(commission=0.0003, slippage=0.0001)
        result = bt.run(prices, weights, benchmark=None)
        print(result.summary())

    参数说明：
        commission: 双边佣金率，默认万 3
        slippage:   单边滑点，默认万 1
                    总交易成本 ≈ 换手率 × (佣金 + 滑点)
    """

    def __init__(
        self,
        commission: float = 0.0003,
        slippage: float = 0.0001,
        risk_free: float = 0.0,
    ):
        self.commission = commission
        self.slippage = slippage
        self.risk_free = risk_free

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> BacktestResult:
        """
        运行回测。

        参数:
            prices: 价格宽表，index=date, columns=symbol, values=close
            weights: 目标权重宽表，index=date, columns=symbol, values ∈ [-1, 1]
                     权重之和应该 ≤ 1（不加杠杆）；可以是负数（做空）
            benchmark: 可选基准日收益率序列（如指数）

        返回:
            BacktestResult
        """
        # 计算每只票的日收益率（用收盘价的简单收益率）
        asset_returns = prices.pct_change().fillna(0.0)

        # 对齐：weights 和 prices 共同的日期、列
        weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

        # 防止未来函数：今天能交易的仓位是昨天盘后算出的
        positions = weights.shift(1).fillna(0.0)

        # 组合收益 = 各资产收益的加权和
        gross_returns = (positions * asset_returns).sum(axis=1)

        # 换手率：今天 vs 昨天的仓位变化（绝对值之和）
        turnover = (positions - positions.shift(1).fillna(0.0)).abs().sum(axis=1)

        # 交易成本：换手率 × (佣金 + 滑点)
        # 注意：买卖一次完整往返是双边，所以 commission 是双边率
        cost = turnover * (self.commission + self.slippage)

        net_returns = gross_returns - cost

        # 净值曲线
        equity = (1.0 + net_returns).cumprod()

        # 指标
        metrics = compute_metrics(net_returns, self.risk_free)
        metrics["平均换手率(单边)"] = float(turnover.mean() / 2)
        metrics["总交易成本"] = float(cost.sum())

        # 基准
        bench_eq = None
        if benchmark is not None:
            benchmark = benchmark.reindex(prices.index).fillna(0.0)
            bench_eq = (1.0 + benchmark).cumprod()

        return BacktestResult(
            returns=net_returns,
            equity_curve=equity,
            positions=positions,
            trades=pd.DataFrame({"turnover": turnover, "cost": cost}),
            metrics=metrics,
            benchmark_returns=benchmark,
            benchmark_equity=bench_eq,
        )
