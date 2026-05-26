"""
示例 31: 有效前沿 + 最大夏普 + 最小方差 在 (波动率, 收益) 平面上可视化
======================================================================

Markowitz 经典图: 在 "波动率 × 期望收益" 坐标系里画出
所有可能的组合, 边界叫"有效前沿".

每个点 = 一组权重 → 算出 (期望收益, 波动率) 两个数 → 一个点.

  期望收益 ↑
          |                ●  最大夏普 (切线点)
          |          ╭───╮
          |        ╭╯      ╲  ← 有效前沿
          |      ╭╯         ╲
          |    ● 最小方差     ╲
          |   ╱
          ────────────────────→ 期望波动率

读完后你能向同学画出这张图, 就理解了"投资组合理论"的灵魂.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data import generate_synthetic_prices
from src.portfolio import (
    efficient_frontier,
    min_variance_weights,
    max_sharpe_weights,
)


def main():
    symbols = [f"STK{i:02d}" for i in range(8)]
    prices = generate_synthetic_prices(symbols, "2022-01-01", "2024-12-31", seed=42)
    returns = prices.pct_change().dropna()

    # 期望收益: 这里偷懒用历史均值 (实战要做 shrinkage)
    mu = returns.mean()
    cov = returns.cov()

    # 1. 算前沿 (30 个目标收益)
    frontier = efficient_frontier(mu, cov, n_points=30)
    print("===== 有效前沿前 5 个点 =====")
    print(frontier.head().round(6).to_string())
    print(f"... 共 {len(frontier)} 个点\n")

    # 2. 算最小方差点
    minv = min_variance_weights(cov)
    minv_ret = float(minv.weights @ mu.values)
    minv_vol = minv.expected_vol
    print(f"最小方差点: 期望收益={minv_ret:.6f}, 波动率={minv_vol:.6f}")

    # 3. 算最大夏普点
    ms = max_sharpe_weights(mu, cov, risk_free=0.0)
    print(f"最大夏普点: 期望收益={ms.expected_return:.6f}, "
          f"波动率={ms.expected_vol:.6f}, 夏普={ms.expected_sharpe:.4f}")

    # 4. 画图保存
    try:
        import matplotlib
        matplotlib.use("Agg")  # 不弹窗, 保存到文件
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(frontier["期望波动率"], frontier["期望收益"], "-", lw=2, label="Efficient Frontier")
        ax.scatter([minv_vol], [minv_ret], color="green", s=150, marker="*",
                   zorder=5, label="Min Variance")
        ax.scatter([ms.expected_vol], [ms.expected_return], color="red", s=150, marker="*",
                   zorder=5, label="Max Sharpe")

        # 顺便点出单个股票
        for sym in returns.columns:
            ax.scatter(returns[sym].std(), returns[sym].mean(),
                       color="gray", alpha=0.5, s=30)
            ax.annotate(sym, (returns[sym].std(), returns[sym].mean()),
                        fontsize=7, alpha=0.6)

        ax.set_xlabel("Expected Volatility (daily std)")
        ax.set_ylabel("Expected Return (daily mean)")
        ax.set_title("Markowitz Efficient Frontier (Synthetic Data)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_dir = Path(__file__).resolve().parents[1] / "reports"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "31_efficient_frontier.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n图已保存: {out_path}")
    except ImportError:
        print("\nmatplotlib 没装, 跳过画图.")

    print("\n小白怎么读:")
    print("  - 灰点 = 单只股票, 散布在图上")
    print("  - 蓝线 = 有效前沿, 同收益下波动最小, 同波动下收益最大")
    print("  - 绿星 = 最小方差点 (左下角)")
    print("  - 红星 = 最大夏普点 (过原点切线的切点)")
    print("  - 理论上你想要的组合都在这条蓝线上, 线下面的都不划算")


if __name__ == "__main__":
    main()
