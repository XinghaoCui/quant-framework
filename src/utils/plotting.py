"""画图工具：净值曲线、回撤曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 兼容中文显示（Windows 系统字体）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_equity_curve(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    title: str = "策略净值曲线",
    save_path: str | Path | None = None,
):
    """
    画净值曲线，可选加基准对照。
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="策略", linewidth=1.5)
    if benchmark_equity is not None:
        ax.plot(benchmark_equity.index, benchmark_equity.values,
                label="基准", linewidth=1.0, alpha=0.7, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("净值（起点=1.0）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_drawdown(returns: pd.Series, title: str = "回撤曲线", save_path: str | Path | None = None):
    """画回撤曲线（一直为负或零的曲线，越深越糟糕）。"""
    net = (1.0 + returns).cumprod()
    peak = net.cummax()
    dd = net / peak - 1.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.plot(dd.index, dd.values, color="red", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("回撤")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
