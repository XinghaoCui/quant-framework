"""把回测结果存成 csv + png 报告。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plotting import plot_equity_curve, plot_drawdown


def save_report(result, name: str, output_dir: str | Path = "reports"):
    """
    把回测结果存到 output_dir/<name>/ 下：
        metrics.csv     指标表
        equity.png      净值曲线
        drawdown.png    回撤曲线
        returns.csv     原始日收益序列（便于二次分析）
    """
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 指标表
    metrics_df = pd.DataFrame(list(result.metrics.items()), columns=["指标", "数值"])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    # 收益序列
    result.returns.to_csv(output_dir / "returns.csv", header=["return"], encoding="utf-8-sig")

    # 图
    plot_equity_curve(
        result.equity_curve,
        result.benchmark_equity,
        title=f"{name} 净值曲线",
        save_path=output_dir / "equity.png",
    )
    plot_drawdown(
        result.returns,
        title=f"{name} 回撤曲线",
        save_path=output_dir / "drawdown.png",
    )

    print(f"报告已保存到 {output_dir}")
    return output_dir
