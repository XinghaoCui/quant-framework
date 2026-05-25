"""
ML 模型评估 —— 但用"量化的方式"评估，而不是普通 ML 的方式。

普通 ML 看准确率/R²/MSE。但在量化里这些几乎没用：
    - 预测收益的 R² 通常只有 0.001~0.01（信噪比极低），看着像没用，其实能赚钱
    - 准确率 51% 听着差，但放大到几千次交易就是稳定 alpha

量化关心的是"预测的排序能不能赚钱"：
    1. IC (信息系数)：预测值和真实收益的相关性
    2. 分组回测：按预测值分组，看高分组是否真的跑赢低分组
    3. 多空收益：买预测最高组、卖预测最低组的收益
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    prediction: pd.Series,
    actual: pd.Series,
    method: str = "spearman",
) -> dict[str, float]:
    """
    计算 IC：每个截面（每天）上，预测值与真实收益的相关系数，再取时序均值。

    参数:
        prediction: 模型预测值，MultiIndex=(date, symbol)
        actual: 真实未来收益，同索引
        method: "spearman" (Rank IC, 更稳健) / "pearson"

    返回:
        dict: IC 均值、IC 标准差、ICIR、IC>0 占比

    判读:
        |IC| > 0.03 已经可用，> 0.05 不错，> 0.1 很强
        ICIR > 0.5 说明 IC 稳定
    """
    df = pd.DataFrame({"pred": prediction, "actual": actual}).dropna()

    # 按日期分组，每天算一个截面相关系数
    def _corr(g):
        if len(g) < 3:
            return np.nan
        return g["pred"].corr(g["actual"], method=method)

    daily_ic = df.groupby(level="date").apply(_corr).dropna()

    if len(daily_ic) == 0:
        return {"IC均值": 0.0, "IC标准差": 0.0, "ICIR": 0.0, "IC>0占比": 0.0}

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    return {
        "IC均值": float(ic_mean),
        "IC标准差": float(ic_std),
        "ICIR": float(ic_mean / ic_std) if ic_std > 0 else 0.0,
        "IC>0占比": float((daily_ic > 0).mean()),
    }


def quantile_backtest(
    prediction: pd.Series,
    actual: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    分组回测：每天按预测值把股票分成 n 组，看各组的平均真实收益。

    如果模型有效，应该看到：
        最高分组(Q5) 平均收益 > ... > 最低分组(Q1)
    呈现单调递增，这叫"单调性"，是因子/模型有效的有力证据。

    返回:
        DataFrame，index=分组(Q1..Qn)，列含 平均收益、累计收益。
    """
    df = pd.DataFrame({"pred": prediction, "actual": actual}).dropna()

    def _assign_q(g):
        if len(g) < n_quantiles:
            g["q"] = np.nan
            return g
        # 按预测值分位数打标签 Q1(最低) .. Qn(最高)
        g["q"] = pd.qcut(g["pred"], n_quantiles, labels=False, duplicates="drop")
        return g

    df = df.groupby(level="date", group_keys=False).apply(_assign_q).dropna()

    # 每组每天的平均收益
    grouped = df.groupby(["date", "q"])["actual"].mean().unstack()

    result = pd.DataFrame({
        "平均收益": grouped.mean(),
        "收益标准差": grouped.std(),
        "累计收益": (1 + grouped).prod() - 1,
    })
    result.index = [f"Q{int(i)+1}" for i in result.index]
    # 多空价差（最高组 - 最低组）
    result.attrs["多空价差_日均"] = float(grouped.iloc[:, -1].mean() - grouped.iloc[:, 0].mean())
    return result


def regression_metrics(prediction: pd.Series, actual: pd.Series) -> dict[str, float]:
    """传统回归指标（参考用，量化里 R² 低是常态，别被吓到）。"""
    from sklearn.metrics import r2_score, mean_squared_error
    df = pd.DataFrame({"pred": prediction, "actual": actual}).dropna()
    return {
        "R2": float(r2_score(df["actual"], df["pred"])),
        "RMSE": float(np.sqrt(mean_squared_error(df["actual"], df["pred"]))),
        "预测真实相关性": float(df["pred"].corr(df["actual"])),
    }
