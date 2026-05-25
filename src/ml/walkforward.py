"""
滚动重训（Walk-Forward）—— 让 ML 策略贴近实盘的关键。

问题：
    "训练一次"的模型假设市场规律永远不变。
    但市场是非平稳的：因子会衰减、régime 会切换、有效的规律会失效。
    用 2015 年学的模型预测 2024 年，等于刻舟求剑。

解决：
    周期性地用"截至当前的最新数据"重新训练模型，再预测下一段。
    这模拟了实盘中"每季度/每年重训一次模型"的真实流程。

两种窗口模式：
    expanding（扩展窗口）：训练集起点固定，终点不断后移 → 训练集越来越大。
        优点：数据利用充分。缺点：很久以前的旧规律一直影响模型。
    rolling（滚动窗口）：训练集长度固定，起点终点一起后移 → 只用最近一段。
        优点：能"遗忘"过时规律，更快适应市场变化。缺点：丢弃了老数据。

防泄露（和 time_series_split 一致）：
    每次训练用 [train_start, train_end] 的数据，
    预测段从 train_end + embargo 之后才开始（embargo ≥ horizon），
    保证训练样本的"未来标签期"不和预测段重叠。

时间轴示意（expanding，retrain_freq=季度）：
    |---- 初始训练 ----|emb|-- 预测Q1 --|
    |------- 训练 ------|emb|-- 预测Q2 --|
    |--------- 训练 -----|emb|-- 预测Q3 --|
    各预测段无缝拼接 → 完整的样本外预测序列
"""

from __future__ import annotations

import pandas as pd

from .features import make_dataset
from .models import ModelWrapper


def walk_forward_predict(
    prices: pd.DataFrame,
    model_name: str = "ridge",
    task: str = "regression",
    horizon: int = 5,
    initial_train_days: int = 504,
    retrain_freq: int = 63,
    window: str = "expanding",
    rolling_window_days: int = 504,
    embargo_days: int | None = None,
    model_kwargs: dict | None = None,
    verbose: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    滚动重训并产出完整的样本外预测。

    参数:
        prices: 价格宽表 (index=date, columns=symbol)
        model_name: "ridge"/"rf"/"gbdt" 等
        task: "regression" / "classification"
        horizon: 预测未来多少天收益
        initial_train_days: 初始训练窗口长度（交易日），之后才开始预测
        retrain_freq: 每隔多少天重训一次（如 63≈季度, 252≈年）
        window: "expanding"（扩展窗口）/ "rolling"（滚动窗口）
        rolling_window_days: rolling 模式下训练窗口固定长度
        embargo_days: 训练末尾和预测之间的隔离带，默认 = horizon
        model_kwargs: 透传给模型的超参数
        verbose: 是否打印每轮重训进度

    返回:
        predictions: 样本外预测 Series，MultiIndex=(date, symbol)
        log: DataFrame，每轮重训的记录（训练区间、预测区间、样本数）
    """
    if window not in ("expanding", "rolling"):
        raise ValueError(f"window 必须是 'expanding' 或 'rolling'，收到 {window}")

    kind = "binary" if task == "classification" else "return"
    X, y = make_dataset(prices, horizon=horizon, kind=kind)

    embargo = embargo_days if embargo_days is not None else horizon

    # 所有唯一交易日，排序
    dates = X.index.get_level_values("date").unique().sort_values()
    n = len(dates)
    date_level = X.index.get_level_values("date")

    preds: list[pd.Series] = []
    log_rows: list[dict] = []

    i = initial_train_days
    while i + embargo < n:
        # --- 训练区间 ---
        train_end_date = dates[i]
        if window == "expanding":
            train_start_idx = 0
        else:  # rolling
            train_start_idx = max(0, i - rolling_window_days)
        train_start_date = dates[train_start_idx]

        # --- 预测区间（留 embargo 隔离带，无缝拼接到下一段）---
        pred_start_idx = i + embargo
        pred_end_idx = min(i + embargo + retrain_freq, n)  # 右开
        pred_dates = dates[pred_start_idx:pred_end_idx]
        if len(pred_dates) == 0:
            break

        train_mask = (date_level >= train_start_date) & (date_level <= train_end_date)
        pred_mask = date_level.isin(pred_dates)

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_pr = X[pred_mask]

        if len(X_tr) < 50 or len(X_pr) == 0:
            i += retrain_freq
            continue

        # --- 训练 + 预测 ---
        model = ModelWrapper(model_name, task=task, **(model_kwargs or {}))
        model.fit(X_tr, y_tr)
        preds.append(model.predict(X_pr))

        log_rows.append({
            "重训序号": len(log_rows) + 1,
            "训练起": str(train_start_date.date()),
            "训练止": str(train_end_date.date()),
            "训练样本": len(X_tr),
            "预测起": str(pred_dates[0].date()),
            "预测止": str(pred_dates[-1].date()),
            "预测样本": len(X_pr),
        })
        if verbose:
            r = log_rows[-1]
            print(f"  [第{r['重训序号']:>2}轮] 训练 {r['训练起']}~{r['训练止']} "
                  f"({r['训练样本']}样本) → 预测 {r['预测起']}~{r['预测止']}")

        i += retrain_freq

    if not preds:
        raise ValueError(
            f"没有产生任何预测。检查 initial_train_days({initial_train_days}) "
            f"是否过大，数据总交易日只有 {n} 天。"
        )

    predictions = pd.concat(preds).sort_index()
    predictions.name = "prediction"
    log = pd.DataFrame(log_rows)
    return predictions, log
