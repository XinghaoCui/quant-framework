"""
示例 40: 基本面数据获取与"防未来函数"演示
==========================================

研究问题:
    1. 怎么拿到一只股票的财报数据 (模拟兜底, 也支持真实 akshare)
    2. 财报是季频, 怎么对齐到日频?
    3. 关键: 怎么防止"用未来还没披露的财报" 这种致命错误?

输出会清晰展示:
    - 一份完整的季频财务报表
    - 同一个 ROE 在不同日期的"可见值": 报告期 2024-03-31 的 ROE
      要等到约 2024-05-15 (披露日近似 = 报告期 + 45 天) 才能用
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import (
    generate_synthetic_prices,
    generate_synthetic_fundamentals,
    get_fundamental_factor,
)


def main():
    symbols = [f"STK{i:02d}" for i in range(5)]
    prices = generate_synthetic_prices(symbols, "2022-01-01", "2024-12-31", seed=42)

    # ===== 步骤 1: 拿原始季频财报 =====
    print("===== 步骤 1: 原始季频财报 (来自模拟器, 跟价格同种子) =====")
    bundles = generate_synthetic_fundamentals(symbols, "2022-01-01", "2024-12-31",
                                              seed=42, prices=prices)
    print(f"\n{bundles['STK00']}")
    print("\nSTK00 完整季频报表:")
    print(bundles["STK00"].quarterly.round(4).to_string())

    # ===== 步骤 2: 对齐到日频 (防未来函数) =====
    print("\n\n===== 步骤 2: ROE 对齐到日频 (防未来函数) =====")
    roe = get_fundamental_factor(
        "roe", symbols, prices.index, use_synthetic=True, prices=prices,
        announce_lag_days=45,  # 报告期 + 45 天 ≈ 披露日
    )
    print(f"\nroe shape: {roe.shape}")
    print(f"roe 头 3 行 (空仓期, 第一份报告还没披露):")
    print(roe.head(3).round(4).to_string())
    print(f"\nroe 末 3 行 (拿到了最新季报):")
    print(roe.tail(3).round(4).to_string())

    # ===== 步骤 3: 重点演示 — 同一个报告期, 跨披露日的可见值变化 =====
    print("\n\n===== 步骤 3: 防未来函数演示 ★ 重点看这里 =====")
    print(f"STK00 的 2024-Q1 报告期 = 2024-03-31")
    print(f"披露日近似 = 2024-03-31 + 45 天 = 2024-05-15")
    print(f"在 2024-05-14 之前, ROE 还应该是 2023-Q4 的值; 5-15 起才更新.\n")

    window = roe["STK00"].loc["2024-05-10":"2024-05-20"]
    for d, v in window.items():
        flag = "  ← 跳到新季报值" if d.strftime("%Y-%m-%d") == "2024-05-15" else ""
        print(f"  {d.strftime('%Y-%m-%d')}: ROE = {v:.4f}{flag}")

    print("\n这就是防未来函数: 如果在 5-10 用了 5-15 才能拿到的 ROE,")
    print("你的回测会比实盘乐观, 实盘上线就翻车.")

    # ===== 步骤 4: 多个因子横截面 =====
    print("\n\n===== 步骤 4: 同一天多只票的横截面 (2024-12-31) =====")
    factors_to_show = ["roe", "net_margin", "revenue_yoy", "profit_yoy", "debt_ratio"]
    snapshot = {}
    for f in factors_to_show:
        snapshot[f] = get_fundamental_factor(
            f, symbols, prices.index, use_synthetic=True, prices=prices,
        ).iloc[-1]
    print(pd.DataFrame(snapshot).round(4).to_string())

    print("\n小白怎么用:")
    print("  1. 把上面这些字段当因子用, 高的 ROE / 高增长 → 看多")
    print("  2. 拉真实数据只需 use_synthetic=False (本机有 akshare 可达)")
    print("  3. 关键: 永远只用 get_fundamental_factor 接口, 别自己拿 quarterly 数据 reindex")


if __name__ == "__main__":
    main()
