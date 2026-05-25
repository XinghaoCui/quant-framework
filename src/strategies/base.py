"""
策略基类。

约定：
    所有策略必须实现 generate_weights(prices) → DataFrame，
    输出和 prices 同样形状的"目标权重"矩阵。
    引擎负责执行（shift、计算收益、计费用），策略只管"想买什么"。

为什么这么设计？
    强制把"产生信号"和"执行交易"分开，这是量化系统的核心解耦：
        - 信号研究员只关心 alpha 来自哪里
        - 执行工程师只关心怎么把信号落地
    两边各自迭代，不互相牵制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """所有策略的基类。"""

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        根据价格数据生成目标权重矩阵。

        参数:
            prices: 价格宽表，index=date, columns=symbol

        返回:
            weights: 同形状 DataFrame，每行是某天的目标权重向量
                     权重和 ≤ 1（不加杠杆），可以小于 1（部分空仓）
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
