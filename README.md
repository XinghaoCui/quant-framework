# 量化策略学习框架

> 一个面向量化金融学习者的多市场（A 股 + 美股）策略回测框架，包含数据层、向量化回测引擎、因子库、经典策略，以及详细中文学习文档。

## 目录结构

```
量化/
├── src/                      # 核心代码
│   ├── data/                 # 数据层：A 股(akshare) + 美股(yfinance) 统一接口
│   ├── engine/               # 向量化回测引擎 + 绩效指标
│   ├── strategies/           # 经典策略：双均线、动量、价值因子、多因子合成
│   ├── factors/              # 因子库：技术因子 + 因子评估工具 (IC/分组/衰减)
│   ├── ml/                   # 机器学习：特征工程/时序划分/模型/评估/ML策略
│   └── utils/                # 画图、报告
├── examples/                 # 可直接运行的示例脚本
│   ├── 00_模拟数据_全策略对比.py   # 不需联网，跑通验证
│   ├── 01_双均线_美股.py
│   ├── 02_动量_美股.py
│   ├── 03_价值因子_A股.py
│   ├── 10_ML_线性模型预测.py       # 机器学习入门
│   ├── 11_ML_随机森林选股.py
│   ├── 12_ML_端到端工作流.py
│   ├── 13_ML_滚动重训.py           # walk-forward 滚动重训
│   ├── 20_因子IC评估.py            # 因子研究方法论
│   ├── 21_分组回测.py
│   ├── 22_IC衰减.py
│   └── 23_多因子合成策略.py
├── 思考与学习/                # 详细中文学习文档（含设计思路 + ML + 因子研究方法论）
├── reports/                  # 回测输出（净值曲线、回撤、指标 csv）
├── data_cache/               # 行情本地缓存（gitignore）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 跑通验证（无需联网）

```bash
python examples/00_模拟数据_全策略对比.py
```

输出会在 `reports/` 下生成净值曲线 PNG、回撤曲线、指标 csv。

### 3. 真实数据

```bash
python examples/01_双均线_美股.py
python examples/02_动量_美股.py
python examples/03_价值因子_A股.py
```

> 如果 yfinance 报 `Too Many Requests`，等几分钟再跑（外部 API 限流）。
> 如果 akshare 报代理错误，关闭 Clash/V2Ray 等代理软件（接口是国内的）。

## 已实现策略

| 策略 | 类型 | 文件 |
|---|---|---|
| 双均线（SMA Cross） | 趋势 / 时序 | `src/strategies/sma_cross.py` |
| 横截面动量 | 因子 / 横截面 | `src/strategies/momentum.py` |
| 价值因子（反转代理） | 因子 / 横截面 | `src/strategies/value_factor.py` |
| ML 策略（线性/RF/GBDT） | 机器学习 | `src/ml/ml_strategy.py` |
| 多因子合成（equal / IC 加权） | 因子 / 横截面 | `src/strategies/multi_factor.py` |

## 已实现因子

- 因子计算 (`src/factors/technical.py`)：SMA、EMA、RSI、ATR、布林带、动量、波动率、反转
- 因子评估 (`src/factors/evaluation.py`)：IC / Rank IC / IR / t 统计 / 分组回测 / IC 衰减 / 综合报告

## 绩效指标

回测引擎自动输出：
- 累计收益、年化收益、年化波动率
- **夏普比率（Sharpe）**、**Sortino 比率**、**Calmar 比率**
- **最大回撤（Max Drawdown）**
- 胜率、盈亏比
- 换手率、总交易成本

## 学习文档

打开 `思考与学习/` 目录，按编号顺序阅读：

1. **01_架构与设计** — 为什么这样分层、向量化回测原理
2. **02_核心术语** — 量化金融术语表、收益率定义
3. **03_因子与策略** — 因子是什么、每个策略详解
4. **04_回测指标解读** — 夏普/回撤等指标的数学含义和经验判读
5. **05_扩展指南** — 如何新增策略、新增因子、数据源排查
6. **06_机器学习入门** — ML 在量化的定位 → 监督学习 → 特征工程 → 数据泄露与过拟合 → 线性/树模型 → IC评估 → 端到端工作流 → 滚动重训
7. **07_因子研究方法论** — 因子研究四问 (IC / IR / 分组 / 衰减) → 多因子合成 → 端到端研究流程

## 机器学习模块

`src/ml/` 把机器学习接入策略框架，配套 9 篇中文文档（`思考与学习/06_机器学习入门/`）。

```bash
python examples/10_ML_线性模型预测.py    # IC + 分组回测
python examples/11_ML_随机森林选股.py    # 线性 vs 树模型对比
python examples/12_ML_端到端工作流.py    # 训练→预测→选股→回测全流程
python examples/13_ML_滚动重训.py        # walk-forward：训练一次 vs expanding vs rolling
```

核心理念（贯穿文档）：**特征工程 >> 模型选择**；**严格的时序验证防数据泄露**；**复杂模型不一定更好，先打简单基准**；**滚动重训贴近实盘，样本外更诚实**。

## 因子研究方法论模块

`src/factors/evaluation.py` + `src/strategies/multi_factor.py` 提供工业级因子研究工具，配套 7 篇中文文档（`思考与学习/07_因子研究方法论/`）。

```bash
python examples/20_因子IC评估.py        # IC / IR / t 统计判读
python examples/21_分组回测.py          # 分组单调性、多空价差
python examples/22_IC衰减.py            # IC 衰减曲线、调仓周期
python examples/23_多因子合成策略.py     # equal vs IC 加权 vs 单因子 vs 基准
```

核心理念：**先 IC 再夏普**（IC/IR 比单次回测夏普更可信）；**单调性比 IC 更硬**；**衰减节奏决定调仓频率**；**多因子合成靠"低相关"，不靠"复杂权重"**；**新因子的价值看的是"在现有组合里的边际贡献"，不是单因子 IC**。

## 如何扩展

### 新增一个策略

```python
# src/strategies/my_strategy.py
from .base import Strategy

class MyStrategy(Strategy):
    name = "MyStrategy"

    def generate_weights(self, prices):
        # 返回和 prices 同形状的 DataFrame
        ...
```

然后在 `src/strategies/__init__.py` 导出，写个 `examples/04_my_strategy.py` 跑回测。

详细步骤见 `思考与学习/05_扩展指南/如何新增一个策略.md`。

## 项目目标

- 学习量化研究的标准代码组织
- 理解回测的核心数学（向量化、防未来函数、交易成本）
- 积累可复用的因子和策略代码
- 作为量化求职的项目展示
