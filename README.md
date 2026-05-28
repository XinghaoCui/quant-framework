# 量化策略学习框架

> 一个面向量化金融学习者的多市场（A 股 + 美股）策略回测框架，包含数据层、向量化回测引擎、因子库、经典策略，以及一套中文学习文档。

## 从零开始

没有编程或金融基础？打开 [思考与学习/00_写给零基础的你/](./思考与学习/00_写给零基础的你/00_读我.md) 这一章。8 篇文档从 0 讲到“跑通一次回测、看懂报告”，顺利的话 30 分钟就能出第一张净值曲线。

已经会 Python 和 pandas？直接看下面的“快速开始”，三条命令跑通。

## 目录结构

```
量化/
├── src/                      # 核心代码
│   ├── data/                 # 数据层：A 股(akshare) + 美股(yfinance) + 基本面财报 + 防未来函数对齐
│   ├── engine/               # 向量化回测引擎 + 绩效指标
│   ├── strategies/           # 经典策略：双均线、动量、价值因子、多因子合成、组合优化策略
│   ├── factors/              # 因子库：技术 + 因子评估 (IC/分组/衰减) + 基本面 (价值/质量/成长/安全)
│   ├── ml/                   # 机器学习：特征工程/时序划分/模型/评估/ML策略
│   ├── portfolio/            # 组合优化：min_var / risk_parity / mean_var / 有效前沿
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
│   ├── 23_多因子合成策略.py
│   ├── 30_组合优化对比.py          # 组合优化（08 章）
│   ├── 31_有效前沿可视化.py
│   ├── 32_组合优化策略接入回测.py
│   ├── 40_基本面数据获取.py        # 基本面（09 章）
│   ├── 41_价值与质量因子IC.py
│   ├── 42_质量因子分组回测.py
│   └── 43_技术与基本面多因子.py
├── 思考与学习/                # 中文学习文档（00 写给零基础的你 + 9 章主线）
├── reports/                  # 回测输出（净值曲线、回撤、指标 csv）
├── data_cache/               # 行情本地缓存（gitignore）
├── requirements.txt
├── AUTHOR.md                 # 关于作者
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
| 组合优化策略（min_var / risk_parity / inv_vol） | 组合 | `src/strategies/optimized.py` |

## 已实现因子

- 技术因子 (`src/factors/technical.py`)：SMA、EMA、RSI、ATR、布林带、动量、波动率、反转
- 因子评估 (`src/factors/evaluation.py`)：IC / Rank IC / IR / t 统计 / 分组回测 / IC 衰减 / 综合报告
- 基本面因子 (`src/factors/fundamental.py`)：EP / BP 价值, ROE / 毛利率 / 净利率 质量, 营收/净利增速 成长, 低杠杆 安全
- 基本面数据层 (`src/data/fundamentals.py`)：akshare 真实接口 + 模拟兜底 + 防未来函数对齐 (报告期 + 45 天 → 披露日)

## 已实现组合优化方法

`src/portfolio/optimizer.py` 全部基于 scipy.optimize（不引入新依赖）:
- 等权 / 反波动率 / 风险平价 / 最小方差 / 均值-方差 / 最大夏普 / 有效前沿

## 绩效指标

回测引擎自动输出：
- 累计收益、年化收益、年化波动率
- 夏普比率、Sortino 比率、Calmar 比率
- 最大回撤（Max Drawdown）
- 胜率、盈亏比
- 换手率、总交易成本

## 学习文档

打开 `思考与学习/` 目录，按编号顺序阅读：

0. 00_写给零基础的你：零基础起步、跑通第一个回测、看懂报告、阅读地图、FAQ
1. 01_架构与设计：为什么这样分层、向量化回测原理
2. 02_核心术语：量化金融术语表、收益率定义
3. 03_因子与策略：因子是什么、每个策略详解
4. 04_回测指标解读：夏普/回撤等指标的数学含义和经验判读
5. 05_扩展指南：如何新增策略、新增因子、数据源排查
6. 06_机器学习入门：ML 在量化的定位、监督学习、特征工程、数据泄露与过拟合、线性/树模型、IC 评估、端到端工作流、滚动重训
7. 07_因子研究方法论：因子研究四问 (IC / IR / 分组 / 衰减)、多因子合成、端到端研究流程
8. 08_组合优化：为什么不等权、协方差矩阵、最小方差、风险平价、Markowitz 与有效前沿、约束与稳健性、端到端接入
9. 09_基本面数据：为什么要基本面、三大报表速读、防未来函数、价值/质量/成长/安全 4 类因子、技术+基本面整合

## 机器学习模块

`src/ml/` 把机器学习接入策略框架，配套 9 篇中文文档（`思考与学习/06_机器学习入门/`）。

```bash
python examples/10_ML_线性模型预测.py    # IC + 分组回测
python examples/11_ML_随机森林选股.py    # 线性 vs 树模型对比
python examples/12_ML_端到端工作流.py    # 训练→预测→选股→回测全流程
python examples/13_ML_滚动重训.py        # walk-forward：训练一次 vs expanding vs rolling
```

贯穿这一章的几条经验：特征工程比模型选择重要得多；时序验证要做严格，否则很容易数据泄露；复杂模型不一定更好，先用简单模型打个基准；滚动重训更接近实盘，样本外的结果也更诚实。

## 因子研究方法论模块

`src/factors/evaluation.py` + `src/strategies/multi_factor.py` 提供一套因子研究工具，配套 7 篇中文文档（`思考与学习/07_因子研究方法论/`）。

```bash
python examples/20_因子IC评估.py        # IC / IR / t 统计判读
python examples/21_分组回测.py          # 分组单调性、多空价差
python examples/22_IC衰减.py            # IC 衰减曲线、调仓周期
python examples/23_多因子合成策略.py     # equal vs IC 加权 vs 单因子 vs 基准
```

几条经验：先看 IC 再看夏普，IC/IR 比单次回测的夏普更可信；分组单调性比 IC 数值更硬；IC 衰减的节奏决定了该多久调一次仓；多因子合成靠的是因子之间相关性低，而不是复杂的权重公式；判断一个新因子值不值，要看它在现有组合里的边际贡献，而不是它单独的 IC。

## 组合优化模块

`src/portfolio/optimizer.py` + `src/strategies/optimized.py` 提供入门级组合优化工具，配套 7 篇中文文档（`思考与学习/08_组合优化/`）。

```bash
python examples/30_组合优化对比.py            # 4 种"只看 Σ"方法权重对比
python examples/31_有效前沿可视化.py          # Markowitz 前沿 + 最小方差/最大夏普点
python examples/32_组合优化策略接入回测.py     # 同样的因子打分 × 5 种分仓方式回测对比
```

几条经验：组合优化是放大器，不是创造者，本身没有 alpha 时它救不了你；最大的命门是期望收益 μ 估不准，直接裸用均值方差很危险；等权其实是个很难打败的基准，复杂方法没有明显优势就别用；协方差比期望收益好估得多，所以 min_var、risk_parity 这类只依赖协方差的方法更工程化。

## 基本面数据模块

`src/data/fundamentals.py` + `src/factors/fundamental.py` 提供基本面数据接入和四类基本面因子，配套 7 篇中文文档（`思考与学习/09_基本面数据/`）。

```bash
python examples/40_基本面数据获取.py        # 防未来函数演示 (报告期 vs 披露日)
python examples/41_价值与质量因子IC.py      # 7 个基本面因子 IC 评估
python examples/42_质量因子分组回测.py      # ROE 5 分组单调性
python examples/43_技术与基本面多因子.py    # 4 个家族对比 (技术 / 基本面 / 等权合成 / IC加权合成)
```

几条经验：报告期不等于披露日，这是基本面回测最容易爆雷的地方；披露日大约是报告期加 45 天，对应 A 股的监管 deadline；价值、质量、成长、安全四类因子互补，QVGS 可以看成巴菲特选股法的量化版；技术因子和基本面因子相关性低，合成起来几乎是免费午餐；ic_weighted 比等权更鲁棒，因为它能自动识别因子方向。

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

## 关于作者

崔兴昊（Cui Xinghao），北京大学光华管理学院金融学在读。

这个仓库是我学习量化的过程，也是给同样在入门的朋友的一份“我自己希望刚开始就有”的教程。欢迎交流学习、提建议、合作或招聘等任何形式的联系。

- 邮箱: [2500015858@stu.pku.edu.cn](mailto:2500015858@stu.pku.edu.cn)
- 手机 / 微信: 18294156737
- GitHub: [@XinghaoCui](https://github.com/XinghaoCui)
