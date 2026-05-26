# 量化策略学习框架 — 项目上下文（给 Claude）

> 崔兴昊的量化学习 + 求职展示项目。本文件让 Claude 在新会话里快速进入状态，无需通读全部代码。

## 项目定位

- 多市场（A股 + 美股）量化策略框架，定位"学习 + 求职作品"，与 OptionPricer 并列为求职核心项目。
- 用户是量化金融专业。代码注释、文档、commit message **一律中文**，重在讲清"为什么这么做"，便于学习。

## 工作约定（务必遵守）

1. **中文优先**：注释、文档、提交信息都用中文。
2. **思考留痕**：设计思路 / 学习要点写进 `思考与学习/` 对应章节，便于用户日后阅读。
3. **策略必带回测**：任何新策略都要跑通完整回测并给出各项指标（夏普/回撤/IC 等）。
4. **新增策略**：继承 `src/strategies/base.py` 的 `Strategy`，实现 `generate_weights`。
5. **新增因子**：放 `src/factors/`，约定"值越大越看多"。
6. **防未来函数**：信号由回测引擎统一 `shift(1)`，策略内不要自己 shift。
7. **ML 防数据泄露**：时序划分 + embargo 隔离带，**绝不随机划分**。

## 结构速览

- `src/data/` 数据层（akshare A股 / yfinance 美股，统一宽表接口 + pickle 缓存 + `generate_synthetic_prices` 模拟数据 + `fundamentals.py` 基本面数据 + 防未来函数对齐）
- `src/engine/` 向量化回测引擎 + 绩效指标
- `src/strategies/` 双均线 / 横截面动量 / 价值因子 / 多因子合成（equal & IC 加权） / OptimizedStrategy（接组合优化）
- `src/factors/` 技术因子库 + 因子评估工具（IC/IR/分组回测/IC衰减/综合报告） + 基本面因子库（价值/质量/成长/安全）
- `src/ml/` 机器学习（特征工程 / 防泄露时序划分 / 模型封装 / IC评估 / ML策略 / walk-forward 滚动重训）
- `src/portfolio/` 组合优化（min_var / risk_parity / inverse_vol / mean_variance / max_sharpe / 有效前沿），全部 scipy.optimize，零额外依赖
- `examples/` 示例（00 模拟数据全策略；01-03 真实数据策略；10-13 机器学习；20-23 因子研究；30-32 组合优化；40-43 基本面）
- `tests/` 单元测试（test_metrics / test_ml / test_walkforward / test_factor_evaluation / test_portfolio / test_fundamentals）
- `思考与学习/` 10 章中文学习文档（00 写给零基础的你 / 01架构 / 02术语 / 03因子策略 / 04指标 / 05扩展 / 06机器学习 / 07因子研究方法论 / 08组合优化 / 09基本面数据）

## 环境与运行

- Python 3.14、pandas 3.0、numpy 2.4、scikit-learn 1.8。
- 无网 / yfinance 限流 / Clash 拦截 akshare 时，用 `generate_synthetic_prices` 跑通验证。
- 跑验证：`PYTHONIOENCODING=utf-8 python examples/00_模拟数据_全策略对比.py`
- 跑测试：`python tests/test_metrics.py && python tests/test_ml.py && python tests/test_walkforward.py && python tests/test_factor_evaluation.py && python tests/test_portfolio.py && python tests/test_fundamentals.py`

## Git / 推送

- 远程：https://github.com/XinghaoCui/quant-framework （私有）。
- 命令行 git 默认不走 Clash 代理，直连 github 会超时。受阻时走 Clash 端口（上次为 **7897**）：
  `git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push`
  （端口可能变，先 `curl -x http://127.0.0.1:端口 https://github.com` 探测）

## 下一步候选扩展（按价值排序）

1. **行业归类 + 行业中性化**：拉 akshare `stock_sector_spot`，在因子排序时按行业内中性，消除行业 beta。这是真实研究的标配。
2. **风险归因 / Brinson 分解**：把组合收益拆成"市场 / 风格 / 选股" 三部分。配 10 章。
3. **协方差 shrinkage（Ledoit-Wolf）**：在 `src/portfolio/optimizer.py` 加 shrinkage 包装，08 章 / 05 已铺垫。
4. **滚动调参**：walk-forward 每轮在训练集内用 `purged_kfold` 选超参（文档 06/08 已铺垫）。
5. **LightGBM/XGBoost**：在 `src/ml/models.py:make_model` 加分支。
6. **Black-Litterman 模型**：把 ML 预测作为"观点"接入均值方差，缓解 μ 估不准的问题。
7. **基本面因子升级**：业绩预告日对齐、TTM / 扣非、Altman Z-Score 等"反操纵"评分。

## 已完成扩展（按时间倒序）

- **09 基本面数据**（2026-05-26）：`src/data/fundamentals.py`（akshare + 模拟兜底 + 防未来函数对齐） + `src/factors/fundamental.py`（价值/质量/成长/安全 4 类 8 个因子） + `examples/40~43` + `tests/test_fundamentals.py`（14 个测试，含 3 个核心防未来函数）+ `思考与学习/09_基本面数据/` 7 篇。
- **08 组合优化**（2026-05-26）：`src/portfolio/optimizer.py`（6 种优化 + 有效前沿） + `src/strategies/optimized.py` + `examples/30~32` + `思考与学习/08_组合优化/` 7 篇。坚持 scipy 零额外依赖。
- **00 写给零基础的你**（2026-05-26）：`思考与学习/00_写给零基础的你/` 8 篇零基础友好文档（量化是什么/环境/第一个回测/报告解读/阅读地图/FAQ/Python 速通）+ README 顶层入口前置。
- **07 因子研究方法论**（2026-05-26）：`src/factors/evaluation.py` + `src/strategies/multi_factor.py` + `examples/20~23` + `思考与学习/07_因子研究方法论/` 7 篇。
- **06 机器学习入门**（含 08 滚动重训）：`src/ml/` 全模块 + 9 篇 ML 文档。
