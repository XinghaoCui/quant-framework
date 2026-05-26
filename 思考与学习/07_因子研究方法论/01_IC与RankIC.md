# IC 与 Rank IC

IC (Information Coefficient, 信息系数) 是量化研究员说得最多的一个词。
本篇从直觉到数学讲清 IC 是什么、怎么算、怎么用。

## 一、一句话定义

> **IC = 因子值排序 与 未来收益排序 的相关系数。**

直觉上: 如果一个因子值高的股票, 真的未来收益就高 (排序一致), IC 接近 +1;
如果毫无关系, IC 接近 0; 如果反向 (因子值高反而收益低), IC 接近 -1。

## 二、为什么是"截面"而不是"时序"

新手最常见的误区: 把 IC 算成"某只股票的因子时序 vs 收益时序的相关性"。**错。**

量化的 IC 一定是**截面 IC** (Cross-Sectional IC):

```
在某一天 t:
    把所有股票的因子值排一列  [f_1, f_2, ..., f_N]
    把所有股票的未来收益排一列 [r_1, r_2, ..., r_N]
    这两列的相关系数 = 第 t 天的 IC

跨日把每天的 IC 取均值 = IC 均值 (这个因子的整体预测力)
```

为什么必须是截面?
- 量化策略是**选股**: 在每一刻"选哪些股票", 而不是"预测某一只股票什么时候涨"
- 截面 IC 直接对应"按因子选 top N" 的逻辑
- 时序 IC 会被市场整体涨跌污染 (所有票都涨, 单只票的因子时序和收益时序自然正相关)

## 三、Pearson IC vs Rank IC

两种算法, 业界主流是 Rank IC:

### Pearson IC (普通线性相关)
$$
IC_t = \mathrm{Pearson}(f_t, r_t) = \frac{\mathrm{Cov}(f_t, r_t)}{\sigma_{f_t} \sigma_{r_t}}
$$

- 假设因子值和未来收益是**线性关系**
- 对异常值非常敏感: 一只极端股票就能把整天的 IC 拉到 +0.5 或 -0.5
- 因子分布偏态时 (如 PE/PB 这种长尾) 极易失真

### Rank IC (Spearman 秩相关)
$$
IC^{rank}_t = \mathrm{Pearson}(\mathrm{rank}(f_t), \mathrm{rank}(r_t))
$$

- 先把因子值和收益**各自转成排名**, 再算 Pearson
- 只看排序关系, 不在意具体数值
- 抗异常值, 对因子分布形状不敏感
- **业界默认就是 Rank IC**, 论文里不特别说明的 "IC" 通常都指 Rank IC

经验:
> Pearson IC 可能比 Rank IC 高也可能低, 差距 < 0.02 算正常。
> 如果两者差距超过 0.05, 几乎肯定是有异常值在作祟, 应该先做去极值。

## 四、IC 判读经验

A 股 + 美股大盘股票池的经验值 (Rank IC, horizon=5~20 天):

| |IC 均值| | 评级 | 说明 |
|---|---|---|
| < 0.01 | 没用 | 噪声 |
| 0.02 | 勉强可用 | 单独不太行, 可以做合成 |
| 0.05 | 不错 | 主流商业因子的水平 |
| 0.08 | 强 | 顶级量化机构追求的水平 |
| > 0.10 | 警惕 | 大概率有未来函数或过拟合, **先查代码** |

> 看到 IC > 0.15 的第一反应不应该是激动, 而是: "我哪里写错了?"
> 真实金融数据信噪比极低, IC 不会动不动 0.2。

## 五、IC 不是越高越好的全部

IC 均值只是开始, 你还要看:

1. **IR (信息比率)** = IC 均值 / IC 标准差 — IC 稳不稳 (下一篇详讲)
2. **IC>0 占比** — 直观看正向天数比例, 60% 以上算稳
3. **IC 时序图** — 是不是有几个时期突然崩盘 (régime 切换信号)
4. **分组单调性** — IC 是被极端股票拉的还是整体单调 (第 3 篇详讲)

## 六、代码: 三行算 IC

`src/factors/evaluation.py` 把 IC 封装得很轻量:

```python
from src.factors.evaluation import compute_ic, forward_returns, ic_summary

fwd = forward_returns(prices, horizon=5)              # 5 天未来收益
ic = compute_ic(factor, fwd, method="spearman")       # Rank IC 时序
stats = ic_summary(ic)
print(stats)
# {'IC均值': 0.0686, 'IC标准差': 0.1784, 'IR': 0.3847, ...}
```

底层是 `factor.corrwith(forward_ret, axis=1)`, 一行 pandas 算所有日期的截面 IC。
跑一遍 `examples/20_因子IC评估.py` 体会一下。

## 七、常见错误

1. **forward shift 方向反了**: 应该 `prices.shift(-h) / prices - 1` (未来除当前), 反了就成"用未来因子预测过去收益", IC 会异常高
2. **整列 corr 而不是按行**: pandas 默认 `corrwith(axis=0)` 是按列, 得显式写 `axis=1`
3. **没去 NaN**: 因子有 lookback 期开头大量 NaN, 每天 dropna 后才算
4. **不同 horizon 比绝对值**: horizon=20 的 IC 自然比 horizon=1 高 (信号累积), 不要直接比

## 八、小结

- IC = 截面相关系数, 每天一个值, 取均值看整体
- 默认用 Rank IC (Spearman), 抗异常值
- 经验: |IC| > 0.05 不错, > 0.10 要查 bug
- IC 只是单一指标, 还要看 IR、单调性、衰减
- 实现就是 `df.corrwith(other, axis=1)` 一行

掌握了 IC, 你就有了"判断因子值不值得用"的第一把尺子。
