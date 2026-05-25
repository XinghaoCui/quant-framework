"""
数据加载层：统一封装 A 股(akshare) 和 美股(yfinance) 的行情接口。

设计思路：
    上层策略代码只关心"我要某些股票从某天到某天的日线数据"，
    不应该关心数据来自哪个数据源、字段名是什么、要不要复权。
    所以这里把所有数据源的输出统一成一个 DataFrame：
        index: pandas.DatetimeIndex（交易日）
        columns: 标的代码（如 "AAPL"、"600519.SH"）
        values: 后复权收盘价（float）

    为什么用"宽表"（columns 是标的）而不是"长表"（columns 是 ['symbol','date','close']）？
        因为做回测的核心操作是"对每个标的同时计算指标"，
        宽表配合 pandas 的向量化操作可以用一行代码完成几百只股票的计算，
        而长表需要 groupby，慢且代码繁琐。

数据缓存：
    第一次拉取的数据存到本地 parquet 文件，之后从本地读，避免反复网络请求。
    缓存策略：以 (数据源, 标的, 起止日期, 复权方式) 为 key。
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd


# 全局缓存目录，放在项目根的 data_cache/ 下，.gitignore 已排除
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / "data_cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(source: str, symbols: tuple[str, ...], start: str, end: str, adjust: str) -> Path:
    """根据请求参数生成缓存文件路径。用 md5 让长参数也能变成短文件名。"""
    raw = f"{source}|{','.join(sorted(symbols))}|{start}|{end}|{adjust}"
    h = hashlib.md5(raw.encode()).hexdigest()[:16]
    # 用 pickle 不用 parquet：parquet 需要 pyarrow 这个 80MB 的额外依赖，
    # 对几只票几年日线数据来说 pickle 已经够用，零依赖。
    return _CACHE_DIR / f"{source}_{h}.pkl"


def _load_us_yfinance(symbols: list[str], start: str, end: str, adjust: str) -> pd.DataFrame:
    """从 yfinance 拉美股日线。adjust='post' 时返回后复权价。"""
    import yfinance as yf

    # yfinance 的 auto_adjust=True 等价于后复权（用 Adj Close 替换 Close）
    auto_adj = adjust == "post"
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=auto_adj,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError(f"yfinance 没拉到数据，symbols={symbols} start={start} end={end}")

    # 多标的时 yfinance 返回 MultiIndex 列 (ticker, field)；单标的时返回单层列
    if len(symbols) == 1:
        out = raw[["Close"]].rename(columns={"Close": symbols[0]})
    else:
        # 提取每只票的 Close 列
        out = pd.DataFrame({sym: raw[sym]["Close"] for sym in symbols if sym in raw.columns.get_level_values(0)})

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    return out.dropna(how="all")


def _load_cn_akshare(symbols: list[str], start: str, end: str, adjust: str) -> pd.DataFrame:
    """从 akshare 拉 A 股日线。adjust='post' 对应 'hfq'（后复权），'none' 对应 ''。"""
    import akshare as ak

    adj_map = {"post": "hfq", "pre": "qfq", "none": ""}
    ak_adj = adj_map.get(adjust, "hfq")

    # akshare 接受的日期格式是 YYYYMMDD（不带横线）
    start_fmt = start.replace("-", "")
    end_fmt = end.replace("-", "")

    frames = []
    for sym in symbols:
        # akshare 用的代码是 6 位纯数字（如 "600519"），不带后缀
        # 如果用户传的是 "600519.SH" 这种 Wind 风格，剥掉后缀
        code = sym.split(".")[0]
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=ak_adj,
            )
        except Exception as e:
            print(f"[akshare] 拉取 {sym} 失败：{e}")
            continue

        if df is None or df.empty:
            continue

        # akshare 返回中文列名，统一改成英文方便处理
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].rename(sym)
        frames.append(s)

    if not frames:
        raise ValueError(f"akshare 没拉到任何数据，symbols={symbols}")

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    return out


class DataLoader:
    """
    统一的数据加载器。

    用法:
        loader = DataLoader()
        prices = loader.load(["AAPL", "MSFT"], "2020-01-01", "2024-12-31", market="us")
        prices = loader.load(["600519", "000001"], "2020-01-01", "2024-12-31", market="cn")
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache

    def load(
        self,
        symbols: list[str] | str,
        start: str,
        end: str,
        market: Literal["us", "cn"] = "us",
        adjust: Literal["post", "pre", "none"] = "post",
    ) -> pd.DataFrame:
        """
        加载多个标的的收盘价。

        参数:
            symbols: 标的代码列表，或单个字符串
            start, end: 起止日期，格式 "YYYY-MM-DD"
            market: "us" 美股 / "cn" A股
            adjust: "post" 后复权 / "pre" 前复权 / "none" 不复权

        返回:
            DataFrame，index=date, columns=symbols, values=close price
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = list(symbols)

        cache_path = _cache_key(market, tuple(symbols), start, end, adjust)
        if self.use_cache and cache_path.exists():
            return pd.read_pickle(cache_path)

        if market == "us":
            df = _load_us_yfinance(symbols, start, end, adjust)
        elif market == "cn":
            df = _load_cn_akshare(symbols, start, end, adjust)
        else:
            raise ValueError(f"未知市场: {market}")

        if self.use_cache:
            df.to_pickle(cache_path)

        return df


def load_prices(
    symbols: list[str] | str,
    start: str,
    end: str,
    market: Literal["us", "cn"] = "us",
) -> pd.DataFrame:
    """便捷函数：一次性调用，不显式创建 loader。"""
    return DataLoader().load(symbols, start, end, market=market)


def generate_synthetic_prices(
    symbols: list[str],
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
) -> pd.DataFrame:
    """
    生成几何布朗运动模拟价格，用于无网环境验证 pipeline。

    模型：dS/S = μ dt + σ dW
    离散化：S_t = S_{t-1} * exp((μ - σ²/2) dt + σ √dt * Z)

    每只股票独立模拟，drift/vol 在标的间略有差异，
    保证横截面策略（动量/反转）也有差异化信号。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)  # 工作日（约等于交易日）
    n = len(dates)
    dt = 1 / 252

    data = {}
    for i, sym in enumerate(symbols):
        # 让每只股票的 drift 和 vol 略有差异
        mu = annual_drift + (i - len(symbols) / 2) * 0.03
        sigma = annual_vol + (i % 3) * 0.05
        z = rng.standard_normal(n)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        prices = 100 * np.exp(np.cumsum(log_returns))
        data[sym] = prices

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df
