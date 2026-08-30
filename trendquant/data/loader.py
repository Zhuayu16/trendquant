"""数据获取层.

统一输出格式：DataFrame，DatetimeIndex（升序、去重），
列 ``open/high/low/close/volume``，A 股价格为**前复权**。

三种来源
--------
- ``akshare``   A 股日线（东方财富接口，免费，需联网）；
- ``yfinance``  美股及全球市场（自动映射 A 股代码后缀）；
- ``synthetic`` 确定性合成数据（马尔科夫机制切换的几何布朗运动），
  用于单元测试、CI 与无网络环境演示。

本地 CSV 缓存位于 ``cache_dir``，键为 (source, symbol, start, end)。
"""

from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["fetch_ohlcv", "synthetic_ohlcv", "canonical_columns"]

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume"]


def canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """校验并返回规范列序的 OHLCV DataFrame."""
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必需列: {missing}")
    if "volume" not in df.columns:
        df = df.assign(volume=np.nan)
    df = df[CANONICAL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------- 缓存

def _cache_path(cache_dir: Path, source: str, symbol: str, start: str, end: str) -> Path:
    return cache_dir / f"{source}_{symbol}_{start}_{end}.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("命中缓存: %s", path.name)
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


# ---------------------------------------------------------------- akshare

_AKSHARE_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}


def _fetch_akshare(symbol: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """通过 AkShare 获取 A 股前复权日线（东方财富数据源）."""
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 akshare，请执行: pip install akshare （或 pip install -e .[china]）"
        ) from exc

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            if raw is None or raw.empty:
                raise ValueError(f"akshare 返回空数据: {symbol}")
            raw = raw.rename(columns=_AKSHARE_COL_MAP)
            if "date" not in raw.columns:  # 兼容未来英文列名版本
                raw = raw.rename(
                    columns={"date": "date", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"}
                )
            raw["date"] = pd.to_datetime(raw["date"])
            df = raw.set_index("date")
            logger.info("akshare 拉取成功: %s (%d 条)", symbol, len(df))
            return canonical_columns(df)
        except Exception as exc:  # noqa: BLE001 - 网络异常统一重试
            last_err = exc
            logger.warning("akshare 第 %d 次尝试失败(%s): %s", attempt, symbol, exc)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"akshare 拉取 {symbol} 失败: {last_err}") from last_err


# ---------------------------------------------------------------- yfinance

def _yf_ticker(symbol: str) -> str:
    """将 6 位 A 股代码映射为 yfinance 代码；其余原样返回（美股等）."""
    s = symbol.upper()
    if len(s) == 6 and s.isdigit():
        return f"{s}.SS" if s.startswith("6") else f"{s}.SZ"
    return s


def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 yfinance，请执行: pip install yfinance （或 pip install -e .[us]）"
        ) from exc

    ticker = _yf_ticker(symbol)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance 返回空数据: {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index)
    logger.info("yfinance 拉取成功: %s (%d 条)", ticker, len(raw))
    return canonical_columns(raw)


# ---------------------------------------------------------------- 合成数据

def synthetic_ohlcv(
    symbol: str,
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    seed: int | None = None,
) -> pd.DataFrame:
    """确定性合成 OHLCV 序列（两状态马尔科夫机制切换 GBM）.

    由 ``symbol`` 的 CRC32 派生随机种子，保证跨进程可复现，
    用于单元测试、CI 与离线演示（不构成任何真实行情）。
    """
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    index = pd.bdate_range(start, end)
    n = len(index)

    # 机制切换：牛/熊市漂移不同，模拟真实价格路径的趋势段
    mu = np.where(rng.random(n) < 0.5, 6.0e-4, -4.5e-4)  # 每日漂移
    switch = rng.random(n) < 0.015  # ~1.5% 概率切换机制
    for i in range(1, n):
        if switch[i]:
            mu[i] = -mu[i - 1]
        else:
            mu[i] = mu[i - 1]
    sigma = 0.018 + 0.006 * rng.random(n)  # 日波动率 1.8%~2.4%

    ret = mu + sigma * rng.standard_normal(n)
    close = 50.0 * np.exp(np.cumsum(ret))

    gap = 1.0 + 0.003 * rng.standard_normal(n)  # 隔夜跳空
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * gap[1:]

    span_hi = np.abs(rng.standard_normal(n)) * 0.006
    span_lo = np.abs(rng.standard_normal(n)) * 0.006
    high = np.maximum(open_, close) * (1.0 + span_hi)
    low = np.minimum(open_, close) * (1.0 - span_lo)

    volume = np.exp(rng.normal(14.5, 0.35, n)) * (1.0 + 25.0 * np.abs(ret))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    return canonical_columns(df)


# ---------------------------------------------------------------- 对外接口

def fetch_ohlcv(
    symbol: str,
    start: str = "2019-01-01",
    end: str | None = None,
    source: str = "akshare",
    cache_dir: str | Path = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取单只标的的规范化日线数据.

    Parameters
    ----------
    symbol:
        6 位 A 股代码（akshare/yfinance 均可）或任意 yfinance 代码。
    start / end:
        ``YYYY-MM-DD``；``end`` 为 ``None`` 时取今天。
    source:
        ``"akshare" | "yfinance" | "synthetic"``。
    cache_dir:
        本地缓存目录（CSV）。
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cdir = Path(cache_dir)

    if source == "synthetic":
        return synthetic_ohlcv(symbol, start=start, end=end)

    cpath = _cache_path(cdir, source, symbol, start, end)
    if use_cache:
        cached = _read_cache(cpath)
        if cached is not None:
            return canonical_columns(cached)

    if source == "akshare":
        df = _fetch_akshare(symbol, start, end)
    elif source == "yfinance":
        df = _fetch_yfinance(symbol, start, end)
    else:
        raise ValueError(f"未知数据源: {source!r}（可选 akshare/yfinance/synthetic）")

    _write_cache(df, cpath)
    return df
