"""技术指标计算模块.

全部指标使用 pandas/numpy 向量化实现，不依赖 TA-Lib 等外部二进制库。
除特别说明外，指标在窗口未满时返回 NaN（显式 warm-up 期），
保证任何时刻读取的指标值只依赖于该时刻及之前的数据（无前视偏差）。

公式定义与参数依据见 ``docs/methodology.md`` 第 4 节。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "macd",
    "rsi",
    "bollinger",
    "atr",
    "kdj",
    "roc",
]


def sma(close: pd.Series, window: int) -> pd.Series:
    """简单移动平均（Simple Moving Average）。"""
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    """指数移动平均（Exponential Moving Average）。

    使用 ``adjust=False`` 的递归形式（与国内行情软件一致），
    以首值为种子，warm-up 期内返回 NaN。
    """
    out = close.ewm(span=window, adjust=False).mean()
    return out.where(np.arange(len(close)) >= window - 1)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标。

    Returns
    -------
    pd.DataFrame
        列：``dif``（快慢 EMA 之差）、``dea``（dif 的 signal 日 EMA）、
        ``hist``（柱状线，国内软件惯例取 ``2*(dif-dea)``）。
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2.0 * (dif - dea)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder (1978) 相对强弱指标。

    使用 ``ewm(alpha=1/window, adjust=False)`` 实现 Wilder 平滑；
    区间内无下跌（avg_loss=0）时置为 100。
    """
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = -diff.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    no_loss = avg_loss.notna() & (avg_loss == 0.0)
    out.loc[no_loss] = 100.0
    return out


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """布林带（Bollinger Bands）。

    Returns
    -------
    pd.DataFrame
        列：``mid``（中轨，SMA）、``upper``/``lower``（上下轨）、
        ``pctb``（%B，价格在带内的相对位置）、``bandwidth``（带宽）。
    """
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    pctb = (close - lower) / width.replace(0.0, np.nan)
    bandwidth = width / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pctb": pctb, "bandwidth": bandwidth}
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均真实波幅（Average True Range，Wilder 平滑）。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 9,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """KDJ 随机指标（国内软件口径）。

    RSV 经 ``ewm(alpha=1/3, adjust=False)`` 递归平滑得到 K、D，
    ``J = 3K - 2D``。
    """
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    rsv = (close - lowest) / (highest - lowest).replace(0.0, np.nan) * 100.0
    k = rsv.ewm(alpha=1.0 / smooth_k, adjust=False).mean()
    d = k.ewm(alpha=1.0 / smooth_d, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def roc(close: pd.Series, window: int) -> pd.Series:
    """n 日变动率（Rate of Change），即 n 日动量收益。"""
    return close / close.shift(window) - 1.0
