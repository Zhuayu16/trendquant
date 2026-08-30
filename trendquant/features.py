"""机器学习特征工程.

所有特征仅使用 t 日收盘及之前的信息构造，标签（label）使用
t+1 至 t+horizon 的未来收益方向——标签只用于训练，绝不进入特征。
特征定义见表 ``docs/methodology.md`` 第 9 节。
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ta

__all__ = ["build_features", "build_dataset"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """由 OHLCV 日线构造特征矩阵（索引与 ``df`` 对齐）。

    Parameters
    ----------
    df:
        含 ``open/high/low/close/volume`` 列的日线数据。
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    ret1 = close.pct_change()

    feats = pd.DataFrame(index=df.index)
    # ---- 价格动量类 ----
    for lag in (1, 5, 10, 20, 60):
        feats[f"ret_{lag}"] = close / close.shift(lag) - 1.0
    feats["mom_120"] = ta.roc(close, 120)
    # ---- 技术指标类（价格归一化，保证横截面/时间序列可比）----
    feats["rsi_14"] = ta.rsi(close, 14) / 100.0
    macd = ta.macd(close)
    feats["macd_norm"] = macd["hist"] / close
    boll = ta.bollinger(close, 20, 2.0)
    feats["pctb_20"] = boll["pctb"]
    feats["bbw_20"] = boll["bandwidth"]
    feats["atr_norm"] = ta.atr(high, low, close, 14) / close
    feats["ma20_dev"] = close / ta.sma(close, 20) - 1.0
    feats["ma60_dev"] = close / ta.sma(close, 60) - 1.0
    # ---- 波动率与量能类 ----
    feats["vol_20"] = ret1.rolling(20, min_periods=20).std(ddof=1)
    feats["vol_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
    return feats


def build_dataset(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """构造 (X, y) 监督学习数据集.

    标签 ``y[t] = 1`` 当且仅当 ``close[t+horizon] > close[t]``（未来 h 日上涨）。
    样本区间两端各裁剪一次：头部为指标 warm-up（特征全空），尾部为标签未到期。

    Returns
    -------
    (X, y)
        X 可能含 NaN（个别指标在极端行情下未定义），由
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` 原生处理。
    """
    X = build_features(df).dropna(how="all")
    forward_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = (forward_ret > 0.0).astype("float64")
    y = y.reindex(X.index)
    valid = y.notna()
    return X.loc[valid], y.loc[valid].astype("int8")
