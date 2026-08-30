"""基于技术分析规则的策略.

学术背景：技术交易规则的预测能力是长期争论的话题——
Fama (1970) 的有效市场假说认为历史价格信息无超额收益，
而 Brock, Lakonishok & LeBaron (1992) 与 Lo, Mamaysky & Wang (2000)
在道琼斯与美股百年样本上发现了显著的技术信号收益。

约定：策略在 t 日收盘生成目标仓位；不做 shift，
执行时点统一由回测引擎推迟至 t+1 开盘。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta

__all__ = [
    "ma_cross",
    "macd_cross",
    "rsi_mean_reversion",
    "bollinger_mean_reversion",
]


def _validate(df: pd.DataFrame) -> None:
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise KeyError(f"缺少必需列: {sorted(missing)}")


def ma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    """双均线交叉（金叉持有、死叉离场）.

    ``pos_t = 1[SMA_fast(t) > SMA_slow(t)]``。
    """
    _validate(df)
    pos = (ta.sma(df["close"], fast) > ta.sma(df["close"], slow)).astype("float64")
    return pos.fillna(0.0)


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD 金叉/死叉：DIF > DEA 时持有。"""
    _validate(df)
    m = ta.macd(df["close"], fast, slow, signal)
    pos = (m["dif"] > m["dea"]).astype("float64")
    return pos.where(m["dif"].notna() & m["dea"].notna(), 0.0)


def _state_signal(trigger_enter: pd.Series, trigger_exit: pd.Series) -> pd.Series:
    """由进入/离开触发条件构造 0/1 状态序列（先到先触发，其余延续前值）。"""
    raw = pd.Series(
        np.where(trigger_enter, 1.0, np.where(trigger_exit, 0.0, np.nan)),
        index=trigger_enter.index,
    )
    return raw.ffill().fillna(0.0)


def rsi_mean_reversion(
    df: pd.DataFrame, window: int = 14, enter_below: float = 30.0, exit_above: float = 70.0
) -> pd.Series:
    """RSI 超卖买入、超买卖出（均值回归）.

    ``RSI < 30`` 进入，``RSI > 70`` 离场，区间内维持原状态。
    """
    _validate(df)
    rsi = ta.rsi(df["close"], window)
    return _state_signal(rsi < enter_below, rsi > exit_above)


def bollinger_mean_reversion(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """布林带均值回归：跌破下轨进入，升破上轨离场。"""
    _validate(df)
    boll = ta.bollinger(df["close"], window, num_std)
    return _state_signal(boll["pctb"] < 0.0, boll["pctb"] > 1.0)
