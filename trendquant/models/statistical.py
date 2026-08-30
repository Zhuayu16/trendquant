"""统计/因子类基线策略.

与"规则驱动"的技术分析不同，这类信号直接来自已发表的
资产定价实证结论，作为机器学习策略的学术基线（benchmark）。
"""

from __future__ import annotations

import pandas as pd

__all__ = ["tsmom"]


def tsmom(df: pd.DataFrame, lookback: int = 120) -> pd.Series:
    """时间序列动量（Time-Series Momentum）.

    Moskowitz, Ooi & Pedersen (2012)：若标的过去 ``lookback`` 日收益为正
    则持有，否则空仓。warm-up 期（前 lookback 日）为空仓。
    """
    past_ret = df["close"] / df["close"].shift(lookback) - 1.0
    pos = (past_ret > 0.0).astype("float64")
    return pos.fillna(0.0)
