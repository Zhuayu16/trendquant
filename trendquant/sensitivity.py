"""参数稳健性分析.

回应"策略参数是否为样本内挑出来的"（data-snooping）质疑：
对双均线 (fast, slow) 做网格扫描，在组合层计算每个参数组合的
夏普比率。若默认参数只是网格中的孤立峰值，则结果很可能过拟合；
若网格整体平稳、默认参数处于平滑平台上，则结论对参数选择不敏感。

方法论背景见 docs/methodology.md §9（数据窥探偏差）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .backtest.engine import CostModel, backtest
from .backtest.metrics import TRADING_DAYS
from .models import rules

__all__ = ["ma_grid_sharpe", "DEFAULT_FAST_RANGE", "DEFAULT_SLOW_RANGE"]

logger = logging.getLogger(__name__)

DEFAULT_FAST_RANGE = (5, 10, 15, 20, 30)
DEFAULT_SLOW_RANGE = (20, 40, 60, 90, 120, 150)


def ma_grid_sharpe(
    data: dict[str, pd.DataFrame],
    cost: CostModel | None = None,
    fasts: tuple[int, ...] = DEFAULT_FAST_RANGE,
    slows: tuple[int, ...] = DEFAULT_SLOW_RANGE,
) -> pd.DataFrame:
    """双均线 (fast, slow) 网格的组合层夏普矩阵.

    每个参数组合下，对各标的分别回测后做等权合并（与主实验同一
    组合口径），计算夏普比率。``fast >= slow`` 的组合为 NaN。
    """
    cost = cost or CostModel()
    matrix = pd.DataFrame(np.nan, index=list(fasts), columns=list(slows), dtype=float)
    matrix.index.name = "fast"
    matrix.columns.name = "slow"

    for fast in fasts:
        for slow in slows:
            if fast >= slow:
                continue
            returns: dict[str, pd.Series] = {}
            for sym, df in data.items():
                bt = backtest(df, rules.ma_cross(df, fast=fast, slow=slow), cost=cost)
                returns[sym] = bt.net_returns
            if not returns:
                raise ValueError("data 不能为空")
            # 与主实验相同：按日期对齐，并对当日有数据的标的等权。
            pooled = pd.DataFrame(returns).mean(axis=1).dropna()
            sd = pooled.std(ddof=1)
            matrix.loc[fast, slow] = (
                pooled.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
            )
    logger.info("参数网格完成：%d 个有效组合", int(matrix.notna().sum().sum()))
    return matrix
