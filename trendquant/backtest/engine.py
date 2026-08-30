"""无前视偏差的向量化回测引擎.

执行假设（与 ``docs/methodology.md`` 第 6 节一致）
--------------------------------------------------
1. 策略信号在 t 日**收盘后**生成；
2. 于 t+1 日**开盘价**成交，当日开盘→收盘的损益计入策略；
3. 仅做多、仓位 0/1，不使用杠杆与做空（A 股现货约束）；
4. 成本按成交额比例计提：佣金（双边）+ 印花税（仅卖出）+ 滑点（双边）。

持仓状态转移对应的单日毛收益：

===========  ====================  ===========================
状态转移     含义                  单日毛收益
===========  ====================  ===========================
0 -> 1       开盘建仓              close / open - 1
1 -> 1       继续持有              close / close[-1] - 1
1 -> 0       开盘平仓              open / close[-1] - 1
0 -> 0       空仓                  0
===========  ====================  ===========================

由于 t 日仓位 ``pos[t]`` 只由 ``signal[t-1]`` 决定，t-1 日收盘后即已知，
因此对 t 日收益不存在前视偏差；引擎同时保留毛收益序列以便单独
评估交易成本的影响。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["CostModel", "BacktestResult", "backtest", "buy_and_hold"]


@dataclass(frozen=True)
class CostModel:
    """A 股现货交易成本模型（比例费率）.

    Attributes
    ----------
    commission: 佣金，双边收取（默认万 2.5）。
    stamp_duty: 印花税，仅卖出收取（2023-08 起为 0.05%）。
    slippage:   滑点，双边收取（默认 0.1%，近似冲击成本）。
    """

    commission: float = 2.5e-4
    stamp_duty: float = 5.0e-4
    slippage: float = 1.0e-3

    @property
    def buy_cost(self) -> float:
        return self.commission + self.slippage

    @property
    def sell_cost(self) -> float:
        return self.commission + self.stamp_duty + self.slippage


@dataclass
class BacktestResult:
    """单标的 × 单策略的回测产出."""

    equity: pd.Series          # 净值曲线（初始为 1）
    net_returns: pd.Series     # 扣费后日收益
    gross_returns: pd.Series   # 未扣费日收益
    position: pd.Series        # 实际持仓（执行后的 0/1 序列）
    n_trades: int              # 建仓次数（回合数）


def backtest(
    df: pd.DataFrame, signal: pd.Series, cost: CostModel | None = None
) -> BacktestResult:
    """对单一标的执行向量化回测.

    Parameters
    ----------
    df:
        含 ``open/close`` 列的日线数据（ DatetimeIndex，升序）。
    signal:
        t 日收盘生成的目标仓位（0/1），索引与 ``df`` 对齐。
        引擎内部统一 shift 一期以实现 t+1 开盘执行。
    cost:
        成本模型，``None`` 时使用 :class:`CostModel` 默认值。
    """
    if "open" not in df.columns or "close" not in df.columns:
        raise KeyError("回测数据需包含 open/close 列")

    cost = cost or CostModel()
    target = signal.reindex(df.index).fillna(0.0).clip(0.0, 1.0)
    pos = target.shift(1).fillna(0.0)  # t+1 开盘执行
    prev_pos = pos.shift(1).fillna(0.0)

    o, c = df["open"], df["close"]
    prev_c = c.shift(1)

    entry = (pos > 0) & (prev_pos == 0)
    hold = (pos > 0) & (prev_pos > 0)
    exit_ = (pos == 0) & (prev_pos > 0)

    gross = pd.Series(0.0, index=df.index)
    gross.loc[entry] = (c / o - 1.0).loc[entry]
    gross.loc[hold] = (c / prev_c - 1.0).loc[hold]
    gross.loc[exit_] = (o / prev_c - 1.0).loc[exit_]

    net = gross - entry.astype(float) * cost.buy_cost - exit_.astype(float) * cost.sell_cost
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        equity=equity,
        net_returns=net,
        gross_returns=gross,
        position=pos,
        n_trades=int(entry.sum()),
    )


def buy_and_hold(df: pd.DataFrame, cost: CostModel | None = None) -> BacktestResult:
    """买入并持有基准：首日开盘买入后不再调仓（含一次建仓成本）。"""
    return backtest(df, pd.Series(1.0, index=df.index), cost=cost)
