"""回测引擎正确性测试：无前视偏差、执行时点、成本计提."""

import numpy as np
import pandas as pd
import pytest

from trendquant.backtest.engine import CostModel, backtest, buy_and_hold


def make_df(closes, opens=None):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    o = pd.Series(opens if opens is not None else closes, index=idx, dtype=float)
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "open": o, "close": c,
        "high": np.maximum(o, c), "low": np.minimum(o, c), "volume": 1.0,
    })


def test_flat_signal_no_pnl():
    df = make_df([10, 10, 10, 10])
    r = backtest(df, pd.Series(0.0, index=df.index))
    assert (r.equity == 1.0).all()
    assert r.n_trades == 0


def test_no_lookahead_position_starts_next_day():
    df = make_df([10, 12, 14], [10, 11, 13])
    r = backtest(df, pd.Series(1.0, index=df.index))
    assert r.position.iloc[0] == 0.0
    assert r.position.iloc[1] == 1.0                    # 信号日次日才有仓位
    assert r.gross_returns.iloc[0] == 0.0               # 信号当日不计收益
    assert r.gross_returns.iloc[1] == pytest.approx(12 / 11 - 1)  # 建仓日 = close/open-1


def test_exit_day_uses_open():
    # 信号第 1 天翻空 → 第 2 天开盘平仓（执行约定：信号次日开盘成交）
    df = make_df([10, 10, 10], [10, 10, 20])  # 平仓日开盘跳空至 20
    sig = pd.Series([1.0, 0.0, 0.0], index=df.index)
    r = backtest(df, sig)
    assert r.position.iloc[1] == 1.0
    assert r.position.iloc[2] == 0.0
    assert r.gross_returns.iloc[2] == pytest.approx(20 / 10 - 1)  # open/prev_close-1


def test_costs_applied_on_entry_and_exit():
    df = make_df([10, 10, 10])
    cost = CostModel(commission=0.001, stamp_duty=0.001, slippage=0.0)
    sig = pd.Series([1.0, 0.0, 0.0], index=df.index)  # 第1天开盘建仓、第2天开盘平仓
    r = backtest(df, sig, cost=cost)
    expected = (1 - cost.buy_cost) * (1 - cost.sell_cost)  # 无价格变动，仅成本
    assert r.equity.iloc[-1] == pytest.approx(expected)


def test_buy_and_hold_zero_cost_tracks_price():
    closes = list(np.linspace(10, 15, 30))
    opens = [closes[0]] + closes[:-1]  # open = 前收
    df = make_df(closes, opens)
    r = buy_and_hold(df, cost=CostModel(commission=0, stamp_duty=0, slippage=0))
    assert r.equity.iloc[-1] == pytest.approx(closes[-1] / opens[1])


def test_cost_model_defaults_positive():
    c = CostModel()
    assert c.sell_cost > c.buy_cost > 0  # 卖出含印花税


def test_backtest_missing_columns_raises():
    with pytest.raises(KeyError):
        backtest(pd.DataFrame({"close": [1.0, 2.0]}), pd.Series([0.0, 0.0]))


def test_trade_round_trip_composition():
    """持有两天的净值 = 建仓日毛收益 × 持有日毛收益，再扣双边成本。"""
    df = make_df([10, 11, 12, 12], [10, 10, 11, 12])
    sig = pd.Series([1.0, 1.0, 1.0, 0.0], index=df.index)
    cost = CostModel(commission=0.0, stamp_duty=0.0, slippage=0.0)
    r = backtest(df, sig, cost=cost)
    expected = (1 + (11 / 10 - 1)) * (1 + (12 / 11 - 1)) * (1 + (12 / 12 - 1))
    assert r.equity.iloc[-1] == pytest.approx(expected)
