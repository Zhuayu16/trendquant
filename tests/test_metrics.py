"""绩效指标与统计检验测试."""

import numpy as np
import pandas as pd
import pytest

from trendquant.backtest.engine import BacktestResult
from trendquant.backtest.metrics import (
    adf_test,
    binom_direction_test,
    drawdown_series,
    max_drawdown,
    paired_ttest,
    performance_stats,
    trade_episodes,
    ttest_mean_zero,
)

rng = np.random.default_rng(2)


def test_max_drawdown_known_series():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(eq) == pytest.approx(-0.25)
    assert (drawdown_series(eq) <= 0).all()


def test_performance_stats_fields():
    net = pd.Series(rng.normal(0.0005, 0.01, 300))
    eq = (1 + net).cumprod()
    st = performance_stats(net, eq, n_trades=3)
    for key in ("total_return", "annual_return", "annual_vol", "sharpe",
                "sortino", "max_drawdown", "calmar", "win_rate_daily", "n_days"):
        assert key in st
    assert st["n_days"] == 300
    assert 0.0 <= st["win_rate_daily"] <= 1.0


def test_ttest_detects_positive_mean():
    net = pd.Series(rng.normal(0.001, 0.005, 500))  # t ≈ 4.5
    t, p = ttest_mean_zero(net)
    assert t > 3 and p < 0.001


def test_paired_ttest_identical_series():
    a = pd.Series(rng.normal(0, 0.01, 200))
    _, p = paired_ttest(a, a.copy())
    assert np.isnan(p) or p > 0.99  # 完全相同 → 无差异（scipy 返回 nan）


def test_binom_direction_test():
    hr, p = binom_direction_test(60, 100)
    assert hr == 0.6 and p < 0.05
    _, p2 = binom_direction_test(52, 100)
    assert p2 > 0.05


def test_adf_stationarity():
    walk = pd.Series(rng.normal(0, 1, 600).cumsum())
    rets = walk.diff().dropna()
    assert not adf_test(walk)["stationary"]      # 随机游走：价格不平稳
    assert adf_test(rets)["stationary"]          # 差分后平稳


def test_trade_episodes_splitting():
    idx = pd.bdate_range("2024-01-01", periods=6)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], index=idx)
    gross = pd.Series([0.0, 0.10, -0.05, 0.0, 0.0, 0.20], index=idx)
    bt = BacktestResult(equity=(1 + gross).cumprod(), net_returns=gross,
                        gross_returns=gross, position=pos, n_trades=2)
    ep = trade_episodes(bt)
    assert len(ep) == 2
    assert ep.iloc[0]["days"] == 2
    assert ep.iloc[0]["gross_return"] == pytest.approx(1.10 * 0.95 - 1)
    assert bool(ep.iloc[1]["open_at_end"])       # 期末仍持仓
