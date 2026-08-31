"""参数稳健性计算口径测试。"""

import numpy as np
import pandas as pd
import pytest

from trendquant.backtest.engine import CostModel, backtest
from trendquant.models import rules
from trendquant.sensitivity import ma_grid_sharpe


def _prices() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=180)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.4, len(idx))), index=idx)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1_000.0,
    })


def test_ma_grid_sharpe_matches_annualized_definition():
    df = _prices()
    zero_cost = CostModel(commission=0, stamp_duty=0, slippage=0)
    matrix = ma_grid_sharpe({"AAA": df}, cost=zero_cost, fasts=(5,), slows=(20,))
    ret = backtest(df, rules.ma_cross(df, fast=5, slow=20), cost=zero_cost).net_returns
    expected = ret.mean() / ret.std(ddof=1) * np.sqrt(252)
    assert matrix.loc[5, 20] == pytest.approx(expected)


def test_ma_grid_rejects_empty_data():
    with pytest.raises(ValueError, match="data 不能为空"):
        ma_grid_sharpe({}, fasts=(5,), slows=(20,))
