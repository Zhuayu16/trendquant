"""技术指标公式正确性测试."""

import numpy as np
import pandas as pd
import pytest

from trendquant import indicators as ta

rng = np.random.default_rng(0)


def _walk(n=300, s0=100.0):
    return pd.Series(s0 + rng.normal(0, 1.0, n).cumsum())


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ta.sma(s, 3)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_recursion_and_warmup():
    s = pd.Series([10.0] * 30)
    out = ta.ema(s, 5)
    assert pd.isna(out.iloc[3])          # warm-up 期
    assert (out.iloc[4:] == 10.0).all()  # 常数序列 EMA 恒等于常数


def test_macd_identity():
    s = _walk(200)
    m = ta.macd(s)
    assert np.allclose(m["hist"], 2.0 * (m["dif"] - m["dea"]))


def test_rsi_extremes_and_bounds():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ta.rsi(up, 14).iloc[-1] == 100.0  # 全程上涨 → RSI=100
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert ta.rsi(down, 14).iloc[-1] == 0.0  # 全程下跌 → RSI=0
    r = ta.rsi(_walk(300), 14).dropna()
    assert ((r >= 0.0) & (r <= 100.0)).all()


def test_bollinger_geometry():
    b = ta.bollinger(_walk(300), 20, 2.0).dropna()
    assert (b["upper"] >= b["mid"]).all()
    assert (b["mid"] >= b["lower"]).all()
    const = ta.bollinger(pd.Series(np.full(30, 5.0)), 20, 2.0)
    assert pd.isna(const["pctb"].iloc[-1])  # 零带宽 → %B 未定义


def test_atr_nonnegative():
    s = _walk(200)
    high, low = s + 0.5, s - 0.5
    a = ta.atr(high, low, s, 14).dropna()
    assert (a >= 0).all()


def test_roc():
    s = pd.Series([100.0, 110.0, 121.0])
    assert ta.roc(s, 2).iloc[2] == pytest.approx(0.21)


def test_kdj_bounds():
    s = _walk(200)
    k = ta.kdj(s + 0.5, s - 0.5, s)
    k = k.dropna()
    assert ((k["k"] <= 105.0) & (k["k"] >= -5.0)).all()
    assert np.allclose(k["j"], 3 * k["k"] - 2 * k["d"])
