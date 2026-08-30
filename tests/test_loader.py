"""数据层测试：确定性、schema、缓存."""

import numpy as np
import pandas as pd
import pytest

from trendquant.data.loader import fetch_ohlcv, synthetic_ohlcv


def test_synthetic_deterministic_same_seed():
    a = synthetic_ohlcv("AAA", "2022-01-01", "2022-12-31", seed=3)
    b = synthetic_ohlcv("AAA", "2022-01-01", "2022-12-31", seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_symbol_derived_seed_differs():
    a = synthetic_ohlcv("AAA", "2022-01-01", "2022-12-31")
    b = synthetic_ohlcv("BBB", "2022-01-01", "2022-12-31")
    assert not np.allclose(a["close"].values, b["close"].values)


def test_schema_integrity():
    df = synthetic_ohlcv("CCC", "2022-01-01", "2022-06-30", seed=5)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index.name == "date"
    assert df[["open", "high", "low", "close"]].notna().all().all()
    assert (df["close"] > 0).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] > 0).all()


def test_fetch_synthetic_length():
    df = fetch_ohlcv("DDD", "2022-01-01", "2022-03-31", source="synthetic")
    assert 55 <= len(df) <= 70  # 约 3 个月的工作日


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="未知数据源"):
        fetch_ohlcv("XXX", "2022-01-01", "2022-02-01", source="bogus")
