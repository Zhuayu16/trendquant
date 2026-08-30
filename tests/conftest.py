"""pytest 共享 fixture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from trendquant.data.loader import synthetic_ohlcv


@pytest.fixture(scope="session")
def ohlcv():
    """3 年确定性合成日线（~780 根），供多数模块测试复用."""
    return synthetic_ohlcv("TESTER", start="2021-01-01", end="2023-12-31", seed=7)
