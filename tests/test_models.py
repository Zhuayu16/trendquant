"""策略模型测试：信号形态、状态机优先级、ML walk-forward."""

import numpy as np
import pandas as pd

from trendquant.models import rules
from trendquant.models.ml import MLConfig, ml_direction_signals
from trendquant.models.statistical import tsmom


def test_rule_signals_binary_and_aligned(ohlcv):
    for fn in (rules.ma_cross, rules.macd_cross,
               rules.rsi_mean_reversion, rules.bollinger_mean_reversion):
        sig = fn(ohlcv)
        assert sig.index.equals(ohlcv.index)
        assert sig.isin([0.0, 1.0]).all()
        assert sig.iloc[0] == 0.0  # warm-up 期空仓


def test_tsmom_uptrend():
    idx = pd.bdate_range("2022-01-01", periods=200)
    px = np.linspace(10.0, 30.0, 200)
    df = pd.DataFrame({"close": px, "open": px}, index=idx)
    sig = tsmom(df, lookback=120)
    assert sig.iloc[:120].eq(0.0).all()
    assert sig.iloc[130:].eq(1.0).all()


def test_state_signal_enter_priority():
    enter = pd.Series([True, False, True, False])
    exit_ = pd.Series([False, True, True, False])
    sig = rules._state_signal(enter, exit_)
    assert list(sig) == [1.0, 0.0, 1.0, 1.0]  # 同日冲突时进入优先，其余延续


def test_ml_walkforward_out_of_sample(ohlcv):
    cfg = MLConfig(horizon=3, n_splits=2, threshold=0.55)
    res = ml_direction_signals(ohlcv, cfg)
    assert res.signal.index.equals(ohlcv.index)
    assert res.signal.isin([0.0, 1.0]).all()
    assert res.oof_prob.notna().any()
    assert res.oof_prob.dropna().between(0.0, 1.0).all()
    assert 0.0 <= res.hit_rate <= 1.0
    assert res.n_oof > 0 and res.k_correct <= res.n_oof


def test_ml_features_no_lookahead(ohlcv):
    """截断历史后重算特征，历史段取值必须与全样本计算一致（无前视）."""
    from trendquant.features import build_features

    full = build_features(ohlcv)
    cut = ohlcv.index[400]
    partial = build_features(ohlcv.loc[:cut])
    cmp_cols = ["ret_1", "rsi_14", "mom_120"]
    pd.testing.assert_frame_equal(
        full.loc[:cut, cmp_cols], partial.loc[:cut, cmp_cols], check_freq=False
    )
