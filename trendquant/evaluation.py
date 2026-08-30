"""实验编排：多标的 × 多策略回测、统计检验与结果汇总.

组合构建口径：将各标的的日净收益做**等权平均**（近似每日再平衡的
等权组合），在组合层面计算绩效与统计检验，缓解单一标的的偶然性。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.engine import BacktestResult, CostModel, backtest
from .backtest.metrics import (
    adf_test,
    binom_direction_test,
    paired_ttest,
    performance_stats,
    trade_episodes,
    trade_stats,
    ttest_mean_zero,
)
from .data.loader import fetch_ohlcv
from .models import rules
from .models.ml import MLConfig, MLResult, ml_direction_signals
from .models.statistical import tsmom
from .universe import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)

__all__ = ["StrategyDef", "ExperimentConfig", "ExperimentResult", "run_experiment"]


@dataclass(frozen=True)
class StrategyDef:
    """策略注册项：中文报告名 + 英文图表名 + 信号函数."""

    key: str
    name_cn: str
    name_en: str
    fn: object  # Callable[[pd.DataFrame], pd.Series]


def _bah_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


STRATEGIES: list[StrategyDef] = [
    StrategyDef("bah", "买入持有（基准）", "Buy & Hold", _bah_signal),
    StrategyDef("ma", "双均线交叉（20/60）", "MA Crossover (20/60)", rules.ma_cross),
    StrategyDef("macd", "MACD（12/26/9）", "MACD (12/26/9)", rules.macd_cross),
    StrategyDef("rsi", "RSI 均值回归（14）", "RSI Mean-Reversion (14)", rules.rsi_mean_reversion),
    StrategyDef(
        "boll", "布林带均值回归（20, 2σ）", "Bollinger Mean-Reversion (20, 2σ)",
        rules.bollinger_mean_reversion,
    ),
    StrategyDef("tsmom", "时序动量（120 日）", "TSMOM (120d)", tsmom),
]
ML_STRATEGY = StrategyDef("ml", "ML 梯度提升（5 日方向）", "ML-GBDT (5d)", None)


@dataclass
class ExperimentConfig:
    """实验配置."""

    symbols: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNIVERSE))
    start: str = "2019-01-01"
    end: str | None = None
    source: str = "akshare"
    cache_dir: str | Path = "data_cache"
    cost: CostModel = field(default_factory=CostModel)
    ml: MLConfig = field(default_factory=MLConfig)
    skip_ml: bool = False

    @property
    def strategy_names_cn(self) -> list[str]:
        names = [s.name_cn for s in STRATEGIES]
        if not self.skip_ml:
            names.append(ML_STRATEGY.name_cn)
        return names


@dataclass
class ExperimentResult:
    """实验全部产出."""

    cfg: ExperimentConfig
    data: dict[str, pd.DataFrame]                       # symbol -> 日线
    per_symbol: dict[str, dict[str, BacktestResult]]    # symbol -> strategy -> 回测
    pooled_returns: OrderedDict[str, pd.Series]       # strategy(cn) -> 等权日净收益
    pooled_equity: OrderedDict[str, pd.Series]
    stats_pooled: pd.DataFrame                          # 组合层绩效 + 检验
    stats_by_symbol: pd.DataFrame                       # 分标的明细
    adf_table: pd.DataFrame                             # 平稳性检验
    ml_models: dict[str, MLResult]                      # symbol -> ML 产出


def _pooled(per_symbol: dict, name: str) -> pd.Series:
    frame = pd.DataFrame(
        {sym: strat_map[name].net_returns for sym, strat_map in per_symbol.items()}
    )
    return frame.mean(axis=1)  # 等权、跳过 warm-up 期缺失


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    """执行完整实验：拉数据 → 全策略回测 → 组合汇总 → 统计检验."""
    # ---- 1. 数据 ----
    data: dict[str, pd.DataFrame] = {}
    for sym, name in cfg.symbols.items():
        logger.info("拉取数据 %s(%s) [%s]", sym, name, cfg.source)
        data[sym] = fetch_ohlcv(sym, start=cfg.start, end=cfg.end, source=cfg.source,
                                cache_dir=cfg.cache_dir)

    # ---- 2. 分标的回测 ----
    per_symbol: dict[str, dict[str, BacktestResult]] = {}
    ml_models: dict[str, MLResult] = {}
    for sym, df in data.items():
        strat_map: dict[str, BacktestResult] = {}
        for sdef in STRATEGIES:
            strat_map[sdef.name_cn] = backtest(df, sdef.fn(df), cost=cfg.cost)
        if not cfg.skip_ml:
            mlr = ml_direction_signals(df, cfg.ml)
            ml_models[sym] = mlr
            strat_map[ML_STRATEGY.name_cn] = backtest(df, mlr.signal, cost=cfg.cost)
        per_symbol[sym] = strat_map
        logger.info("回测完成 %s(%s)", sym, cfg.symbols[sym])

    # ---- 3. 组合层汇总 ----
    all_names = list(next(iter(per_symbol.values())).keys())
    pooled_returns: OrderedDict[str, pd.Series] = OrderedDict()
    pooled_equity: OrderedDict[str, pd.Series] = OrderedDict()
    for name in all_names:
        pr = _pooled(per_symbol, name)
        pooled_returns[name] = pr
        pooled_equity[name] = (1.0 + pr).cumprod()

    bah = pooled_returns[STRATEGIES[0].name_cn]
    rows = []
    for name in all_names:
        pr = pooled_returns[name]
        st = performance_stats(pr, pooled_equity[name],
                               n_trades=int(sum(per_symbol[s][name].n_trades
                                                for s in per_symbol)))
        t0, p0 = ttest_mean_zero(pr)
        st["t_mu0"], st["p_mu0"] = t0, p0
        if name != bah.name:
            tv, pv = paired_ttest(pr, bah)
            st["t_vs_bah"], st["p_vs_bah"] = tv, pv
        else:
            st["t_vs_bah"], st["p_vs_bah"] = float("nan"), float("nan")
        rows.append(st)
    stats_pooled = pd.DataFrame(rows, index=all_names)

    # ---- 4. 分标的明细 ----
    detail_rows = []
    for sym, strat_map in per_symbol.items():
        for sname, bt in strat_map.items():
            st = performance_stats(bt.net_returns, bt.equity, bt.n_trades)
            st.update(trade_stats(trade_episodes(bt)))
            detail_rows.append({"symbol": sym, "stock_name": cfg.symbols[sym],
                                "strategy": sname, **st})
    stats_by_symbol = pd.DataFrame(detail_rows).set_index(["symbol", "stock_name", "strategy"])

    # ---- 5. 平稳性检验（论证收益建模的合理性）----
    adf_rows = []
    for sym, df in data.items():
        price_res = adf_test(np.log(df["close"]))
        ret_res = adf_test(df["close"].pct_change())
        adf_rows.append({"symbol": sym, "stock_name": cfg.symbols[sym],
                         "log_price_p": price_res["p_value"],
                         "ret_p": ret_res["p_value"],
                         "ret_stationary": ret_res["stationary"]})
    adf_table = pd.DataFrame(adf_rows).set_index(["symbol", "stock_name"])

    return ExperimentResult(
        cfg=cfg, data=data, per_symbol=per_symbol,
        pooled_returns=pooled_returns, pooled_equity=pooled_equity,
        stats_pooled=stats_pooled, stats_by_symbol=stats_by_symbol,
        adf_table=adf_table, ml_models=ml_models,
    )


def ml_binom_test(exp: ExperimentResult) -> tuple[int, int, float]:
    """汇总全部标的的 ML 方向命中率并做二项检验.

    Returns
    -------
    (k_correct, n_oof, p_value)
    """
    k = sum(m.k_correct for m in exp.ml_models.values())
    n = sum(m.n_oof for m in exp.ml_models.values())
    if n == 0:
        return 0, 0, float("nan")
    _, p = binom_direction_test(k, n)
    return k, n, p
