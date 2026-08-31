"""绩效评价与统计推断.

包含三部分：
1. 绩效指标（年化收益、波动、Sharpe、Sortino、最大回撤、Calmar 等）；
2. 交易级统计（回合、胜率、平均持有期）；
3. 统计检验（均值 t 检验、策略 vs 基准配对 t 检验、方向命中二项检验、ADF 平稳性检验）。

公式与假设检验的原假设定义见 ``docs/methodology.md`` 第 7、8 节。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .engine import BacktestResult

__all__ = [
    "TRADING_DAYS",
    "drawdown_series",
    "max_drawdown",
    "performance_stats",
    "trade_episodes",
    "trade_stats",
    "ttest_mean_zero",
    "ttest_hac_mean",
    "paired_ttest",
    "binom_direction_test",
    "adf_test",
]

TRADING_DAYS = 252


# ---------------------------------------------------------------- 绩效指标

def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列：``equity / cummax(equity) - 1``。"""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（非正值；无回撤时为 0）。"""
    return float(drawdown_series(equity).min())


def performance_stats(net_returns: pd.Series, equity: pd.Series, n_trades: int) -> dict:
    """由日净收益与净值计算绩效指标（年化基准 252 个交易日，无风险利率取 0）."""
    net = net_returns.dropna()
    n = len(net)
    if n < 2:
        raise ValueError("样本期过短，无法计算绩效指标")

    final = float(equity.iloc[-1])
    total_return = final - 1.0
    ann_return = final ** (TRADING_DAYS / n) - 1.0 if final > 0 else -1.0

    mu, sd = float(net.mean()), float(net.std(ddof=1))
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")

    # 目标收益率为 0 的下行偏差：所有观察期都进入分母，正收益贡献 0。
    # 这不同于“仅对负收益求标准差”，后者会错误地减去负收益均值。
    downside_deviation = float(np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2)))
    sortino = (
        mu / downside_deviation * np.sqrt(TRADING_DAYS)
        if downside_deviation > 0 else float("nan")
    )

    mdd = max_drawdown(equity)
    calmar = ann_return / abs(mdd) if mdd < 0 else float("nan")
    win_rate = float((net > 0).mean())

    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate_daily": win_rate,
        "n_days": n,
        "n_trades": n_trades,
    }


# ---------------------------------------------------------------- 交易级统计

def trade_episodes(result: BacktestResult) -> pd.DataFrame:
    """将 0/1 持仓序列切分为完整交易回合.

    Returns
    -------
    pd.DataFrame
        列：``entry_date``、``exit_date``、``gross_return``（回合毛收益，
        复利合成）、``days``（持有交易日数）。样本期末仍持有的回合
        以最后一个交易日为名义离场点，``open_at_end = True``。
    """
    pos = result.position
    change = pos.diff()
    first = pos.iloc[0]
    if first > 0:  # 序列起点即持仓（理论上不会出现，防御性处理）
        change.iloc[0] = 1.0

    entries = list(change[change > 0].index)
    exits = list(change[change < 0].index)

    rows: list[dict] = []
    for e in entries:
        e_loc = pos.index.get_loc(e)
        exit_after = [x for x in exits if pos.index.get_loc(x) >= e_loc]
        if exit_after:
            x = exit_after[0]
            x_loc = pos.index.get_loc(x)
        else:  # 期末仍持仓
            x, x_loc = pos.index[-1], len(pos.index) - 1
        seg = result.gross_returns.iloc[e_loc : x_loc + 1]
        # 持有天数 = 仓位为 1 的交易日数；平仓日（开盘卖出）不计入
        days = (x_loc - e_loc) if exit_after else (x_loc - e_loc + 1)
        rows.append(
            {
                "entry_date": e,
                "exit_date": x,
                "gross_return": float((1.0 + seg).prod() - 1.0),
                "days": int(days),
                "open_at_end": not exit_after,  # 样本期末仍持仓，离场日为名义值
            }
        )
    return pd.DataFrame(rows)


def trade_stats(episodes: pd.DataFrame) -> dict:
    """交易回合级统计：胜率、平均收益、盈亏比、平均持有期."""
    if episodes.empty:
        return {"n_round_trips": 0, "win_rate_trade": float("nan"),
                "avg_trade_return": float("nan"), "profit_loss_ratio": float("nan"),
                "avg_days_held": float("nan")}
    ret = episodes["gross_return"]
    wins, losses = ret[ret > 0], ret[ret <= 0]
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = abs(float(losses.mean())) if len(losses) else float("nan")
    ratio = float("nan")
    if not np.isnan(avg_loss) and avg_loss != 0.0:
        ratio = avg_win / avg_loss
    return {
        "n_round_trips": int(len(episodes)),
        "win_rate_trade": float((ret > 0).mean()),
        "avg_trade_return": float(ret.mean()),
        "profit_loss_ratio": ratio,
        "avg_days_held": float(episodes["days"].mean()),
    }


# ---------------------------------------------------------------- 统计检验

def ttest_mean_zero(daily: pd.Series) -> tuple[float, float]:
    """单样本 t 检验：H0 为日均收益 = 0（双侧）.

    Returns
    -------
    (t_stat, p_value)
    """
    daily = daily.dropna()
    res = sps.ttest_1samp(daily, 0.0)
    return float(res.statistic), float(res.pvalue)


def ttest_hac_mean(daily: pd.Series, maxlags: int | None = None) -> tuple[float, float, int]:
    """Newey-West (1987) HAC 稳健 t 检验：H0 为日均收益 = 0（双侧）.

    持仓平滑会使策略日收益呈正自相关，普通 t 检验的低估标准误会
    夸大显著性；本检验以 HAC 协方差校正自相关与异方差。
    滞后阶数默认采用 Newey-West 经验法则
    :math:`L = \\lfloor 4 (n/100)^{2/9} \\rfloor`。

    Returns
    -------
    (t_stat, p_value, lags_used)
    """
    x = daily.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        raise ValueError("样本过短，无法做 HAC 检验")
    if maxlags is None:
        maxlags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    import statsmodels.api as sm

    res = sm.OLS(x, np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}, use_t=True
    )
    return float(res.tvalues[0]), float(res.pvalues[0]), int(maxlags)


def paired_ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """配对 t 检验：H0 为两策略日均收益之差 = 0（双侧）."""
    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    res = sps.ttest_rel(joined["a"], joined["b"])
    return float(res.statistic), float(res.pvalue)


def binom_direction_test(k_correct: int, n: int) -> tuple[float, float]:
    """方向命中二项检验：H0 为命中率 = 50%（单侧，备择为 >50%）.

    Returns
    -------
    (hit_rate, p_value)
    """
    res = sps.binomtest(k_correct, n, p=0.5, alternative="greater")
    return k_correct / n, float(res.pvalue)


def adf_test(series: pd.Series) -> dict:
    """ADF 单位根检验（用于论证收益序列的平稳性）.

    Returns
    -------
    dict
        ``adf_stat``、``p_value``、``stationary``（p<0.05）。
    """
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    stat, pvalue, *_ = adfuller(s, autolag="AIC", result_object=False)
    return {"adf_stat": float(stat), "p_value": float(pvalue), "stationary": bool(pvalue < 0.05)}
