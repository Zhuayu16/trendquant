"""实验图表绘制.

为保证在任何平台（含 CI Linux）渲染一致，图内文字统一使用英文；
中文说明见实验报告 ``report.md`` 与 ``README.md``。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器/CI）下渲染
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .backtest.engine import BacktestResult
from .evaluation import ML_STRATEGY, STRATEGIES, ExperimentResult

__all__ = ["make_figures"]

_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f"]
_BAH_COLOR = "#111111"


def _name_en(cn: str) -> str:
    for s in STRATEGIES:
        if s.name_cn == cn:
            return s.name_en
    if cn == ML_STRATEGY.name_cn:
        return ML_STRATEGY.name_en
    return cn


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _setup(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y", useOffset=False)


def fig_price_with_signals(df: pd.DataFrame, bt: BacktestResult, title: str, path: Path) -> Path:
    """收盘价 + 均线 + 买卖点标记."""
    from . import indicators as ta

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["close"], lw=1.0, color="#1f77b4", label="Close (qfq)")
    ax.plot(df.index, ta.sma(df["close"], 20), lw=0.8, color="#ff7f0e", label="SMA20")
    ax.plot(df.index, ta.sma(df["close"], 60), lw=0.8, color="#9467bd", label="SMA60")

    pos = bt.position
    entry = pos.diff() > 0
    exit_ = pos.diff() < 0
    ax.scatter(df.index[entry], df["open"][entry], marker="^", s=42, color="#2ca02c",
               zorder=5, label="Entry (next-day open)")
    ax.scatter(df.index[exit_], df["open"][exit_], marker="v", s=42, color="#d62728",
               zorder=5, label="Exit (next-day open)")

    ax.set_title(title)
    ax.set_ylabel("Price")
    _setup(ax)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return _save(fig, path)


def fig_equity_curves(pooled_equity: dict[str, pd.Series], path: Path) -> Path:
    """组合层累计净值对比."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, eq) in enumerate(pooled_equity.items()):
        if name == "买入持有（基准）":
            ax.plot(eq.index, eq.values, lw=1.8, color=_BAH_COLOR, ls="--",
                    label=_name_en(name))
        else:
            ax.plot(eq.index, eq.values, lw=1.1, color=_PALETTE[i % len(_PALETTE)],
                    label=_name_en(name))
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title("Equal-weight pooled cumulative NAV by strategy")
    ax.set_ylabel("NAV (start = 1.0)")
    _setup(ax)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return _save(fig, path)


def fig_drawdown(pooled_equity: dict[str, pd.Series], focus: list[str], path: Path) -> Path:
    """重点策略与基准的回撤曲线."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, name in enumerate(focus):
        eq = pooled_equity[name]
        dd = eq / eq.cummax() - 1.0
        color = _BAH_COLOR if name == "买入持有（基准）" else _PALETTE[(i + 1) % len(_PALETTE)]
        ax.plot(dd.index, dd.values, lw=1.0, color=color, label=_name_en(name))
        ax.fill_between(dd.index, dd.values, 0.0, color=color, alpha=0.12)
    ax.set_title("Drawdown: benchmark vs. top strategies (by Sharpe)")
    ax.set_ylabel("Drawdown")
    _setup(ax)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    return _save(fig, path)


def fig_metric_bars(stats_pooled: pd.DataFrame, path: Path) -> Path:
    """年化收益与 Sharpe 水平条形图."""
    s = stats_pooled.sort_values("sharpe")
    labels = [_name_en(i) for i in s.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in s["annual_return"]]
    axes[0].barh(labels, s["annual_return"] * 100, color=colors)
    axes[0].set_xlabel("Annualized return (%)")
    axes[0].axvline(0, color="gray", lw=0.6)

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in s["sharpe"]]
    axes[1].barh(labels, s["sharpe"], color=colors)
    axes[1].set_xlabel("Sharpe ratio (rf = 0)")
    axes[1].axvline(0, color="gray", lw=0.6)

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Pooled performance by strategy", y=1.0)
    return _save(fig, path)


def fig_ml_roc(exp: ExperimentResult, path: Path) -> Path | None:
    """ML 模型样本外 ROC 曲线（合并全部标的 + 各标的细线）."""
    if not exp.ml_models:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 5.6))

    probs_all, ys_all = [], []
    for (sym, mlr), color in zip(exp.ml_models.items(), _PALETTE, strict=False):
        valid = mlr.oof_prob.notna()
        p, y = mlr.oof_prob[valid], mlr.y[valid]
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=0.7, alpha=0.45, color=color,
                    label=f"{sym} (AUC={auc(fpr, tpr):.3f})")
        probs_all.append(p)
        ys_all.append(y)

    yp, yy = pd.concat(probs_all), pd.concat(ys_all)
    fpr, tpr, _ = roc_curve(yy, yp)
    pooled_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.0, color="#1f77b4", label=f"Pooled (AUC={pooled_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="Random (AUC=0.500)")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ML-GBDT out-of-fold ROC (5-day direction)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def make_figures(exp: ExperimentResult, out_dir: Path) -> dict[str, Path]:
    """生成全部图表，返回 {角色: 文件路径}."""
    fig_dir = Path(out_dir) / "figures"
    figs: dict[str, Path] = {}

    figs["equity"] = fig_equity_curves(exp.pooled_equity, fig_dir / "equity_curves.png")

    stats = exp.stats_pooled
    focus_names = ["买入持有（基准）"] + list(
        stats.drop(index="买入持有（基准）", errors="ignore")["sharpe"]
        .sort_values(ascending=False).head(3).index
    )
    figs["drawdown"] = fig_drawdown(exp.pooled_equity, focus_names,
                                    fig_dir / "drawdown.png")
    figs["bars"] = fig_metric_bars(stats, fig_dir / "metric_bars.png")

    if not exp.cfg.skip_ml:
        sym0 = next(iter(exp.data))
        ma_def = next(s for s in STRATEGIES if s.key == "ma")
        title = f"{sym0} - {ma_def.key.upper()} crossover signals"
        figs["signals"] = fig_price_with_signals(
            exp.data[sym0], exp.per_symbol[sym0][ma_def.name_cn], title,
            fig_dir / f"signals_{sym0}.png",
        )
        roc = fig_ml_roc(exp, fig_dir / "ml_roc.png")
        if roc:
            figs["ml_roc"] = roc
    return figs
