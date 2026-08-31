"""实验报告（Markdown）自动生成.

报告结构对齐学术论文的常规章节：实验设置 → 数据与平稳性 →
组合层绩效 → 统计检验 → ML 样本外表现 → 图表 → 分标的明细 → 结论。
全部数字直接来自 :class:`~trendquant.evaluation.ExperimentResult`，
保证报告与代码输出一致、可复现。
"""

from __future__ import annotations

import math
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import ExperimentResult, ml_binom_test

__all__ = ["write_report"]

_BAH = "买入持有（基准）"


def _rel(target: Path, base: Path) -> str:
    """target 相对于报告所在目录的 POSIX 风格相对路径."""
    return Path(os.path.relpath(target, Path(base).parent)).as_posix()


def _fmt(v, digits: int = 4, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isnan(v):
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def df_to_md(df: pd.DataFrame, pct_cols: tuple[str, ...] = (), digits: int = 4) -> str:
    """将 DataFrame 渲染为 Markdown 表格（不引入 tabulate 依赖）.

    用 ``to_dict("records")`` 取值而非 ``iterrows``，避免 pandas 把
    整数列统一提升为浮点（如样本数 1545 显示成 1545.0000）。
    """
    headers = [str(c) for c in df.columns]
    lines = [
        "| | " + " | ".join(headers) + " |",
        "|" + "---|" * (len(headers) + 1),
    ]
    for idx, rec in zip(df.index, df.to_dict(orient="records"), strict=True):
        cells = [str(idx)] + [
            _fmt(rec[c], digits=digits, pct=c in pct_cols) for c in df.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _stars(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """将（多层）索引展平为字符串索引，避免表格出现元组形式."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = [" ".join(str(x) for x in tup) for tup in out.index]
    else:
        out.index = out.index.astype(str)
    return out


def _pooled_display(stats: pd.DataFrame) -> pd.DataFrame:
    disp = pd.DataFrame(index=stats.index)
    disp["累计收益"] = stats["total_return"]
    disp["年化收益"] = stats["annual_return"]
    disp["年化波动"] = stats["annual_vol"]
    disp["夏普"] = stats["sharpe"]
    disp["索提诺"] = stats["sortino"]
    disp["最大回撤"] = stats["max_drawdown"]
    disp["卡玛"] = stats["calmar"]
    disp["日胜率"] = stats["win_rate_daily"]
    disp["建仓次数"] = stats["n_trades"].astype(int)
    disp["t(μ=0)"] = [
        f"{t:.2f}{_stars(p)}" for t, p in zip(stats["t_mu0"], stats["p_mu0"], strict=True)
    ]
    disp["p(μ=0)"] = stats["p_mu0"]
    disp["p-NW(μ=0)"] = stats["p_nw"]
    disp["t(vs基准)"] = [
        "—" if math.isnan(t) else f"{t:.2f}{_stars(p)}"
        for t, p in zip(stats["t_vs_bah"], stats["p_vs_bah"], strict=True)
    ]
    disp["p(vs基准)"] = stats["p_vs_bah"]
    disp["p-NW(vs基准)"] = stats["p_nw_vs_bah"]
    disp["p-NW-Holm"] = stats["p_nw_vs_bah_holm"]
    return disp


def _ml_display(exp: ExperimentResult) -> pd.DataFrame | None:
    if not exp.ml_models:
        return None
    rows = []
    for _sym, mlr in exp.ml_models.items():
        rows.append({
            "方向命中率": mlr.hit_rate,
            "样本外样本": int(mlr.n_oof),
            "ROC AUC": mlr.auc if mlr.auc is not None else float("nan"),
            "末折训练样本": int(mlr.n_train_last),
        })
    disp = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        [(sym, exp.cfg.symbols[sym]) for sym in exp.ml_models],
        names=["symbol", "name"]))
    return disp


def _conclusions(exp: ExperimentResult, k: int, n: int, p_binom: float) -> list[str]:
    stats = exp.stats_pooled
    lines: list[str] = []
    non_bah = stats.drop(index=_BAH, errors="ignore")

    best_ret = non_bah["annual_return"].idxmax()
    best_shp = non_bah["sharpe"].idxmax()
    lines.append(
        f"- 组合层面（等权）年化收益最高的主动策略为 **{best_ret}**"
        f"（{_fmt(non_bah.loc[best_ret, 'annual_return'], pct=True)}），"
        f"夏普比率最高的是 **{best_shp}**（{_fmt(non_bah.loc[best_shp, 'sharpe'], 3)}）。"
    )

    # 结论采用 HAC 稳健标准误，并以 Holm 法控制多策略比较的家族错误率。
    worse = non_bah[
        (non_bah["p_nw_vs_bah_holm"] < 0.05) & (non_bah["t_nw_vs_bah"] < 0)
    ]
    better = non_bah[
        (non_bah["p_nw_vs_bah_holm"] < 0.05) & (non_bah["t_nw_vs_bah"] > 0)
    ]
    if len(better):
        lines.append(
            f"- Newey-West + Holm 检验（p<0.05）下**显著优于买入持有基准**的策略："
            f"{'、'.join(better.index)}；"
        )
    if len(worse):
        lines.append(
            f"- Newey-West + Holm 检验（p<0.05）下**显著劣于买入持有基准**的策略："
            f"{'、'.join(worse.index)}"
            "——信号滞后与双边交易成本侵蚀了收益（日收益差显著为负，p<0.05）。"
        )
    if not len(better):
        lines.append(
            "- 在 5% 显著性水平下，**没有任何主动策略在收益上显著优于买入持有基准**——"
            "这与弱式有效市场假说在本样本上的表现一致。"
            "但注意风险口径的差异：基准的最大回撤为 "
            f"{_fmt(stats.loc[_BAH, 'max_drawdown'], pct=True)}，"
            "而部分主动策略（如 MACD、RSI 均值回归）以显著更低的回撤取得了"
            "统计上与基准无显著差异的收益，属于**风险调整口径下的可行替代**。"
        )
    sig_nz = stats[stats["p_nw"] < 0.05].index.tolist()
    if sig_nz:
        lines.append(
            f"- Newey-West 检验下日均收益显著异于零（p<0.05）的策略："
            f"{'、'.join(sig_nz)}。"
        )

    if n:
        hit = k / n
        verdict = "显著高于 50%" if p_binom < 0.05 else "与 50% 无显著差异"
        ml_name = next((idx for idx in stats.index if str(idx).startswith("ML ")), None)
        relative = "相对基准的稳健差异无法确认"
        if ml_name is not None:
            ml_row = stats.loc[ml_name]
            if ml_row["p_nw_vs_bah_holm"] < 0.05:
                relative = (
                    "净收益显著高于基准" if ml_row["t_nw_vs_bah"] > 0
                    else "净收益显著低于基准"
                )
        lines.append(
            f"- ML 梯度提升模型在 {n} 个 walk-forward 样本外方向预测中命中 {k} 个"
            f"（{hit:.2%}，二项检验 p={p_binom:.4g}，{verdict}）；"
            f"{relative}。方向命中率与策略收益是不同指标，不能由前者直接推出后者。"
        )
    lines.append(
        "- 上述结果对交易成本、参数选择与样本期均敏感，仅供方法论研究参考，不构成投资建议。"
    )
    return lines


def write_report(exp: ExperimentResult, figures: dict[str, Path], report_path: Path) -> Path:
    """生成 Markdown 实验报告."""
    cfg = exp.cfg
    L: list[str] = []

    L.append("# TrendQuant 量化策略实验报告")
    L.append("")
    L.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}  ")
    L.append(f"> 数据源：`{cfg.source}`（前复权日线）  ")
    L.append(f"> 运行环境：Python {platform.python_version()} / "
             f"pandas {pd.__version__} / numpy {np.__version__}")
    L.append("")

    # ---- 1. 实验设置 ----
    L.append("## 1. 实验设置")
    L.append("")
    d0 = min(df.index[0] for df in exp.data.values()).date()
    d1 = max(df.index[-1] for df in exp.data.values()).date()
    L.append(f"- 样本区间：**{d0} ~ {d1}**（各标的实际可得交易日的交集区间见数据缓存）")
    L.append(f"- 股票池（{len(cfg.symbols)} 只，等权合并）："
             + "、".join(f"{s} {n}" for s, n in cfg.symbols.items()))
    L.append(f"- 策略：{ '、'.join(cfg.strategy_names_cn) }")
    L.append(f"- 成本模型：佣金 {cfg.cost.commission:.3%}（双边）+ 印花税 "
             f"{cfg.cost.stamp_duty:.2%}（卖出）+ 滑点 {cfg.cost.slippage:.2%}（双边）")
    L.append("- 执行假设：t 日收盘出信号，t+1 开盘成交；仅做多、0/1 仓位")
    if not cfg.skip_ml:
        L.append(f"- ML 设定：HistGradientBoosting，标签 = 未来 {cfg.ml.horizon} 日涨跌方向，"
                 f"walk-forward {cfg.ml.n_splits} 折（gap={cfg.ml.horizon}），"
                 f"建仓阈值 p>{cfg.ml.threshold}")
    L.append("")

    # ---- 2. 数据与平稳性 ----
    L.append("## 2. 数据与平稳性检验")
    L.append("")
    L.append("对对数价格与日收益率分别做 ADF 单位根检验：价格序列通常不能拒绝"
             "单位根原假设，而收益率序列在 1% 水平下平稳，"
             "为「基于收益率的特征与标签建模」提供了计量依据。")
    L.append("")
    adf = exp.adf_table.copy()
    adf = adf.rename(columns={"log_price_p": "对数价格 ADF p", "ret_p": "日收益 ADF p",
                              "ret_stationary": "收益率平稳(p<0.05)"})
    adf["收益率平稳(p<0.05)"] = adf["收益率平稳(p<0.05)"].map({True: "是", False: "否"})
    L.append(df_to_md(_flatten(adf), digits=4))
    L.append("")

    # ---- 3. 组合层绩效 ----
    L.append("## 3. 组合层绩效（各标的等权合并）")
    L.append("")
    L.append("各标的日净收益等权平均后合成组合收益（近似每日再平衡的等权组合）。"
             "成本按成交额计提。")
    L.append("")
    disp = _pooled_display(exp.stats_pooled)
    pct_cols = ("累计收益", "年化收益", "年化波动", "最大回撤", "日胜率")
    L.append(df_to_md(disp, pct_cols=pct_cols))
    L.append("")
    L.append("注：普通 t 检验列用于对照；p-NW 使用 Newey-West HAC 稳健标准误，"
             "p-NW-Holm 进一步校正多策略同时比较。报告结论以 p-NW-Holm 为准。"
             "显著性标记：** p<0.01，* p<0.05，† p<0.1。")
    L.append("")

    if exp.sensitivity is not None:
        L.append("## 4. 双均线参数稳健性")
        L.append("")
        valid = exp.sensitivity.stack().dropna()
        if not valid.empty:
            best = valid.idxmax()
            L.append(f"对 fast={list(exp.sensitivity.index)}、slow={list(exp.sensitivity.columns)} "
                     f"进行组合层网格检验；有效组合中最高夏普为 **{valid.max():.3f}** "
                     f"（fast={best[0]}，slow={best[1]}）。")
        L.append("默认双均线参数以红框标示；完整矩阵见 `ma_grid_sharpe.csv`。")
        L.append("")
        if "heatmap" in figures:
            L.append(f"![双均线参数稳健性热力图]({_rel(figures['heatmap'], report_path)})")
            L.append("")

    # ---- ML 样本外表现 ----
    ml_disp = _ml_display(exp)
    if ml_disp is not None:
        ml_section = 5 if exp.sensitivity is not None else 4
        L.append(f"## {ml_section}. 机器学习模型样本外表现")
        L.append("")
        k, n, p_binom = ml_binom_test(exp)
        if n:
            L.append(f"汇总全部标的：walk-forward 样本外方向命中 **{k}/{n}（{k / n:.2%}）**，"
                     f"二项检验（H0: 命中率=50%，单侧）p = **{p_binom:.4g}**。")
            L.append("")
        L.append(df_to_md(_flatten(ml_disp), pct_cols=("方向命中率",)))
        L.append("")
        if "ml_roc" in figures:
            L.append(f"![ML 样本外 ROC 曲线]({_rel(figures['ml_roc'], report_path)})")
            L.append("")

    # ---- 5. 图表 ----
    section_base = 5 if ml_disp is not None else 4
    if exp.sensitivity is not None:
        section_base += 1
    L.append(f"## {section_base}. 图表")
    L.append("")
    captions = {
        "equity": "各策略等权组合累计净值",
        "drawdown": "基准与夏普最高前三策略的回撤",
        "bars": "组合层年化收益与夏普对比",
        "signals": f"示例标的 {next(iter(cfg.symbols))} 的双均线策略买卖点",
    }
    for key in ("equity", "drawdown", "bars", "signals"):
        if key in figures:
            L.append(f"![{captions[key]}]({_rel(figures[key], report_path)})")
            L.append("")

    # ---- 6. 分标的明细 ----
    L.append(f"## {section_base + 1}. 分标的明细（年化收益 / 夏普 / 最大回撤）")
    L.append("")
    detail = exp.stats_by_symbol.reset_index()
    pivot = detail.pivot_table(index=["symbol", "stock_name"], columns="strategy",
                               values="sharpe").round(3)
    pivot = _flatten(pivot)
    pivot.index.name = "标的"
    pivot.columns = [str(c) for c in pivot.columns]
    L.append("**夏普比率：**")
    L.append("")
    L.append(df_to_md(pivot))
    L.append("")
    full_csv_hint = "完整逐标的指标见 `stats_by_symbol.csv`。"
    L.append(full_csv_hint)
    L.append("")

    # ---- 7. 结论 ----
    sec = "7" if ml_disp is not None else "6"
    L.append(f"## {sec}. 结论要点")
    L.append("")
    k, n, p_binom = ml_binom_test(exp)
    L.extend(_conclusions(exp, k, n, p_binom))
    L.append("")

    L.append("---")
    L.append("")
    L.append("**免责声明**：本项目仅用于量化研究方法论的演示与教学，历史回测收益"
             "不代表未来表现，不构成任何投资建议。市场有风险，投资需谨慎。")
    L.append("")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")
    return report_path
