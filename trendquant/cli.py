"""命令行入口.

示例
----
.. code-block:: bash

    # 内置 A 股股票池完整实验（akshare 拉取真实数据）
    python -m trendquant

    # 自定义标的与区间
    python -m trendquant --symbols 600519,300750 --start 2021-01-01

    # 无网络演示（合成数据）
    python -m trendquant --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .evaluation import ExperimentConfig, run_experiment
from .models.ml import MLConfig
from .plotting import make_figures
from .reporting import write_report
from .universe import DEFAULT_UNIVERSE

logger = logging.getLogger("trendquant")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trendquant",
        description="TrendQuant —— 股票趋势量化分析与策略评估框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", choices=["akshare", "yfinance", "synthetic"],
                   default="akshare", help="数据源")
    p.add_argument("--symbols", default=None,
                   help="逗号分隔的标的代码，默认使用内置股票池")
    p.add_argument("--start", default="2019-01-01", help="样本起始日 YYYY-MM-DD")
    p.add_argument("--end", default=None, help="样本结束日，默认今天")
    p.add_argument("--out", default="outputs", help="输出目录")
    p.add_argument("--cache-dir", default="data_cache", help="数据缓存目录")
    p.add_argument("--horizon", type=int, default=5, help="ML 标签前瞻期（交易日）")
    p.add_argument("--threshold", type=float, default=0.55, help="ML 建仓概率阈值")
    p.add_argument("--splits", type=int, default=5, help="walk-forward 折数")
    p.add_argument("--skip-ml", action="store_true", help="跳过 ML 策略（快速运行）")
    p.add_argument("--demo", action="store_true",
                   help="演示模式：合成数据 + 前 4 只标的（无需联网）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)

    if args.demo:
        source, symbols = "synthetic", dict(list(DEFAULT_UNIVERSE.items())[:4])
        logger.info("演示模式：合成数据，标的 %s", ",".join(symbols))
    else:
        source = args.source
        if args.symbols:
            codes = [s.strip() for s in args.symbols.split(",") if s.strip()]
            symbols = {c: DEFAULT_UNIVERSE.get(c, c) for c in codes}
        else:
            symbols = dict(DEFAULT_UNIVERSE)

    cfg = ExperimentConfig(
        symbols=symbols, start=args.start, end=args.end, source=source,
        cache_dir=args.cache_dir,
        ml=MLConfig(horizon=args.horizon, threshold=args.threshold, n_splits=args.splits),
        skip_ml=args.skip_ml,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始实验：%d 只标的 × %s", len(symbols),
                "含 ML" if not cfg.skip_ml else "不含 ML")
    exp = run_experiment(cfg)
    figures = make_figures(exp, out_dir)
    report = write_report(exp, figures, out_dir / "report.md")

    exp.stats_pooled.to_csv(out_dir / "stats_pooled.csv", encoding="utf-8-sig")
    exp.stats_by_symbol.to_csv(out_dir / "stats_by_symbol.csv", encoding="utf-8-sig")
    exp.adf_table.to_csv(out_dir / "adf_tests.csv", encoding="utf-8-sig")

    key = ["annual_return", "sharpe", "max_drawdown", "p_mu0", "p_vs_bah"]
    print("\n=== 组合层绩效摘要（等权合并） ===")
    print(exp.stats_pooled[key].to_string(float_format=lambda v: f"{v: .4f}"))
    print(f"\n实验报告: {report.resolve()}")
    print(f"图表目录: {(out_dir / 'figures').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
