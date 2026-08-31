"""端到端集成测试：合成数据跑通 实验→图表→报告 全流程（无网络）。"""

import math

from trendquant.evaluation import ExperimentConfig, run_experiment
from trendquant.models.ml import MLConfig
from trendquant.plotting import make_figures
from trendquant.reporting import write_report


def test_end_to_end_synthetic(tmp_path):
    cfg = ExperimentConfig(
        symbols={"AAA": "演示甲", "BBB": "演示乙"},
        start="2021-01-01", end="2023-06-30", source="synthetic",
        cache_dir=tmp_path / "cache",
        ml=MLConfig(horizon=3, n_splits=2),
        skip_ml=False,
    )
    exp = run_experiment(cfg)

    # 6 个规则/统计策略 + 1 个 ML 策略
    assert exp.stats_pooled.shape[0] == 7
    assert set(exp.data) == {"AAA", "BBB"}
    assert "买入持有（基准）" in exp.stats_pooled.index
    assert math.isnan(exp.stats_pooled.loc["买入持有（基准）", "p_nw_vs_bah"])
    assert "p_nw_vs_bah_holm" in exp.stats_pooled.columns
    assert not exp.adf_table.empty

    figs = make_figures(exp, tmp_path)
    assert (tmp_path / "figures" / "equity_curves.png").exists()
    assert (tmp_path / "figures" / "metric_bars.png").exists()
    assert (tmp_path / "figures" / "ml_roc.png").exists()
    assert (tmp_path / "figures" / "ma_grid_sharpe.png").exists()

    report = write_report(exp, figs, tmp_path / "report.md")
    text = report.read_text(encoding="utf-8")
    assert "TrendQuant 量化策略实验报告" in text
    assert "统计检验" in text or "组合层绩效" in text
    assert "结论要点" in text
    assert "参数稳健性" in text


def test_end_to_end_skip_ml(tmp_path):
    cfg = ExperimentConfig(
        symbols={"AAA": "演示甲"},
        start="2022-01-01", end="2023-06-30", source="synthetic",
        cache_dir=tmp_path / "cache", skip_ml=True,
    )
    exp = run_experiment(cfg)
    assert exp.stats_pooled.shape[0] == 6
    assert not exp.ml_models
    figs = make_figures(exp, tmp_path)
    assert "ml_roc" not in figs
