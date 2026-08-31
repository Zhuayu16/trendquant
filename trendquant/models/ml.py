"""基于梯度提升的方向预测模型（walk-forward 样本外评估）.

方法论要点
----------
1. 标签为未来 ``horizon`` 日涨跌方向（二分类），特征全部来自 t 日及以前；
2. 使用 :class:`sklearn.model_selection.TimeSeriesSplit` 做 expanding-window
   walk-forward，且 ``gap = horizon``：训练集末尾样本的标签最多"前瞻"
   horizon 日，gap 保证标签窗口与测试期不重叠，杜绝标签泄漏；
3. 仅取样本外（out-of-fold）预测概率生成仓位信号，
   与规则/统计策略在同一回测框架下公平比较。

相关文献：Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine
Learning*, Review of Financial Studies 33(5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ..features import build_dataset

__all__ = ["MLConfig", "MLResult", "ml_direction_signals"]


@dataclass
class MLConfig:
    """梯度提升方向预测模型的配置."""

    horizon: int = 5                 # 标签前瞻期（交易日）
    n_splits: int = 5                # walk-forward 折数
    threshold: float = 0.55          # 看涨概率超过该阈值才建仓（保守触发）
    random_state: int = 42
    model_params: dict = field(
        default_factory=lambda: dict(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
        )
    )


@dataclass
class MLResult:
    """单标的 ML 信号的产出."""

    signal: pd.Series        # 0/1 目标仓位（与日线同索引）
    oof_prob: pd.Series      # 样本外看涨概率
    y: pd.Series             # 与 oof_prob 对齐的真实标签（用于汇总检验/ROC）
    hit_rate: float          # 样本外方向命中率（概率>0.5 视为看涨）
    auc: float | None        # 样本外 ROC AUC（类别不全时为 None）
    n_train_last: int        # 最后一折的训练样本量

    @property
    def n_oof(self) -> int:
        """样本外样本数."""
        return int(self.oof_prob.notna().sum())

    @property
    def k_correct(self) -> int:
        """样本外方向命中数（以 0.5 为分类阈值）."""
        valid = self.oof_prob.notna()
        pred_up = self.oof_prob[valid] > 0.5
        return int((pred_up == (self.y[valid] == 1)).sum())


def ml_direction_signals(df: pd.DataFrame, cfg: MLConfig | None = None) -> MLResult:
    """生成单标的的 walk-forward 样本外方向预测信号."""
    cfg = cfg or MLConfig()
    X, y = build_dataset(df, horizon=cfg.horizon)

    n_splits = min(cfg.n_splits, max(1, len(X) // 250))
    if n_splits < 2:
        raise ValueError(
            f"有效样本仅 {len(X)} 条，至少需要 500 条才能进行 2 折 walk-forward"
        )
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=cfg.horizon)
    prob = pd.Series(float("nan"), index=X.index, name="p_up")

    model = HistGradientBoostingClassifier(random_state=cfg.random_state, **cfg.model_params)
    n_train_last = 0
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        n_train_last = len(train_idx)

    signal = (prob > cfg.threshold).astype("float64").reindex(df.index).fillna(0.0)

    valid = prob.notna()
    pred_up = prob[valid] > 0.5
    hit_rate = float((pred_up == (y[valid] == 1)).mean()) if valid.any() else float("nan")
    auc: float | None = None
    if valid.any() and y[valid].nunique() == 2:
        auc = float(roc_auc_score(y[valid], prob[valid]))
    return MLResult(
        signal=signal,
        oof_prob=prob,
        y=y,
        hit_rate=hit_rate,
        auc=auc,
        n_train_last=n_train_last,
    )
