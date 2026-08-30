"""TrendQuant —— 基于多源数据的股票趋势量化分析与策略评估框架.

模块结构
--------
- :mod:`trendquant.data`      数据获取（AkShare / yfinance / 合成数据）与本地缓存
- :mod:`trendquant.indicators` 技术指标（纯 pandas 向量化实现）
- :mod:`trendquant.features`   机器学习特征工程
- :mod:`trendquant.models`     策略模型（技术规则 / 统计因子 / 梯度提升）
- :mod:`trendquant.backtest`   无前视偏差回测引擎与绩效评价
- :mod:`trendquant.evaluation` 多标的 × 多策略实验编排、统计检验与报告生成
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
