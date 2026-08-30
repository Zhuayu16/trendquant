<div align="center">

# TrendQuant 📈

**基于多源数据的股票趋势量化分析与策略评估框架**

*A reproducible, statistics-first framework for stock trend analysis and strategy evaluation on Chinese A-shares.*

[![CI](https://github.com/YOUR_USERNAME/trendquant/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/trendquant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)
![Code style](https://img.shields.io/badge/lint-ruff-261230)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## 📌 项目定位

这不是一个「炒股软件」，而是一个**结论可检验、结果可复现**的量化研究管线：
在严格无前视偏差的统一回测协议下，比较技术规则、统计动量与机器学习三类策略
相对买入持有基准的样本外表现，并用假设检验回答一个问题——

> **「看走势做判断」的技术信号，扣除真实成本后到底有没有统计上可辨认的预测力？**

设计对齐实证金融的研究规范：信号 t 日收盘生成、t+1 开盘成交、双边计入
佣金/印花税/滑点、ML 使用 gap 等于标签前瞻期的 walk-forward 协议。
负结果同样完整报告（本样本上，没有策略在收益口径显著跑赢买入持有——
详见[核心发现](#-核心发现)与[实验报告](docs/experiment_report.md)）。

## ✨ 核心特性

- **多数据源**：AkShare（A股前复权）/ yfinance（备用与全球市场）/ 确定性合成数据（测试与离线演示），本地 CSV 缓存与重试
- **纯向量化指标库**：SMA/EMA/MACD/RSI(Wilder)/BOLL/ATR/KDJ，零 TA-Lib 依赖，窗口未满显式返回 NaN
- **三类策略同台**：技术规则（双均线、MACD、RSI、布林带）· 统计动量（TSMOM）· 梯度提升方向预测
- **无前视回测引擎**：t 收盘出信号 → t+1 开盘成交；建仓/持有/平仓日收益分别精确建模
- **ML 防泄漏**：15 个特征的标签只用于训练；`TimeSeriesSplit(gap=horizon)` 保证标签窗口与测试期不重叠；仅用样本外概率生成信号
- **统计推断内建**：单样本 t / 配对 t / 二项 / ADF 平稳性检验，报告自动标注显著性
- **一键产出**：Markdown 实验报告 + 5 张出版级图表 + 3 个 CSV，全部由代码生成
- **工程质量**：35 个单元/集成测试、Ruff 静态检查、GitHub Actions 多版本 CI、pyproject 现代打包

## 📊 核心发现

8 只 A 股等权组合 · 2019-01 ~ 2026-08 · 1856 个交易日 · 含双边交易成本：

| 策略 | 年化收益 | 夏普 | 最大回撤 | p(μ=0) | p(vs 基准) |
|---|---:|---:|---:|---:|---:|
| 买入持有（基准） | **23.46%** | 1.118 | -26.56% | 0.0024 | — |
| MACD (12/26/9) | 14.24% | 1.045 | **-14.66%** | 0.005 | 0.057 |
| RSI 均值回归 (14) | 10.76% | 1.046 | -16.59% | 0.005 | 0.026 ⁻ |
| 双均线交叉 (20/60) | 9.80% | 0.743 | -24.13% | 0.044 | 0.003 ⁻ |
| 布林带均值回归 | 7.22% | 0.673 | -19.98% | 0.068 | 0.004 ⁻ |
| 时序动量 (120d) | 6.78% | 0.533 | -38.29% | 0.148 | 0.0004 ⁻ |
| ML-GBDT (5日方向) | 2.47% | 0.284 | -20.77% | 0.441 | 0.0002 ⁻ |

> ⁻ = 配对 t 检验下**显著劣于**基准（p<0.05）

**三条主要结论**（详见[实验报告 §7](docs/experiment_report.md)）：

1. **收益口径**：5% 显著性水平下，没有任何主动策略显著优于买入持有——与弱式有效市场假说在本样本的表现一致；
2. **风险口径**：MACD 以基准一半的回撤（-14.7% vs -26.6%）取得了与基准无显著差异的收益（p=0.057），Calmar 0.97 反超基准的 0.88，是风险调整口径下的可行替代；
3. **ML 的启示**：方向命中率 50.83%（12360 个样本外预测，二项检验 p=0.034，显著高于抛硬币）却依然亏给基准——**预测精度 ≠ 策略盈利能力**，成本与保守触发阈值吃掉了全部统计优势。

| 各策略累计净值 | 回撤对比 |
|---|---|
| ![净值曲线](docs/figures/equity_curves.png) | ![回撤](docs/figures/drawdown.png) |

## 🚀 快速开始

```bash
git clone https://github.com/YOUR_USERNAME/trendquant.git
cd trendquant

# Python ≥ 3.10；推荐 conda 环境
pip install -r requirements.txt        # 含 akshare、pytest、ruff
python -m pytest tests -q              # 35 个测试，~1 分钟

# 无网络演示（合成数据，10 秒出全套报告）
python -m trendquant --demo

# A 股真实数据完整实验（内置 8 只标的，首次运行自动缓存）
python -m trendquant --source akshare --start 2019-01-01

# 自定义股票池 / 跳过 ML
python -m trendquant --symbols 600519,300750,00700 --source yfinance
python -m trendquant --skip-ml         # 只跑规则与统计策略
```

输出（`outputs/`，已 gitignore）：

```
outputs/
├── report.md               # 自动生成的实验报告（与 docs/experiment_report.md 同构）
├── stats_pooled.csv        # 组合层绩效 + 检验
├── stats_by_symbol.csv     # 分标的 × 分策略明细
├── adf_tests.csv           # 平稳性检验
└── figures/                # 净值、回撤、条形图、买卖点、ROC
```

## 🔬 方法论

```
┌─────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ 数据层   │ → │  指标与特征   │ → │      策略层          │ → │      回测层           │
│ AkShare │   │ 15 个 ML 特征 │   │ 技术规则 × 4         │   │ t 收盘出信号          │
│ yfinance│   │ SMA/EMA/MACD │   │ TSMOM 动量 × 1       │   │ t+1 开盘成交          │
│ 合成数据 │   │ RSI/BOLL/ATR │   │ ML-GBDT walk-forward │   │ 佣金+印花税+滑点      │
└─────────┘   └──────────────┘   └─────────────────────┘   └──────────┬───────────┘
                                                                      ↓
                    ┌───────────────────────────┐   ┌──────────────────────────────┐
                    │        报告层             │ ← │        统计检验层             │
                    │ Markdown 报告 + 5 张图表   │   │ t 检验 / 配对 t / 二项 / ADF  │
                    └───────────────────────────┘   └──────────────────────────────┘
```

学术设计的完整推导（含全部公式、执行协议、检验的原假设与**诚实的局限清单**）
见 **[docs/methodology.md](docs/methodology.md)**；
实跑产出的完整报告见 **[docs/experiment_report.md](docs/experiment_report.md)**。

关键防错设计一览：

- ✅ 信号 t 收盘生成，t+1 开盘成交（`tests/test_backtest.py` 逐情形验证）
- ✅ ML 特征历史截断一致性测试（`test_ml_features_no_lookahead`）
- ✅ `gap=horizon` 杜绝标签窗口与测试期重叠
- ✅ 印花税仅计入卖出，滑点双边计提
- ✅ 报告与图表 100% 由当次运行生成，数字与代码输出严格一致

## 📁 项目结构

```
trendquant/
├── trendquant/                 # 主包
│   ├── data/loader.py          #   AkShare / yfinance / 合成数据 + 缓存
│   ├── indicators.py           #   技术指标（纯 pandas 向量化）
│   ├── features.py             #   ML 特征工程与标签
│   ├── models/
│   │   ├── rules.py            #   技术规则策略
│   │   ├── statistical.py      #   时序动量
│   │   └── ml.py               #   梯度提升 + walk-forward
│   ├── backtest/
│   │   ├── engine.py           #   无前视回测引擎
│   │   └── metrics.py          #   绩效指标 + 统计检验
│   ├── evaluation.py           #   实验编排（多标的 × 多策略）
│   ├── plotting.py             #   图表（英文标注，跨平台渲染一致）
│   ├── reporting.py            #   Markdown 报告生成
│   ├── universe.py             #   默认 A 股股票池
│   └── cli.py                  #   python -m trendquant 入口
├── docs/
│   ├── methodology.md          #   学术方法论（公式 + 检验 + 局限）
│   ├── experiment_report.md    #   示例实验报告（真实数据跑出的结果）
│   └── figures/                #   示例图表
├── tests/                      #   35 个单元/集成测试
├── .github/workflows/ci.yml    #   CI（Python 3.10 & 3.13 × lint + pytest）
└── pyproject.toml              #   打包 / Ruff / Pytest 配置
```

## 🗺️ Roadmap

- [ ] Newey-West / block bootstrap 稳健推断（处理日收益自相关）
- [ ] 涨跌停、T+1、停牌的成交约束建模
- [ ] 横截面 pooled ML（多标的共享模型）与简单组合优化
- [ ] 参数平面稳健性热力图（对抗数据窥探）
- [ ] 基于样本期初成分股的 Universe（消除幸存者偏差）

## ⚠️ 免责声明

本项目**仅用于量化研究方法论的演示与教学**。历史回测收益不代表未来表现，
所有输出不构成任何投资建议。市场有风险，投资需谨慎。

## 🔖 引用

如果你在研究中使用了本项目，欢迎引用（上传后 GitHub 会出现 "Cite this repository" 按钮）：

```bibtex
@software{TrendQuant,
  title  = {TrendQuant: 基于多源数据的股票趋势量化分析与策略评估框架},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/trendquant},
  note   = {Multi-source stock trend analysis and strategy evaluation}
}
```

## 📚 参考文献

精选（完整列表见 [docs/methodology.md](docs/methodology.md)）：

- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2).
- Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5).
- Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2).
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5).

---

<div align="center">

**MIT License** · 用统计说话，对结果诚实。

</div>
