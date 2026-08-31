# TrendQuant 方法论

本文档系统阐述 TrendQuant 的研究设计：数据、指标定义、策略构造、回测协议、
绩效评价与统计推断。所有公式与实现一一对应（模块/函数见各节标注），
实验结果见 [experiment_report.md](experiment_report.md)。

---

## 摘要

本文构建了一个可复现的 A 股日频趋势策略评估管线，在同一无前视偏差的回测
协议下，比较了 **5 个技术规则策略、1 个统计动量策略与 1 个梯度提升机器学习
策略** 相对买入持有基准的样本外表现，并以 t 检验、配对 t 检验与二项检验对
差异做显著性推断。设计上强制：信号 t 日收盘生成、t+1 开盘成交、双边计入
佣金/印花税/滑点、ML 采用 gap 等于标签前瞻期的 walk-forward 协议。

---

## 1. 引言与相关工作

历史价格是否包含可利用信息，是实证金融的长久争论：

- **有效市场假说**：Fama (1970) 提出弱式有效市场下历史价格信息不产生超额收益；
- **技术分析实证**：Brock, Lakonishok & LeBaron (1992) 在道琼斯指数百年样本
  上发现均线交叉与支撑阻力信号显著；Lo, Mamaysky & Wang (2000) 用非参数方法
  复核了形态识别的统计显著性；Park & Irwin (2007) 综述指出许多早期正结果
  在考虑数据窥探与交易成本后衰减；
- **动量效应**：Jegadeesh & Titman (1993) 记录了截面动量，Moskowitz, Ooi &
  Pedersen (2012) 记录了时间序列动量（TSMOM）；
- **机器学习定价**：Gu, Kelly & Xiu (2020) 系统比较了线性、树模型与神经网络
  在收益率预测上的表现，发现树模型与神经网络显著优于线性基准。

本文不预设立场，而是提供一个**结论可检验、结果可复现**的开源实验框架：
若策略不显著优于基准，这一"负结果"同样被完整报告（见实验报告 §7）。

## 2. 数据

### 2.1 数据源与复权

| 数据源 | 接口 | 复权方式 | 用途 |
|---|---|---|---|
| AkShare（东方财富） | `stock_zh_a_hist` | 前复权（qfq） | A 股默认来源 |
| yfinance | `yf.download(auto_adjust=True)` | 自动调整 | 备用/全球市场 |
| 合成数据 | 机制切换 GBM | — | 测试、CI、离线演示 |

前复权保证收益率序列在除权除息日连续，是收益建模的前提。
实现：`trendquant/data/loader.py`，含 CSV 本地缓存与重试。

### 2.2 股票池

8 只大市值、高流动性、跨行业标的（白酒/保险/电池/汽车/银行/券商/有色/公用
事业）。**需注意**：以"当前知名度"选股隐含幸存者偏差，见 §10。

### 2.3 平稳性

对对数价格与日收益率分别做 ADF 单位根检验（`backtest/metrics.adf_test`）。
实证结果（见实验报告 §2）：对数价格多数不能拒绝单位根，日收益率全部在
1% 水平平稳——这为"以收益率/价格比值为特征、以未来收益方向为标签"的
建模方式提供了计量依据。

## 3. 记号

记 $P_t$ 为 t 日收盘价（前复权），$O_t$ 为开盘价，日简单收益
$R_t = P_t / P_{t-1} - 1$。策略在 t 日收盘后输出的目标仓位记
$s_t \in \{0, 1\}$（1 = 满仓持有，0 = 空仓；仅做多）。

## 4. 技术指标定义

实现：`trendquant/indicators.py`（纯 pandas 向量化；窗口未满返回 NaN）。

**SMA**：

$$\mathrm{SMA}_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

**EMA**（递归式，与国内行情软件一致）：

$$\mathrm{EMA}_n(t) = \alpha P_t + (1-\alpha)\,\mathrm{EMA}_n(t-1),\quad \alpha = \frac{2}{n+1}$$

**MACD**（12/26/9）：

$$\mathrm{DIF} = \mathrm{EMA}_{12} - \mathrm{EMA}_{26},\qquad
\mathrm{DEA} = \mathrm{EMA}_{9}(\mathrm{DIF}),\qquad
\mathrm{HIST} = 2\,(\mathrm{DIF} - \mathrm{DEA})$$

**RSI**（Wilder 1978；用 $\alpha = 1/n$ 的递归平滑实现 Wilder 平滑）：

$$\mathrm{RSI}_n(t) = 100 \cdot \frac{\overline{G}_n(t)}{\overline{G}_n(t) + \overline{L}_n(t)}$$

其中 $\overline{G}_n$、$\overline{L}_n$ 分别为上涨幅度与下跌幅度的 Wilder
平均。区间内无下跌时 RSI 置 100。

**布林带**（20, 2σ，σ 取总体标准差）：

$$\mathrm{MID}_n = \mathrm{SMA}_n,\quad \mathrm{UP} = \mathrm{MID}_n + k\sigma_n,\quad
\mathrm{LOW} = \mathrm{MID}_n - k\sigma_n,\quad
\%B = \frac{P_t - \mathrm{LOW}}{\mathrm{UP} - \mathrm{LOW}}$$

**ATR**（真实波幅的 Wilder 平滑）：

$$\mathrm{TR}_t = \max\big(H_t - L_t,\ |H_t - P_{t-1}|,\ |L_t - P_{t-1}|\big)$$

**KDJ**（国内口径）：$\mathrm{RSV}_n = \frac{P_t - \min_n L}{\max_n H - \min_n L}\times 100$，
K 为 RSV 的 $\alpha=1/3$ 递归平滑，D 为 K 的同型平滑，$J = 3K - 2D$。

## 5. 策略构造

实现：`trendquant/models/`。统一接口：输入日线，输出 $s_t$（0/1）。

### 5.1 技术规则族（`models/rules.py`）

| 策略 | 进入条件 | 离场条件 |
|---|---|---|
| 双均线交叉 (20/60) | $\mathrm{SMA}_{20} > \mathrm{SMA}_{60}$ | 反之 |
| MACD (12/26/9) | $\mathrm{DIF} > \mathrm{DEA}$ | 反之 |
| RSI 均值回归 (14) | $\mathrm{RSI} < 30$ | $\mathrm{RSI} > 70$ |
| 布林带均值回归 (20, 2σ) | $\%B < 0$ | $\%B > 1$ |

### 5.2 时序动量（`models/statistical.py`）

$$s_t = \mathbb{1}\{P_t > P_{t-120}\}$$

即过去 120 日收益为正则持有（Moskowitz et al. 2012 的日频简化版）。

### 5.3 梯度提升方向预测（`models/ml.py`，`features.py`）

**特征**（15 个，全部仅用 t 日及以前信息）：

| 组 | 特征 |
|---|---|
| 价格动量 | $R^{(k)} = P_t/P_{t-k}-1,\ k \in \{1,5,10,20,60,120\}$ |
| 技术指标 | RSI/100、MACD HIST/$P_t$、%B、带宽、ATR/$P_t$、$P_t/\mathrm{SMA}_{20}-1$、$P_t/\mathrm{SMA}_{60}-1$ |
| 波动与量能 | 20 日收益波动率、成交量 / 20 日均量 |

**标签**（仅训练用，绝不进入特征）：

$$y_t = \mathbb{1}\{P_{t+h} > P_t\},\qquad h = 5$$

**模型**：`HistGradientBoostingClassifier`（max_iter=200，lr=0.05，
max_leaf_nodes=15，min_samples_leaf=40，L2=1.0，random_state=42）。
NaN 特征由模型原生处理。

**Walk-forward 协议**：`TimeSeriesSplit(n_splits=5, gap=h)` 扩展窗口；
仅取样本外（out-of-fold）预测概率，$s_t = \mathbb{1}\{\hat p_t > 0.55\}$。
**gap = h** 是防泄漏关键：训练集末尾样本的标签最多"前瞻" $h$ 日，
gap 保证标签窗口与测试期不重叠。

## 6. 回测协议

实现：`trendquant/backtest/engine.py`。

### 6.1 执行假设

信号 t 日收盘生成，t+1 日**开盘价**成交（消除"用 t 日收盘成交"的
前视嫌疑）。持仓状态转移对应的单日毛收益：

| 状态转移 | 含义 | 单日毛收益 |
|---|---|---|
| 0→1 | 开盘建仓 | $R^{g}_t = P_t / O_t - 1$ |
| 1→1 | 持有 | $R^{g}_t = P_t / P_{t-1} - 1$ |
| 1→0 | 开盘平仓 | $R^{g}_t = O_t / P_{t-1} - 1$ |
| 0→0 | 空仓 | $0$ |

### 6.2 成本模型

$$1+R_t = (1+R^{g}_t)(1-c_t),$$

其中建仓日 $c_t=c_{\text{buy}}$，平仓日 $c_t=c_{\text{sell}}$，其余日期为 0。
成本与价格收益乘法复合，避免忽略 $R^g_t c_t$ 交叉项。

$$c_{\text{buy}} = \text{佣金} + \text{滑点} = 0.025\% + 0.10\%,\qquad
c_{\text{sell}} = \text{佣金} + \text{印花税} + \text{滑点} = 0.175\%$$

净值 $V_t = \prod_{i \le t}(1 + R_i)$，初始 $V_0 = 1$。

### 6.3 无前视保障

仓位实现 $pos_t = s_{t-1}$：t 日收益只依赖 t-1 日收盘后可得信息。
`tests/test_models.py::test_ml_features_no_lookahead` 验证了特征在
历史截断下的取值与全样本计算完全一致；`tests/test_backtest.py` 验证了
执行时点与成本计提。

### 6.4 组合合成

各标的日净收益等权平均（近似每日再平衡的等权组合）：

$$R^{p}_t = \frac{1}{N}\sum_{i=1}^{N} R^{(i)}_t$$

## 7. 绩效评价指标

实现：`trendquant/backtest/metrics.py`（年化基准 252 交易日，无风险利率取 0）。

$$\text{年化收益} = V_T^{252/T} - 1,\qquad
\text{年化波动} = \hat\sigma_R \sqrt{252}$$

$$\text{Sharpe} = \frac{\bar R}{\hat\sigma_R}\sqrt{252},\qquad
\text{Sortino} = \frac{\bar R}{\sqrt{T^{-1}\sum_t\min(R_t,0)^2}}\sqrt{252}$$

$$\text{MDD} = \min_{t}\left(\frac{V_t}{\max_{i\le t} V_i} - 1\right),\qquad
\text{Calmar} = \frac{\text{年化收益}}{|\text{MDD}|}$$

另报告日胜率、建仓次数、交易回合胜率与平均持有期（`trade_episodes`）。

## 8. 统计推断

1. **收益显著性**：单样本 t 检验，$H_0: \mathbb{E}[R^{p}] = 0$（双侧）；
2. **相对基准**：配对 t 检验，$H_0: \mathbb{E}[R^{p} - R^{b}] = 0$（双侧），
   其中 $R^{b}$ 为同组合口径的买入持有日收益；
3. **方向预测能力**：二项检验，$H_0: p_{\text{hit}} = 0.5$（单侧，备择 $>0.5$）；
4. **稳健标准误**：主要结论使用 Newey-West HAC 标准误，滞后阶数按
   $\lfloor4(T/100)^{2/9}\rfloor$ 选取；普通 t 检验仅作为对照。
5. **多重检验**：策略相对基准的 Newey-West p 值再用 Holm 方法校正，
   控制多策略比较的家族错误率；报告同时保留原始 p 值供审计。

## 9. 局限与偏差来源（诚实清单）

- **数据窥探/多重检验**：策略参数取文献与业界常用值，但仍属事后选择；
  框架已报告双均线参数平面，但尚未采用独立验证集选择全部规则参数。
- **幸存者偏差**：股票池以当前大市值标的构成，隐含"它们在过去 7 年存续
  且做大"的前视选择。严谨的 Universe 应基于样本期初的成分股名单。
- **微观结构简化**：未建模 T+1 申赎限制、涨跌停无法成交、停牌、
  冲击成本随规模的非线性。
- **单一市场状态**：2019–2026 样本包含结构性牛市段，做多偏向的策略
  天然受益；结论不应外推至长期熊市。
- **ML 的样本效率**：单标的约 1500 个样本外样本对树模型而言偏少，
  横截面 pooled 建模（多标的共享模型）是有前景的改进。
- **剩余推断风险**：Newey-West 可缓解异方差与短程自相关，但不能消除结构突变、
  非平稳市场状态与样本选择偏差；更严格的扩展可采用 block bootstrap。

## 10. 复现

```bash
pip install -r requirements.txt
python -m pytest tests -q                     # 35 个单元/集成测试
python -m trendquant --source akshare --start 2019-01-01   # A股真实数据
python -m trendquant --demo                   # 无网络演示
```

环境：Python ≥ 3.10；随机性仅存在于模型初始化（random_state=42 固定）；
`HistGradientBoosting` 的 OpenMP 并行可能带来个位数样本的运行间差异。

## 参考文献

1. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
2. Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731–1764.
3. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.
4. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
5. Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of technical analysis. *Journal of Finance*, 55(4), 1705–1765.
6. Park, C.-H., & Irwin, S. H. (2007). What do we know about the profitability of technical analysis? *Journal of Economic Surveys*, 21(4), 786–826.
7. Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
8. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.

---

*TrendQuant 仅用于量化研究方法论的演示与教学，不构成任何投资建议。*
