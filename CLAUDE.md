# Collective_Memory 项目

## 📋 目录

1. [项目总览](#项目总览)
2. [共享理论基础](#共享理论基础)
3. [子项目分支](#子项目分支)
   - [分支 1: Rich's Conflict Resolution Model](#分支-1-richs-conflict-resolution-model-)
   - [分支 2: Tim's Bahrami Parameter Sweep](#分支-2-tims-bahrami-parameter-sweep-)
   - [分支 3: IRB Protocol #29910](#分支-3-irb-protocol-29910-)
   - [分支 4: Online Experiment System](#分支-4-online-experiment-system-)
4. [会议记录](#会议记录)
5. [项目管理](#项目管理)

---

# 项目总览

## 核心研究问题

**当多个个体各自拥有对同一项目的记忆证据时，群体如何整合这些证据形成集体判断？**

子问题：
1. **规范性**：什么是"最优"的群体整合方式？（贝叶斯上限）
2. **描述性**：人们实际上如何整合群体信息？
3. **机制**：不同整合规则在什么条件下表现更好/更差？
4. **社会因素**：信心交流、专长差异、社会压力如何影响群体判断？

## 研究团队

| 成员 | 单位 | 专长 | 角色 |
|------|------|------|------|
| **Rich Shiffrin** | Indiana University | REM 模型创始人 | 导师（模型理论） |
| **Tim Pleskac** | UCR | 决策建模、贝叶斯推理 | 导师（群体决策） |
| **Steve Clark** | UCR | 识别记忆实验、SDT | 实验设计顾问 |
| **Suparna Rajaram** | Stony Brook | 协作记忆、群体抑制 | 协作记忆专家 |
| **Yiyan (你)** | UCR | 博士生 | REM 模拟实现 |

## 项目架构

```
Collective_Memory/
│
├── 理论基础（共享）
│   ├── REM 模型
│   ├── Ernst & Banks (2002) - MLE 最优整合
│   ├── Bahrami et al. (2010) - 社会贝叶斯
│   └── Signal Detection Theory
│
├── 分支 1: Rich's Model (独立) ⭐
│   └── Conflict Resolution Model
│       └── P = ((1+D)/(2+D))^beta
│
├── 分支 2: Tim's Models (独立) ⭐
│   ├── Bahrami Parameter Sweep
│   ├── Rich's Theory Verification
│   └── Confidence Miscalibration
│
├── 分支 3: IRB (独立) 📋
│   └── Protocol #29910 修订
│
└── 分支 4: Online Experiment (独立) 🌐
    └── Firebase + React 实时多人实验
```

**重要**：分支 1 和分支 2 完全独立，各做各的模型。Rich 的模型关注冲突解决，Tim 的模型关注能力异质性和信心校准。

---

# 共享理论基础

所有分支共享的理论框架和术语。

## REM 模型（Retrieving Effectively from Memory）

**来源**：Shiffrin & Steyvers (1997)

### 核心公式

**熟悉度比（Odds）**：
$$\text{Odds} = \frac{P(\text{Old} \mid D)}{P(\text{New} \mid D)}$$

**完整推导**：
$$\frac{P(O \mid D)}{P(N \mid D)} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{c + (1-c)g}{g}\right)^{K_M^i} (1-c)^{K_N^i}$$

**参数**：
- $K_M^i$：probe 与第 i 条记忆痕迹**匹配**的特征数
- $K_N^i$：**不匹配**的特征数
- $c$：学习准确度（encoding accuracy）
- $g$：猜中特征的基础概率

### 关键洞见

> REM 的输出不是 yes/no 二值决策，而是**连续的证据强度**（log-odds）。
> 这正是社会决策规则（WCS / DSS）能够"接上"的关键！

### 判断准则

- Odds > 1 → 判断 "Old"
- Odds < 1 → 判断 "New"
- 准则 = 1 是 REM 推导自然给出的，**不需要人为设定**

### REM 在本项目中的角色

REM 只是"证据生成器"，负责产生个体层面的内部证据（log-odds），**不负责社会整合**。社会整合由各个分支的决策规则实现。

---

## Ernst & Banks (2002) - 最优多感觉整合

**来源**：Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature*, 415, 429-433.

### 核心理论

跨感觉线索的最优整合方式是**按可靠性（1/方差）加权平均**。

**MLE 公式**：
$$\hat{S} = \sum_i w_i \hat{S}_i \quad \text{where} \quad w_i = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$$

**整合后方差**：
$$\sigma^2_{\text{combined}} = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

### 与本项目的关系

Ernst 的框架是 **DSS（Direct Signal Sharing）** 的规范基础：如果群体成员能直接共享内部证据，最优整合方式就是可靠性加权（在 REM 框架下即 log-odds 相加）。

---

## Bahrami et al. (2010) - 社会贝叶斯决策

**来源**：Bahrami, B., et al. (2010). Optimally interacting minds. *Science*, 329, 1081-1085.

### 核心命题（原文）

> "These patterns can be explained by a model in which two heads are Bayes optimal under the assumption that individuals accurately communicate their level of confidence."

### 四种社会决策规则

| 模型 | 全称 | 机制描述 | 与 REM 的关系 |
|------|------|----------|---------------|
| **CF** | Coin Flip | 分歧时随机选择 | 完全忽略 REM 证据 |
| **UM** | Uniform/Majority | 多数投票 | REM odds → 二值决策 → 投票 |
| **WCS** | Weighted Confidence Sharing | 按信心加权平均 | REM odds ≈ confidence |
| **DSS** | Direct Signal Sharing | 直接整合内部证据 | REM log-odds 直接相加（贝叶斯最优） |

### 关键发现

1. **相似性效应**：当群体成员能力相似时，群体表现最优
2. **信心交流**：准确的信心交流是达到贝叶斯最优的关键
3. **理论上限**：DSS 是理论上限，但人类无法真正"直接共享内部信号"

---

## Signal Detection Theory (SDT)

### 基础指标

**d' (d-prime)**：敏感度
$$d' = \Phi^{-1}(\text{HR}) - \Phi^{-1}(\text{FAR})$$

**Hautus Correction**（避免 HR=1 或 FAR=0）：
$$\text{HR} = \frac{\text{hits} + 0.5}{S + 1}, \quad \text{FAR} = \frac{\text{fas} + 0.5}{N + 1}$$

### SDT 四象限

| 真实 \ 判断 | 判断 "Old" | 判断 "New" |
|-------------|-----------|-----------|
| **真 Old** | Hit ✓ | Miss ✗ |
| **真 New** | False Alarm ✗ | Correct Rejection ✓ |

### 与 REM 的对应

| SDT 概念 | REM 对应 |
|----------|----------|
| d' | log(Odds) 的区分度 |
| yes/no decision | Odds > 1 |

---

# 子项目分支

## 分支 1: Rich's Conflict Resolution Model ⭐

**负责人**：Rich Shiffrin
**状态**：✅ 已完成实现（2025年1月18日）
**独立性**：与 Tim 的模型完全独立

### 研究问题

**当两个 agent 意见不一致时（一个说 Old，一个说 New），群体如何基于证据强度差异来决定听谁的？**

### 核心公式

#### 强度计算

1. **Log-odds → Odds**：
   $$\phi = \exp(L)$$

2. **固定幂次缩放**：
   $$\phi' = \phi^{1/11}$$

   **注意**：1/11 是固定值，**不是**可调参数。

3. **强度（Strength）**：
   $$S = \max(\phi', 1/\phi')$$

   确保 $S \geq 1$（强度总是正值）。

4. **强度差异**：
   $$D = |S_A - S_B|$$

#### 决策规则

**原始公式**（错误）：
$$P_{\text{choose stronger}} = \frac{1 + D}{2 + D}$$

**修正公式**（当前实现）：
$$P_{\text{choose stronger}} = \left(\frac{1 + D}{2 + D}\right)^{\beta}$$

**参数**：
- $\beta$：默认 1.0，可调整以测试模型灵活性
- 当 $D = 0$：$P = 0.5$（随机猜）
- 当 $D \to \infty$：$P \to 1$（确定选择更强的）

### 实现细节

#### 文件位置

**主文件**：`/Users/yiytan/Collective_Memory/Simulations/src/group_rules.py`
**函数**：`rich_conflict_rule()` (Lines 455-536)

#### 函数签名

```python
def rich_conflict_rule(
    L_A: np.ndarray,           # Agent A 的 log-odds
    L_B: np.ndarray,           # Agent B 的 log-odds
    labels: np.ndarray,        # 真实标签（1=Old, 0=New）
    rng: np.random.Generator,  # 随机数生成器
    beta: float = 1.0          # 幂次参数
) -> dict
```

#### 返回值

```python
{
    'dprime_A': float,
    'dprime_B': float,
    'dprime_team': float,
    'decisions': np.ndarray,      # 团队决策数组
    'conflict_mask': np.ndarray,  # 冲突试次标记
    'strength_A': np.ndarray,     # Agent A 的强度
    'strength_B': np.ndarray,     # Agent B 的强度
    'D_values': np.ndarray        # 强度差异值
}
```

#### 关键实现细节

**冲突定义**：
```python
conflict_mask = (D_A != D_B)  # 一个说 Old，一个说 New
```

**强度计算**：
```python
odds_A = np.exp(L_A)
phi_scaled_A = np.power(odds_A, 1/11)  # 固定 1/11
S_A = np.maximum(phi_scaled_A, 1/phi_scaled_A)
```

**概率计算**：
```python
P_choose_stronger = np.power((1.0 + D) / (2.0 + D), beta)
```

### 验证实验

**文件**：`/Users/yiytan/Collective_Memory/Simulations/src/run_simulation.py`
**函数**：`run_rich_conflict_simulation(beta=1.0)` (Lines 582-745)

**输出**：
- `rich_conflict_results.csv`：冲突试次的经验概率 vs 理论预测
- `rich_conflict_plot.png`：散点图 + 理论曲线

**验证目标**：
- 经验概率（模拟）是否匹配理论公式
- Beta 参数是否按预期影响曲线形状

### 与 Tim 模型的区别

| 维度 | Rich's Model | Tim's Model |
|------|-------------|-------------|
| 关注点 | 冲突解决（disagree 时听谁的） | 能力异质性对群体增益的影响 |
| 核心指标 | P(choose stronger \| conflict) | Collective Benefit Ratio |
| 分析对象 | **仅冲突试次** | 所有试次 |
| 独立性 | 完全独立分析 | 扫描参数空间 |

### 使用示例

```python
from src import group_rules
import numpy as np

# 生成模拟数据
L_A = np.random.randn(100)
L_B = np.random.randn(100)
labels = np.random.randint(0, 2, 100)
rng = np.random.default_rng(42)

# 运行 Rich's 模型
result = group_rules.rich_conflict_rule(L_A, L_B, labels, rng, beta=1.0)

# 查看冲突试次比例
print(f"Conflict trials: {np.sum(result['conflict_mask'])}")

# 查看平均强度差异
print(f"Mean D: {np.mean(result['D_values'][result['conflict_mask']])}")
```

---

## 分支 2: Tim's Bahrami Parameter Sweep ⭐

**负责人**：Tim Pleskac
**状态**：✅ 已完成实现（2025年1月18日）
**独立性**：与 Rich 的模型完全独立

### 研究问题

**群体成员能力异质性（ability heterogeneity）如何影响不同决策规则下的群体表现？**

### 三个分析模块

#### 模块 1: Bahrami Parameter Sweep

**目标**：比较 5 种决策规则在不同能力组合下的表现

**实验设计**：
- 固定：Agent A 能力 $c_A = 0.7$（专家）
- 扫描：Agent B 能力 $c_B \in [0.1, 0.9]$，步长 0.1
- 规则：CF, UW, DMC, DSS, BF

**核心指标**：
$$\text{CBR} = \frac{d'_{\text{team}}}{\max(d'_A, d'_B)}$$

- CBR > 1：群体优于最佳个体（集体增益）
- CBR = 1：群体等于最佳个体
- CBR < 1：群体劣于最佳个体（集体损失）

**输出**：
- `bahrami_sweep_final.csv`：45 行（9 c_B × 5 rules）
- `bahrami_sweep_plot.png`：5 条曲线对比图

#### 模块 2: Rich's Theory Verification

**目标**：验证 DSS 是否达到 SDT 理论上限

**理论预测**（独立噪声假设）：
$$d'_{\text{optimal}} = \sqrt{d'_A^2 + d'_B^2}$$

**验证**：
- 对比 DSS 模拟结果 vs 理论公式
- 定量测量偏差：$|d'_{\text{DSS}} - d'_{\text{theory}}|$

**输出**：
- `rich_theory_verification.png`：左图 d' 绝对值，右图 CBR

**注意**：这里的 "Rich's verification" 是验证 **DSS 的数学推导**，与分支 1 的 "Rich's conflict model" 无关。

#### 模块 3: Confidence Miscalibration

**目标**：研究信心校准偏差（confidence miscalibration）对群体决策的影响

**实验设计**：
- 固定：$c_A = c_B = 0.7$，$\alpha_A = 1.2$（A 过度自信）
- 扫描：$\alpha_B \in [0.5, 1.5]$，步长 0.1
- 规则：UW_Miscal, DMC_Miscal, DSS, CF

**重要**：UW_Miscal 是 UW + Prelec，不是 Bahrami 的 WCS。命名反映 Tim 的模型语义。

**Prelec 概率权重函数**：
$$w(p) = \exp(-\beta \cdot (-\ln p)^\alpha)$$

**约束条件**：
$$\beta = (\ln 2)^{1-\alpha} \quad \Rightarrow \quad w(0.5) = 0.5$$

**参数解释**：
- $\alpha = 1$：完美校准（$w = p$）
- $\alpha > 1$：过度自信（极端值被夸大）
- $\alpha < 1$：不够自信（极端值被压缩）

**输出**：
- `miscalibration_sweep.csv`：44 行（11 α_B × 4 models）
- `miscalibration_plot.png`：4 条曲线对比图（UW_Miscal, DMC_Miscal, DSS, CF）

### 实现细节

#### 文件位置

**核心文件**：
- `/Users/yiytan/Collective_Memory/Simulations/src/group_rules.py`（7 个规则函数）
- `/Users/yiytan/Collective_Memory/Simulations/src/run_simulation.py`（3 个扫描函数）
- `/Users/yiytan/Collective_Memory/Simulations/notebooks/bahrami_sweep_demo.ipynb`（Jupyter 包装器）

#### 7 个群体决策规则

**基础规则**（5 个）：

1. **CF (Coin Flip)**：
   ```python
   # 分歧时随机选择
   if D_A != D_B:
       D_team = rng.choice([D_A, D_B])
   else:
       D_team = D_A
   ```

2. **UW (Uniform Weighting)**：
   ```python
   # 原始 odds 算术平均（非 log 空间！）
   odds_A = np.exp(L_A)
   odds_B = np.exp(L_B)
   mean_odds = (odds_A + odds_B) / 2
   D_team = (mean_odds > 1).astype(int)
   ```

3. **DMC (Defer to Max Confidence)**：
   ```python
   # 听信心最大的人
   conf_A = np.abs(L_A)
   conf_B = np.abs(L_B)
   D_team = np.where(conf_A > conf_B, D_A, D_B)
   ```

4. **DSS (Direct Signal Sharing)**：
   ```python
   # log-odds 相加（贝叶斯最优）
   L_team = L_A + L_B
   D_team = (L_team > 0).astype(int)
   ```

5. **BF (Behavior & Feedback)**：
   ```python
   # 基于个体正确率学习（trial-by-trial）
   # CRITICAL: 更新基于个体正确性，非群体决策正确性
   for i in range(n):
       if D_A[i] == labels[i]: score_A += 1
       if D_B[i] == labels[i]: score_B += 1
       D_team[i] = D_A[i] if score_A >= score_B else D_B[i]
   ```

**信心校准规则**（2 个）：

6. **UW_Miscal**（原 WCS_Miscal）：
   ```python
   # UW + Prelec weighting（算术平均 w，不是信心加权）
   w_A = prelec_weighting(L_A, alpha_A)
   w_B = prelec_weighting(L_B, alpha_B)
   w_team = (w_A + w_B) / 2  # 简单算术平均
   D_team = (w_team > 0.5).astype(int)  # 严格大于
   ```

7. **DMC_Miscal**：
   ```python
   # DMC + Prelec weighting（选择 max |w - 0.5|）
   w_A = prelec_weighting(L_A, alpha_A)
   w_B = prelec_weighting(L_B, alpha_B)
   conf_A = np.abs(w_A - 0.5)  # 距离中点的距离
   conf_B = np.abs(w_B - 0.5)
   D_team = np.where(conf_A > conf_B, D_A, D_B)
   ```

#### 关键设计决策

**1. UW 必须用原始 Odds**（非 log 空间）：
```python
# CORRECT
mean_odds = (np.exp(L_A) + np.exp(L_B)) / 2

# WRONG
mean_log_odds = (L_A + L_B) / 2  # 这是 DSS!
```

**2. BF 更新机制**（CRITICAL）：
- 分数更新基于**个体正确性**
- 不是基于群体决策正确性
- Trial-by-trial 顺序处理

**3. Prelec 约束**：
$$\beta = (\ln 2)^{1-\alpha}$$
确保 $w(0.5) = 0.5$（交叉点不变）。

**4. DMC_Miscal 信心定义**：
```python
conf = |w - 0.5|  # 距离中点的距离
```
不是直接用 $w$ 作为信心。

#### RNG 独立性策略

```python
# 条件间独立
for idx, condition in enumerate(conditions):
    condition_seed = seed_master + idx
    rng_test = np.random.default_rng(condition_seed)
    rng_A = np.random.default_rng(condition_seed + 1000)
    rng_B = np.random.default_rng(condition_seed + 2000)
    rng_cf = np.random.default_rng(condition_seed + 3000)
```

**设计原理**：
- 每个参数点独立初始化 RNGs
- Agent A 和 B 使用不同 seed（+1000, +2000）
- 刺激列表共享（rng_test），确保两人看到相同项目

### 使用示例

```bash
# 进入模拟目录
cd /Users/yiytan/Collective_Memory/Simulations/

# 运行完整扫描（Python 脚本）
python run_simulation.py

# 或使用 Jupyter Notebook
jupyter notebook bahrami_sweep_demo.ipynb
```

**交互式运行**：
```python
import run_simulation

# 运行 Bahrami sweep
df_bahrami = run_simulation.run_bahrami_sweep()

# 运行 miscalibration sweep
df_miscal = run_simulation.run_miscalibration_sweep()

# 查看结果
print(df_bahrami.groupby('rule')['collective_benefit_ratio'].mean())
```

### 输出文件

```
Simulations/outputs/
├── bahrami_sweep_final.csv          # 45 rows (9 c_B × 5 rules)
├── miscalibration_sweep.csv         # 44 rows (11 α_B × 4 models)
├── bahrami_sweep_plot.png           # 5-curve comparison
├── rich_theory_verification.png     # DSS vs theory (2 subplots)
└── miscalibration_plot.png          # 4-model comparison
```

### 验证清单

完成实现后验证的关键点：

- ✅ UW 计算原始 odds 平均（非 log 空间）
- ✅ BF 基于个体正确性更新分数
- ✅ 所有规则返回 d' 指标
- ✅ Bahrami sweep 生成 45 行（9 × 5）
- ✅ Prelec beta 确保 w(0.5) = 0.5
- ✅ UW_Miscal 使用简单算术平均 w（不是信心加权）
- ✅ DMC_Miscal 使用 |w - 0.5| 作为信心
- ✅ 三个分析部分完全独立
- ✅ CSV 包含所有必需列
- ✅ traces 在循环内生成（Monte Carlo 活跃）
- ✅ test_items 在循环外生成（固定考试）

---

## 分支 3: IRB Protocol #29910 📋

**负责人**：Yiyan（你）
**状态**：✅ 已提交修订（2025年1月19日）
**独立性**：完全独立的行政任务

### 背景

IRB Protocol #29910 提交后收到反馈，要求修订多个文档以解决一致性和细节问题。

### 核心策略：Umbrella Protocol

**目标**：让 IRB 批准一个通用框架，具体实验可以在框架内灵活调整。

**原则**：
> "只要 IRB 没问的，你就不要写那么清楚"

具体实现：
1. ✅ **只回答 IRB 明确问到的问题**
2. ❌ **不要过度详细化实验设计**
3. ✅ **保持程序描述的通用性**
4. ❌ **不要把实验"写死"**

### 修订内容

#### 网页问答（3个问题）

**Question 0812: Research Procedures**
- 说明时长：20 分钟
- 描述两个主要阶段：Individual + Collaborative
- 说明在线协作的方式（不详细到具体参数）
- 强调两个平台（Prolific vs SONA）程序相同

**Question 0813: Identifiable Information**
- Answer: No
- 说明只收集 ID 用于补偿，48小时内移除

**Question 0818: Payment Arrangement**
- Prolific: $3.50
- SONA: 0.5 学分 + 替代方式（写短文）

#### 修订文档（6个）

**文件位置**：`/Users/yiytan/Collective_Memory/IRB/`

生成的文件（带 `_REVISED` 后缀）：
1. `#29910- SIS_Prolific_REVISED.docx`
2. `#29910- SIS_SONA_REVISED.docx`
3. `#29910- Recruitment_Ad_Prolific_REVISED.docx`
4. `#29910- Recruitment_Ad_SONA_REVISED.docx`
5. `#29910- INST_Prolific_REVISED.docx` ⚠️ 需要添加刺激
6. `#29910- INST_SONA_REVISED.docx` ⚠️ 需要添加刺激

**INST 文件的特殊处理**：
- 添加明确的占位符：`[USER TO ADD STIMULI HERE]`
- 提供示例格式（但不填入实际内容）
- 分为两个部分：Study Phase Stimuli + Test Phase Stimuli

### 关键经验

#### IRB 问答的艺术

| IRB 问题 | 过度详细（❌） | 恰当（✅） |
|---------|--------------|----------|
| How long? | "5 phases: Setup (1 min), Study (5 min), Test (7 min)..." | "Approximately 20 minutes" |
| What procedures? | "50 words, 4 seconds each, 100 test trials..." | "View stimuli, complete memory test, may collaborate online" |
| How collaborate? | "Custom chat interface, 500 char limit..." | "See others' responses, view statistics, optionally discuss via text" |

#### 有用的模板短语

**时间描述**：
- "approximately 20 minutes"
- "will vary among participants"
- "on average"

**程序描述**：
- "such as"
- "may include"
- "participants can optionally"
- "if they choose to"

**灵活表述**：
- "a series of stimuli (e.g., words or images)"
- "various memory test formats"
- "different types of collaboration"

**避免使用**：
- 精确数字（"exactly 50 items"）
- 固定时间（"Study Phase - 5 minutes"）
- 绝对语句（"all participants will"）

### 一致性检查清单

- ✅ 所有文档的时间都是 20 分钟
- ✅ Prolific 支付金额一致（$3.50）
- ✅ SONA 学分一致（0.5 credit）
- ✅ SONA 包含 alternative credit 选项
- ✅ Prolific 和 SONA 的程序描述相同
- ✅ INST 文件包含实际材料的占位符
- ✅ 所有 SIS 都提到数据去标识化（48小时内）

### 下次修订 IRB 的快速检查表

1. ☐ 阅读所有 IRB feedback PDF
2. ☐ 列出所有 Action Items
3. ☐ 向用户确认关键参数（时间、补偿、程序）
4. ☐ 询问是否需要保持灵活性（Umbrella protocol?）
5. ☐ 生成网页问答答案（简洁版）
6. ☐ 生成修订后的文档（_REVISED 版本）
7. ☐ 一致性检查（时间、金额、程序描述）
8. ☐ 创建使用说明文件
9. ☐ 告知用户哪里需要手动添加内容

---

## 分支 4: Online Experiment System 🌐

**负责人**：Yiyan（你）
**状态**：✅ 已部署（2025年1月）
**独立性**：纯工程实现，独立于模拟分支

### 项目位置

`/Users/yiytan/memory-game`

**部署地址**：https://collective-memory-d3802.web.app
**Firebase 控制台**：https://console.firebase.google.com/project/collective-memory-d3802/overview

### 技术架构

```
Frontend:
  - React 19.2.0 (Create React App)
  - Lucide React (图标库)
  - CSS-in-JS (inline styles)

Backend:
  - Firebase Firestore (实时数据库)
  - Firebase Authentication (匿名登录)
  - Firebase Hosting (部署)

Build & Deploy:
  - npm run build → /build 目录
  - firebase deploy → Hosting
```

### 实验流程

```
1. Login & Matchmaking
   ↓
2. Lobby Wait (等待其他被试加入)
   ↓
3. Study Phase (学习单词，所有人看相同材料)
   ↓
4. Test Phase
   ├─ Step 1: Individual Decision (独立判断 Old/New + 信心评分)
   └─ Step 2: Group Discussion (圆桌视图 + 文字讨论 + 修改决策)
   ↓
5. Results (完成界面 + Completion Code)
```

### 关键参数配置

**位置**：`src/App.js` Lines 35-39

```javascript
const DEBUG_MODE = true;  // 调试模式（单人测试）
const TARGET_GROUP_SIZE = DEBUG_MODE ? 1 : 3;  // 群体大小
const AUTO_START_DELAY = DEBUG_MODE ? 2 : 5;   // 自动开始延迟（秒）
const STUDY_WORD_DURATION = DEBUG_MODE ? 1000 : 2000;  // 单词显示时长（ms）
const STUDY_GAP_DURATION = DEBUG_MODE ? 500 : 500;     // 间隔时长（ms）
```

**重要**：部署到正式实验前，将 `DEBUG_MODE` 改为 `false`。

### Firestore 数据结构

**Document 路径**：`experiments/auto_room_{roomId}`

```javascript
{
  // 房间信息
  roomId: string,
  hostId: string,
  status: 'lobby' | 'study' | 'test' | 'finished',

  // 参与者
  players: {
    [uid]: { name, oderId, joinedAt }
  },

  // 实验材料
  testList: [{ word, type: 'target'|'lure' }],

  // 响应数据
  responses: {
    [trialIndex]: {
      [userId]: {
        initial: {
          decision: 'old'|'new',
          confidence: 1-5,
          rt: number,
          isCorrect: boolean,
          sdtCategory: 'hit'|'miss'|'false_alarm'|'correct_rejection',
          ...
        },
        final: {...}  // 群体讨论后的决策
      }
    }
  },

  // 聊天消息
  chatMessages: {
    [trialIndex]: [
      { oderId, name, message, timestamp }
    ]
  }
}
```

### 响应数据格式（心理学实验标准）

```javascript
{
  // 核心响应
  decision: 'old' | 'new',
  confidence: 1-5,
  rt: number,               // Reaction Time (ms)
  timestamp: number,
  timeElapsed: number,      // 从测试阶段开始的累计时间

  // 刺激信息
  stimulus: string,
  stimulusType: 'target' | 'lure',
  isOld: boolean,

  // 准确性
  isCorrect: boolean,
  sdtCategory: 'hit' | 'miss' | 'false_alarm' | 'correct_rejection',

  // 试次信息
  trialIndex: number,
  step: 1 | 2,              // 个体/群体阶段

  // 参与者信息
  oderId: string
}
```

### 开发和部署工作流

#### 本地开发

```bash
cd /Users/yiytan/memory-game
npm install  # 首次
npm start    # 启动开发服务器，访问 localhost:3000
```

#### 部署到 Firebase

```bash
npm run build      # 构建生产版本 → /build 目录
firebase deploy    # 部署到 Hosting
```

#### Firebase CLI 配置

```bash
firebase login            # 首次部署需登录
firebase init hosting     # 初始化配置
```

### 常见修改场景

#### 1. 修改实验材料

**位置**：`src/App.js` Lines 60-67

```javascript
const TARGET_WORDS = [
  "Cat", "Book", "Tree", ...  // 你的单词
];
const LURE_WORDS = [
  "Dog", "Pen", "Flower", ...  // 你的干扰词
];
```

#### 2. 调整时间参数

**位置**：`src/App.js` Lines 35-39

```javascript
const STUDY_WORD_DURATION = 3000;  // 改为 3 秒
const STUDY_GAP_DURATION = 1000;   // 改为 1 秒间隔
```

#### 3. 切换到正式实验模式

**位置**：`src/App.js` Line 35

```javascript
const DEBUG_MODE = false;  // 关闭调试模式
```

**效果**：
- 需要完整的组才能开始
- 测试阶段包含两个步骤（个体 + 群体）
- 恢复正常的时间参数

### 数据导出和分析

#### 方法 1：Firebase Console（手动）

访问：https://console.firebase.google.com/project/collective-memory-d3802/firestore
Collection: `experiments`

#### 方法 2：Firebase Admin SDK（推荐）

```javascript
// data_export.js
const admin = require('firebase-admin');
const fs = require('fs');

admin.initializeApp({
  credential: admin.credential.applicationDefault(),
  projectId: 'collective-memory-d3802'
});

const db = admin.firestore();

async function exportData() {
  const snapshot = await db.collection('experiments').get();
  const data = snapshot.docs.map(doc => ({
    id: doc.id,
    ...doc.data()
  }));

  fs.writeFileSync('experiment_data.json', JSON.stringify(data, null, 2));
}

exportData();
```

#### Python 数据处理示例

```python
import json
import pandas as pd

# 读取导出的数据
with open('experiment_data.json') as f:
    rooms = json.load(f)

# 提取所有响应数据
all_responses = []
for room in rooms:
    room_id = room['roomId']
    players = room['players']
    responses = room.get('responses', {})

    for trial_idx, trial_data in responses.items():
        for user_id, user_responses in trial_data.items():
            # 个体阶段
            if 'initial' in user_responses:
                initial = user_responses['initial']
                all_responses.append({
                    'room_id': room_id,
                    'user_id': user_id,
                    'participant_name': players[user_id]['name'],
                    'trial': trial_idx,
                    'phase': 'individual',
                    'decision': initial['decision'],
                    'confidence': initial['confidence'],
                    'rt': initial['rt'],
                    'is_correct': initial['isCorrect'],
                    'sdt_category': initial['sdtCategory']
                })

            # 群体阶段
            if 'final' in user_responses:
                final = user_responses['final']
                all_responses.append({...})  # 类似结构

df = pd.DataFrame(all_responses)
df.to_csv('analysis_ready_data.csv', index=False)
```

### 与 REM 模拟的关系

**当前 Demo**：
- 纯行为实验
- 记录 decision (Old/New) 和 confidence
- 适合测试群体决策规则（Majority, WCS 等）

**未来整合 REM**：
1. 在服务器端运行 REM 模拟生成 log-odds
2. 用 REM 输出替代真实被试的部分角色（confederate agents）
3. 分析真实数据后用 REM 拟合参数

**数据对应**：
- Decision (Old/New) ↔ REM: Odds > 1
- Confidence (1-5) ↔ REM: |log(Odds)|
- Group decision ↔ REM + Social rules (DSS, WCS, etc.)

### 快速上手（下次启动时）

```bash
# 1. 本地测试
cd /Users/yiytan/memory-game
npm start
# 打开 localhost:3000，输入 Participant ID，开始测试

# 2. 修改材料/参数
# 编辑 src/App.js 第 35-67 行，保存后自动刷新

# 3. 部署到线上
npm run build
firebase deploy
# 访问 https://collective-memory-d3802.web.app

# 4. 查看数据
# Firebase Console → Firestore → experiments collection
```

**忘记功能在哪？搜索关键词**：
- `handleManualJoin` - 匹配机制
- `StudyPhase` - 学习阶段
- `TestPhase` - 测试阶段
- `handleSubmit` - 提交响应
- `sendChatMessage` - 聊天功能

### 文件结构

```
/Users/yiytan/memory-game/
├── src/
│   ├── App.js         # ⭐ 核心实验逻辑（1481 行）
│   ├── index.js       # React 入口
│   └── ...
├── build/             # 生产构建输出
├── package.json       # 项目配置
├── firebase.json      # Firebase Hosting 配置
└── .firebaserc        # Firebase 项目关联
```

### 重要提醒

1. **不要修改原始文件**：始终在 Git 管理下工作
2. **调试模式开关**：部署前确认 `DEBUG_MODE = false`
3. **Firebase 用量**：Firestore 免费额度有限，大规模测试前检查配额
4. **数据备份**：定期从 Firestore 导出数据到本地

---

# 会议记录

## 2024年12月大会议

### 参与者

- Rich Shiffrin（Indiana）
- Tim Pleskac（UCR）
- Steve Clark（UCR）
- Suparna Rajaram（Stony Brook）
- Yiyan（UCR，因网络问题大部分时间只能听）

### 关键讨论点

#### 1. 识别 vs 回忆的群体效应

**Suparna Rajaram 的观点**：
- 在**回忆**任务中，群体协作通常导致**抑制**（collaborative inhibition）
- 原因：每个人有自己的提取顺序，协作打乱了这个顺序
- 但在**线索回忆（cued recall）**中，抑制效应消失（因为每个人用相同的线索）

**对识别的推论**：
- 识别任务中也用统一的"线索"（即测试项目本身）
- 因此可能不会看到抑制效应
- 更有趣的问题是：**人们如何使用关于他人和项目的信息？**

#### 2. 项目差异与个体差异

**Rich Shiffrin 的观察**：
- Rob Nosofsky 的研究显示：不同项目的可记忆性差异很大
- 如果每个人都记住相同的项目，群体就无法获得"众人智慧"的增益
- 在最近的识别实验中，83 名被试的表现从接近随机到接近完美呈**线性分布**

**Steve Clark 的分析**：
- 计算了两人之间的**分歧次数**
- 观察值略低于独立假设的预测值
- 说明人们在相同项目上倾向于正确或错误

#### 3. 信心评分的作用

**Steve Clark 的实验观察**：

> "If you have people make confident judgments and then talk about the responses they've made, 90% of their conversation is about their confidence."
>
> "Oh, I picked horse. Oh, I picked table. Oh, I gave it a 9. I gave it a 7. You win. Next."

**问题**：信心评分可能让任务变得"过于简单"，不能反映真实的协作过程。

**Steve 的新实验**：不收集信心评分，观察人们如何在没有这个"捷径"的情况下协商。

#### 4. 信心校准与社会因素

**观察到的现象**：
- 一个人开始给 10 分，另一个给 6-7 分
- 给 10 分的人为了"不显得像混蛋"，主动降低了自己的评分
- 这是一个**社会校准**过程

**Rich 的担忧**：
- 被试可能相信"高信心 = 高准确"
- 但实际上信心和准确性的相关性不完美
- 如果有反馈，被试可能会学习调整信心
- 这是一个需要控制的混淆因素

#### 5. 实验设计建议

**Steve Clark 的预曝光设计**：

```
阶段1（分开）：
- 被试 A：学习词表的前半部分（3次）
- 被试 B：学习词表的后半部分（3次）

阶段2（一起）：
- 两人坐在同一个屏幕前
- 学习完整词表（1次）

测试：
- 两人合作做识别测试
- 被试 A 对前半部分更有"专长"
- 被试 B 对后半部分更有"专长"
```

**优势**：
- 被试确信他们看到了相同的内容（因为在同一屏幕前）
- 但他们带着不同的"专长"进入任务
- 可以研究人们如何利用专长差异

**Suparna 的补充**：
- 可以在协作前收集个体信心评分
- 这样可以知道"自然"的信心水平
- 然后观察社会压力下的校准

### 会议结论

1. **下一步**：每人提出实验设计建议，放入共享 Google Doc
2. **研究方向**：从"是否有群体增益"转向"人们如何使用社会信息"
3. **方法论**：先做静态规则比较，暂不做学习和 Output Interference

### 各参与者观点与性格

**Rich Shiffrin**：
- 审慎，不轻易相信直觉，坚持用模拟验证
- 系统性，强调一次只改变一个因素
- 开放，愿意改变研究方向

**Tim Pleskac**：
- 理论驱动，总是从模型预测出发
- 桥梁角色，连接不同领域
- 务实，关注"下一步做什么"

**Steve Clark**：
- 实验主义者，亲自运行实验、观察被试
- 细节导向，注意到信心校准的社会动态
- 创新，提出"分开学习、一起测试"的设计

**Suparna Rajaram**：
- 综合者，能把不同观点整合成连贯框架
- 好奇，想知道"人们到底在做什么"
- 社会敏感，注意到"不想显得像混蛋"的社会因素

---

# 项目管理

## 项目文件结构

```
Collective_Memory/
├── CLAUDE.md                         # 本文档（项目知识库）
│
├── Simulations/                      # 分支 1 & 2: REM 模拟
│   ├── src/
│   │   ├── rem_core.py              # REM 引擎（READ-ONLY）
│   │   ├── group_rules.py           # 7 种群体决策规则
│   │   └── run_simulation.py        # 参数扫描主程序
│   ├── notebooks/
│   │   └── bahrami_sweep_demo.ipynb # Jupyter 包装器
│   └── outputs/
│       ├── bahrami_sweep_final.csv
│       ├── miscalibration_sweep.csv
│       ├── rich_conflict_results.csv
│       └── *.png                     # 可视化图表
│
├── IRB/                              # 分支 3: IRB 文档
│   ├── Protocols1.pdf               # IRB 反馈（只读）
│   ├── Protocols2.pdf
│   ├── Protocols3.pdf
│   ├── *_REVISED.docx               # 修订后的文档（6个）
│   └── IRB_REVISION_INSTRUCTIONS.txt
│
├── papers/                           # 相关论文 PDF
│   ├── shiffrin_steyvers_1997_REM.pdf
│   ├── ernst_banks_2002_nature.pdf
│   ├── bahrami_2010_science.pdf
│   └── ...
│
├── data/                             # 模拟数据归档
├── experiments/                      # 实验设计文档
└── docs/                             # 其他文档
    ├── meeting_notes/
    └── email_threads/

/Users/yiytan/memory-game/            # 分支 4: 在线实验系统（独立仓库）
├── src/
│   └── App.js                        # 核心实验逻辑（1481 行）
├── build/                            # 生产构建
├── package.json
└── firebase.json
```

## 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| **REM 相关** | | |
| REM | Retrieving Effectively from Memory | Rich Shiffrin 的识别记忆模型 |
| Odds | 熟悉度比 | P(Old\|Data) / P(New\|Data) |
| Log-odds | 对数优势比 | ln(Odds)，REM 输出的自然形式 |
| **SDT 相关** | | |
| d' | d-prime | 敏感度指标（信号-噪音距离） |
| HR | Hit Rate | P(say "Old" \| truly Old) |
| FAR | False Alarm Rate | P(say "Old" \| truly New) |
| **决策规则（基础）** | | |
| CF | Coin Flip | 分歧时随机决策 |
| UW | Uniform Weighting | 原始 odds 算术平均 |
| DMC | Defer to Max Confidence | 听信心最大的人 |
| DSS | Direct Signal Sharing | log-odds 相加（理论上限） |
| BF | Behavior & Feedback | 基于个体历史正确率学习 |
| **决策规则（扩展）** | | |
| WCS | Weighted Confidence Sharing | 按信心加权整合（Bahrami 原版） |
| UW_Miscal | UW + Miscalibration | UW（算术平均 w）+ Prelec 权重函数 |
| DMC_Miscal | DMC + Miscalibration | DMC + Prelec 权重函数（max |w - 0.5|）|
| **理论概念** | | |
| CBR | Collective Benefit Ratio | d'_team / max(d'_A, d'_B) |
| Prelec weighting | Prelec 概率权重 | w(p) = exp(-β(-ln p)^α) |
| α (alpha) | 校准参数 | α=1 校准，α>1 过度自信，α<1 不足自信 |
| Orthogonal Sum | 正交和 | d'_optimal = √(d'_A² + d'_B²) |
| MLE | Maximum Likelihood Estimation | 最大似然估计 |
| **其他** | | |
| OI | Output Interference | 提取导致的干扰效应 |
| Hautus Correction | Hautus 修正 | 避免 HR=1 或 FAR=0 的 d' 计算方法 |
| SDT | Signal Detection Theory | 信号检测理论 |
| Umbrella Protocol | 伞状协议 | IRB 策略：通用框架，保留灵活性 |

## 常用命令

### REM 模拟（分支 1 & 2）

```bash
# 进入模拟目录
cd /Users/yiytan/Collective_Memory/Simulations/

# 运行参数扫描
python run_simulation.py

# 或使用 Jupyter Notebook
jupyter notebook bahrami_sweep_demo.ipynb

# 交互式运行
python
>>> import run_simulation
>>> df_bahrami = run_simulation.run_bahrami_sweep()
>>> df_miscal = run_simulation.run_miscalibration_sweep()

# 查看输出文件
ls -lh outputs/*.csv outputs/*.png
```

### 在线实验（分支 4）

```bash
# 进入实验目录
cd /Users/yiytan/memory-game

# 本地开发
npm start

# 构建和部署
npm run build
firebase deploy

# 查看部署状态
firebase hosting:channel:list
```

## 更新日志

- **2025年1月24日**：
  - ✅ 修复 Miscalibration Sweep 重大 Bug（traces frozen 问题）
  - ✅ 重命名 WCS_Miscal → UW_Miscal（符合 Tim 的模型语义）
  - ✅ 整合历史日志文件到 CLAUDE.md
  - ✅ 添加 "Debug 日志与经验教训" 章节
  - ✅ 添加 Monte Carlo 设计原则文档

- **2025年1月23日**：
  - ✅ 修复 RNG 交叉污染 Bug
  - ✅ 修复分母不稳定 Bug（测试项目变化）
  - ✅ 发现并修复 Over-Correction Bug（traces 过度冻结）

- **2025年1月19日**：
  - ✅ 重组 CLAUDE.md 为模块化结构（分支 1-4 独立）
  - ✅ 添加在线实验系统文档（分支 4）
  - ✅ 完成 IRB Protocol #29910 修订（分支 3）

- **2025年1月18日**：
  - ✅ 完成 Rich's Conflict Resolution Model 实现（分支 1）
  - ✅ 完成 Bahrami Parameter Sweep 实现（分支 2，含 3 个分析模块）
  - ✅ 目录重组（创建 src/, notebooks/, outputs/, archive/）

- **2025年1月14日**：
  - ✅ 初始文档创建，整合项目背景和理论框架

---

## Debug 日志与经验教训

> **重要规则**：所有 debug 日志、经验教训、Bug 修复记录都应记录在本文件中。**不要创建新的 .md 或 .txt 日志文件。**

### 2025-01-24：Miscalibration Sweep 命名修正与 Monte Carlo 修复

#### 问题诊断

1. **图表混乱**：DSS、WCS_Miscal、DMC_Miscal 三条曲线几乎完全重叠
2. **DSS Variance = 0**：代码 over-freeze 了 traces，导致 Monte Carlo 失效
3. **命名错误**：WCS_Miscal 实际上是 UW + Prelec，不是 Bahrami 的 WCS

#### Tim 的模型意图对齐

根据 Tim 的邮件和说明，核心意图是：

1. **Prelec 只作用在 "confidence as probability judgment" 上**
   - REM → odds φ → p = φ/(1+φ)
   - p → w(p; α)
   - α 只表示主观校准偏差

2. **"Replace UW Model" 的含义**
   - 把原来 UW（算术平均 p）的 aggregation，换成"算术平均 w(p)"
   - **这是 UW + Prelec，不是 Bahrami 的 WCS**

3. **"Replace Defer-to-Max" 的含义**
   - 决策仍是 selection
   - 但 max 的不是原始 odds，而是 subjective confidence |w - 0.5|

4. **DSS 的角色**
   - DSS 只是 baseline/ceiling
   - DSS 不使用 confidence，不参与 Prelec
   - **不需要过度关注 DSS 的行为**

#### 核心修复

1. **traces 移回循环内**：每个 α_B 条件使用不同的 traces（不同的大脑）
2. **seed 间隔增大**：用 `idx * 100` 而不是 `+ idx`，避免相邻条件相关
3. **重命名**：WCS_Miscal → UW_Miscal
4. **添加 Monte Carlo 重复**：n_reps = 20，稳定期望

#### Monte Carlo 设计原则

> **Monte Carlo is used to stabilize expectations, not for inference.**
>
> Tim 没有要求 error bars、标准误、或统计推断。Monte Carlo 只是 implementation detail，用于防止单次 realization 的假象。

**固定 vs 变化的组件**：

| 组件 | 是否固定 | 原因 |
|------|---------|------|
| test_items | ✅ 固定 | 控制任务难度（公平的考试） |
| labels | ✅ 固定 | 控制 base rate |
| traces | ❌ 每轮变化 | 表示被试内编码噪声（不同的大脑） |
| L_A, L_B | ❌ 每轮重算 | 依赖于 traces |
| α_B | 条件变量 | 只影响 Prelec subjective mapping |

**"Exam vs Brain" 比喻**：
> - **test_items = The exam** → 固定（每个人考同一张试卷）
> - **traces = The brain** → 变化（不同被试有不同的编码噪声）

**目标是 "Stable Mean + Natural Variance"**

#### 三个边界声明（防止解释翻车）

1. **关于 DSS**：DSS is used only as a reference baseline. Its variance is a diagnostic for frozen traces, not a modeling target.

2. **关于 Monte Carlo**：Monte Carlo repetitions are internal implementation details used to stabilize expectations, not an experimental dimension.

3. **关于展示**：Primary figures for discussion will show **Monte Carlo–averaged trends** across α_B; single-realization plots are used only for internal diagnostics. Use "shaded variability bands" instead of "error bars" to avoid triggering inferential interpretations.

#### 关键经验教训

| 原则 | 正确做法 | 错误做法 |
|------|---------|---------|
| Monte Carlo 定位 | 用于稳定期望，不是推断 | 当成统计检验工具 |
| DSS Variance | 原则判断（> 0 即可） | 设数值门槛（如 ~0.001） |
| d_best | per-realization 参考 | 跨 rep 平均 |
| 展示策略 | MC-averaged trends | 单次 noisy curve |
| 展示用语 | "shaded variability bands" | "error bars"（触发推断联想） |

---

### 2025-01-23：Prelec & Bahrami Sweep Bug 修复

#### Bug #1: RNG 交叉污染

**问题**：单个 `rng_cf` 被 4 个规则顺序消费，造成规则之间的虚假依赖。

**证据**：DSS variance 跨 α_B 为 0.1483，理论上应接近零（DSS 在数学上与 α 无关）。

**影响**：本应独立的规则通过共享 RNG 状态产生了相关性。

**修复**：为每个规则创建独立的 RNG。

```python
# 修复后：每个规则独立 RNG
rng_uw = np.random.default_rng(condition_seed + 5000)
rng_dmc = np.random.default_rng(condition_seed + 6000)
rng_dss = np.random.default_rng(condition_seed + 7000)
rng_cf = np.random.default_rng(condition_seed + 8000)
```

#### Bug #2: 分母不稳定

**问题**：测试项目（有时包括 traces）在每次 sweep 迭代中重新生成，导致 d_best 人为变化。

**证据**：
- d_best 标准差：0.0326（跨 α_B 迭代）
- 在特定 α_B 值（1.2, 1.5）出现尖峰

**影响**：CBR 变化是由于分母波动，而非真正的性能差异。

**修复**：测试项目在循环外生成一次（固定考试）。

#### Bug #3: Over-Correction（过度修正）

**问题**：早晨的 bug fix 把 traces 也移到了循环外，导致 DSS Variance = 0.0（完全冻结）。

**证据**：DSS 变成完美平坦的"死线"，没有任何 Monte Carlo 噪声。

**影响**：这不是 Monte Carlo，是 deterministic sensitivity analysis。

**修复**：traces 移回循环内，但测试项目保持在循环外。

#### 验证结果

**修复前后对比**：

| 指标 | 修复前 (Buggy) | Over-Frozen | 修复后 (Correct) | 状态 |
|------|---------------|-------------|------------------|------|
| DSS Variance | 0.148 | 0.000 | ~0.0145 | ✅ |
| d_best Stability | 变化 | 常数 | 略有变化（自然） | ✅ |
| RNG Independence | 共享 | 独立 | 独立 | ✅ |
| Monte Carlo | 破坏 | 停止 | 活跃 | ✅ |

---

### 2025-01-18：目录重组

**执行的操作**：

1. ✅ 创建新目录结构：src/, notebooks/, outputs/, archive/
2. ✅ 移动核心代码到 src/：rem_core.py, group_rules.py, run_simulation.py
3. ✅ 移动 notebooks 和导出：bahrami_sweep_demo.ipynb → notebooks/
4. ✅ 归档遗留文件：legacy_code/, legacy_results/, legacy_docs/
5. ✅ 清理系统文件：删除 .DS_Store 和 __pycache__/
6. ✅ 创建 .gitignore
7. ✅ 更新所有代码路径

**新结构**：
```
Simulations/
├── src/                    # 3 core Python files
├── notebooks/              # 1 notebook + 3 exports
├── outputs/               # 5 current result files
├── archive/               # 14 legacy files (organized)
├── README.md              # Updated with usage instructions
└── .gitignore            # Git configuration
```

---

### 2025-01-24：高性能重构经验教训

#### 实际完成的优化

1. **"Compute Once, Transform Many" 模式** ✅ 已合并
   - 核心洞察：α_B 只影响 Prelec 变换，不影响 REM 证据 (L_A, L_B)
   - 循环结构翻转：`rep 外层 → α_B 内层`
   - REM 调用从 220 次 → 20 次 = **11x 加速**
   - 这是主要性能收益来源

2. **REM Trace-level 向量化**（当前实验采用，experiment-specific）
   - `compute_log_odds_vectorized` 移除了 trace-level Python loop
   - 所有 trace likelihood 被压进 NumPy matrix + logsumexp
   - 公式严格复刻原实现：`λ_v = (c + (1 - c) * P_v) / P_v`，其中 `P_v = g(1-g)^(v-1)`
   - **数值验证**：在当前参数区间内，max diff = 4.44e-16（机器精度）

   **⚠️ Fact vs Norm 分离**：
   - **Fact（当前实验做了什么）**：当前 miscalibration sweep 实验使用 trace-level 向量化，已在该实验的参数区间内通过数值验证
   - **Norm（未来实验的默认选择）**：Trace-level 向量化**不是**推荐的 REM 设计模式。当前实现是 **"Verified exception for this experiment, not a general design pattern"**

3. **诊断检查修正** ✅
   - DSS 跨 α_B 的方差设计上为 0（DSS 不依赖 α）
   - 正确检查：`cbr_std`（Monte Carlo 跨 rep 的标准差）

#### 关键安全约束

1. **决策规则 RNG 作用域**（硬性约束）
   - **DMC tie-breaking**：使用专用 `rng_dmc_tie = np.random.default_rng(rep_seed + 999)`
   - **CF (Coin Flip)**：使用专用 `rng_cf = np.random.default_rng(rep_seed + 555)`
   - **两者都必须在 α_B 循环外、per Monte Carlo rep 创建一次**
   - **跨所有 α_B 值复用同一个 RNG 实例**
   - **禁止在 α_B 循环内实例化任何决策相关的 RNG**
   - 决策随机性是 rep-level only，与 α_B 无关。任何偏离都是建模错误。

2. **`np.allclose` 使用规则**（硬性分离）
   - ✅ **允许**：函数级数值等价检查（如 scalar `compute_log_odds` vs vectorized 实现）
   - ❌ **严格禁止**用于：
     - Monte Carlo 输出
     - 聚合指标
     - 任何 ratio 类指标（如 d'_team / d'_best）
   - Ratio 指标本质上有 Monte Carlo 噪声，逐点等价检查会 debug 不存在的 bug

#### 红线声明（未来修改必读）

当前 **miscalibration sweep 实验**使用了 trace-level 向量化的 REM likelihood 计算，作为**该实验验证过的例外**，而非推荐的通用 REM 实现模式。

这已在该实验的参数区间内通过数值验证，但：

**红线 1 - 向量化边界**：
- 未来修改**不得假设** trace-level 向量化是普遍安全的
- 任何修改**必须**重新对照原始 scalar 实现验证
- **特别关注 edge trials 和 conflict trials**——向量化错误最可能在这些 trial 上暴露

**红线 2 - RNG 作用域**：
- DMC tie-breaking：`rng_dmc_tie = np.random.default_rng(rep_seed + 999)`
- CF (Coin Flip)：`rng_cf = np.random.default_rng(rep_seed + 555)`
- 两者都必须在 α_B 循环外创建，禁止在 α 循环内实例化

**红线 3 - 验证策略**：
- `np.allclose` 仅用于函数级数值等价检查
- 禁止对 Monte Carlo 输出或 ratio 指标使用逐点等价检查

这不是关于代码速度，而是关于科学可解释性。

---

**🔒 优化阶段关闭**

Optimization phase closed for the current miscalibration sweep experiment.
Trace-level vectorization is a verified exception for this specific experiment, not a general REM design pattern.
Future experiments or parameter regions require scientific justification and re-verification against the scalar implementation.

不引入进一步重构或优化，不重新审视向量化决策，仅执行科学分析。

#### 最终性能

| 指标 | 优化前 | 优化后 | 加速 |
|------|--------|--------|------|
| 总运行时间 | ~8 分钟 | ~12 秒 | **40x** |
| `compute_log_odds` 单次 | 1.7ms | 0.075ms | **22x** |
| REM 调用次数 | 220 | 20 | **11x** |

---

*最后更新：2025年1月24日*
