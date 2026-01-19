# Collective_Memory 项目

## 项目概述

本项目研究 **群体记忆（Collective Memory）** 中的识别判断机制，核心目标是：

> **用 REM 模型生成个体内部证据，再用不同的社会决策规则（CF / UM / WCS / DSS）将个体证据整合为群体判断。**

这是一个跨学科研究项目，整合了：
- **认知心理学**：REM 识别记忆模型
- **决策科学**：贝叶斯最优整合理论
- **社会心理学**：群体决策与协作记忆

### 研究团队

| 成员 | 单位 | 专长 |
|------|------|------|
| **Rich Shiffrin** | Indiana University | REM 模型创始人、识别记忆理论 |
| **Tim Pleskac** | UCR | 决策建模、贝叶斯推理 |
| **Steve Clark** | UCR | 识别记忆实验、信号检测论 |
| **Suparna Rajaram** | Stony Brook | 协作记忆、群体抑制效应 |
| **Yiyan (你)** | UCR | 博士生、REM 模拟实现 |

---

## 核心研究问题

### 主要问题

**当多个个体各自拥有对同一项目的记忆证据时，群体如何整合这些证据形成集体判断？**

### 子问题

1. **规范性问题**：什么是"最优"的群体整合方式？（贝叶斯上限）
2. **描述性问题**：人们实际上如何整合群体信息？
3. **机制问题**：不同整合规则在什么条件下表现更好/更差？
4. **社会因素**：信心交流、专长差异、社会压力如何影响群体判断？

---

## 理论框架：三条理论线索的整合

### 线索一：REM 模型（个体层）

**来源**：Shiffrin & Steyvers (1997)

**核心机制**：REM 为每个个体、每个测试项目计算一个 **熟悉度比值（Odds）**：

$$\text{Odds} = \frac{P(\text{Old} \mid D)}{P(\text{New} \mid D)}$$

**完整推导公式**：

$$\frac{P(O \mid D)}{P(N \mid D)} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{c + (1-c)g}{g}\right)^{K_M^i} (1-c)^{K_N^i}$$

其中：
- $K_M^i$：probe 与第 i 条记忆痕迹**匹配**的特征数
- $K_N^i$：**不匹配**的特征数
- $c$：学习准确度（encoding accuracy）
- $g$：猜中特征的基础概率

**关键洞见**：

> REM 的输出不是 yes/no 二值决策，而是**连续的证据强度**。
> 这正是 WCS / DSS 能够"接上"的关键！

**判断准则**：
- Odds > 1 → 判断 "Old"
- Odds < 1 → 判断 "New"
- 准则 = 1 是 REM 推导自然给出的，**不需要人为设定**

---

### 线索二：Ernst & Banks (2002)（规范上限）

**来源**：Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. *Nature*, 415, 429-433.

**核心理论**：跨感觉线索的最优整合方式是**按可靠性（1/方差）加权平均**。

**原文公式（Maximum Likelihood Estimation）**：

$$\hat{S} = \sum_i w_i \hat{S}_i \quad \text{where} \quad w_i = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$$

**整合后方差**：

$$\sigma^2_{\text{combined}} = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}$$

**理论意义**：

> Ernst 的框架是 DSS（直接信号共享）的规范基础：
> 如果群体成员能直接共享内部证据，最优整合方式就是可靠性加权。

---

### 线索三：Bahrami et al. (2010)（社会贝叶斯）

**来源**：Bahrami, B., et al. (2010). Optimally interacting minds. *Science*, 329, 1081-1085.

**核心命题**（原文）：

> "These patterns can be explained by a model in which two heads are Bayes optimal under the assumption that individuals accurately communicate their level of confidence."

**四种社会决策规则**：

| 模型 | 全称 | 机制描述 | 与 REM 的关系 |
|------|------|----------|---------------|
| **CF** | Coin Flip | 分歧时随机选择 | 完全忽略 REM 证据 |
| **UM** | Uniform/Majority | 多数投票（少数服从多数） | REM odds → 二值决策 → 投票 |
| **WCS** | Weighted Confidence Sharing | 按信心加权平均 | REM odds ≈ confidence |
| **DSS** | Direct Signal Sharing | 直接整合内部证据 | REM odds 直接相乘/log相加 |

**Bahrami 的关键发现**：

1. **相似性效应**：当群体成员能力相似时，群体表现最优
2. **信心交流**：准确的信心交流是达到贝叶斯最优的关键
3. **上限与现实**：DSS 是理论上限，但人类无法直接共享内部信号

---

### 线索四：Enright et al. (2020)（基线模型）

**来源**：Enright, M., et al. (2020). 群体识别记忆研究（具体引用待补充）

**Uniform Weighting (UW) 模型**：

$$d'_{\text{team}} = \frac{\sum_i d'_i}{\sqrt{m}}$$

**与 REM 的对应关系**：

| SDT 概念 | REM 对应 |
|----------|----------|
| d' | log(Odds) 的区分度 |
| yes/no | Odds > 1 |

**理论意义**：

> UM / Majority 在 REM 框架下是**必要的 baseline**，
> 用来检验群体是否"超越简单统计聚合"。

---

## 当前模拟任务（Current Simulation Task）

### 任务定义

**一句话定义（可直接用于 dissertation/proposal）**：

> **English**: "This simulation uses the REM model to generate individual recognition evidence (odds), and then applies alternative social decision rules (CF, UM, WCS, DSS) to map individual evidence onto group-level recognition judgments, following Bayesian and signal-detection–theoretic frameworks of collective decision-making."

> **中文**: "该模拟使用 REM 模型生成个体的识别证据（熟悉度比），并在此基础上引入不同的社会决策规则（CF、UM、WCS、DSS），将个体证据映射为群体识别判断，从而检验不同群体整合机制在规范与非规范条件下的表现。"

### 三层模拟结构

```
┌─────────────────────────────────────────────────────────┐
│                    第一层：个体层（REM）                   │
├─────────────────────────────────────────────────────────┤
│  输入：study list + test probe                           │
│  输出：每个个体对每个 item 的 odds / λ                    │
│  ✓ REM 只负责生成证据，不负责社会整合                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              第二层：群体规则层（Social Decision Rule）     │
├─────────────────────────────────────────────────────────┤
│  规则选项：CF / UM / WCS / DSS                           │
│  输入：个体 odds 或个体决策                               │
│  输出：群体 odds 或群体决策                               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                第三层：评价层（Performance Metrics）       │
├─────────────────────────────────────────────────────────┤
│  分开报告：Targets（真旧词）vs Foils（真新词）             │
│  指标：Hit Rate, FA Rate, Accuracy, d', AUC              │
└─────────────────────────────────────────────────────────┘
```

### 四种规则的具体实现

#### 方法一：平均熟悉度（Average the Odds）

```python
def average_odds(individual_odds):
    """
    把所有人的 Odds 取平均，平均值 > 1 判断 Old
    """
    group_odds = np.mean(individual_odds)
    return "Old" if group_odds > 1.0 else "New"
```

**示例**：
- 成员 A: Odds = 1.5
- 成员 B: Odds = 0.8
- 成员 C: Odds = 1.3
- 平均 = (1.5 + 0.8 + 1.3) / 3 = 1.2 > 1.0 → **Old**

#### 方法二：多数投票（Majority Rules / UM）

```python
def majority_vote(individual_odds):
    """
    每个人先做二值决策，然后多数投票
    """
    decisions = ["Old" if odds > 1.0 else "New" for odds in individual_odds]
    old_votes = decisions.count("Old")
    new_votes = decisions.count("New")
    return "Old" if old_votes > new_votes else "New"
```

**示例**：
- 成员 A: Odds = 1.5 → Old
- 成员 B: Odds = 0.8 → New
- 成员 C: Odds = 1.3 → Old
- 投票：2 Old vs 1 New → **Old**

#### 方法三：听最自信的（Defer to Largest Odds）

```python
def defer_to_most_confident(individual_odds):
    """
    找 Odds 最高的人，听他的决策
    """
    max_odds = max(individual_odds)
    return "Old" if max_odds > 1.0 else "New"
```

**注意**：这里有一个细微问题——如果最高 Odds 仍然 < 1，应该判断 New。

#### 方法四：DSS（Direct Signal Sharing）

```python
def direct_signal_sharing(individual_odds):
    """
    理论上限：log-odds 相加（等价于 odds 相乘）
    """
    log_odds = [np.log(odds) for odds in individual_odds]
    group_log_odds = sum(log_odds)
    group_odds = np.exp(group_log_odds)
    return "Old" if group_odds > 1.0 else "New"
```

**数学等价**：
$$\text{Odds}_{\text{group}} = \prod_i \text{Odds}_i$$
$$\log \text{Odds}_{\text{group}} = \sum_i \log \text{Odds}_i$$

---

## 已完成的实现（2025年1月）

### 实现概述

成功完成了 Bahrami Parameter Sweep 模拟框架，包含三个独立的分析模块：

1. **Part 1: Bahrami Parameter Sweep**（Tim's Analysis）
   - 比较 5 种决策规则（CF, UW, DMC, DSS, BF）
   - 固定专家能力（c_A = 0.7），扫描新手能力（c_B = 0.1 to 0.9）
   - 核心指标：Collective Benefit Ratio = d'_team / max(d'_A, d'_B)

2. **Part 2: Theoretical Verification**（Rich's Request）
   - 验证 DSS 是否恢复 SDT 理论预测
   - 理论上限：d'_theory = √(d'_A² + d'_B²)（正交和）
   - 定量比较模拟 vs 理论

3. **Part 3: Confidence Miscalibration**（Prelec Weighting）
   - 研究信心校准偏差对群体决策的影响
   - 使用 Prelec 概率权重函数
   - 测试 4 种模型：WCS_Miscal, DMC_Miscal, DSS, CF

### 核心文件

#### 1. `group_rules.py`（449 行）

**功能**：实现所有群体决策规则，统一返回 d' 指标

**已实现的规则**（7个）：

```python
# 基础规则（5个）
1. coin_flip_rule()           # CF - 分歧时随机
2. uniform_weighting_rule()   # UW - 原始 odds 算术平均
3. defer_to_max_confidence()  # DMC - 听最自信的
4. direct_signal_sharing()    # DSS - log-odds 相加（贝叶斯最优）
5. behavior_feedback_rule()   # BF - 基于个体正确率学习

# 信心校准规则（2个）
6. wcs_miscal_rule()          # WCS + Prelec weighting
7. dmc_miscal_rule()          # DMC + Prelec weighting
```

**关键设计决策**：

1. **UW Rule - 原始 Odds 平均**（非 log 空间）
   ```python
   odds_A = np.exp(L_A)
   odds_B = np.exp(L_B)
   mean_odds = (odds_A + odds_B) / 2
   team_decision = (mean_odds > 1).astype(int)
   ```

2. **BF Rule - 个体学习**（CRITICAL）
   - 分数更新基于**个体正确性**，非群体决策正确性
   - Trial-by-trial 顺序处理
   ```python
   if D_A[i] == labels[i]: score_A += 1
   if D_B[i] == labels[i]: score_B += 1
   ```

3. **Prelec Weighting Function**
   ```python
   def prelec_weighting(L, alpha):
       p = 1.0 / (1.0 + np.exp(-L))  # sigmoid
       beta = np.power(np.log(2), 1 - alpha)  # 确保 w(0.5) = 0.5
       w = np.exp(-beta * np.power(-np.log(p), alpha))
       return w
   ```

4. **DMC Miscalibration - 信心定义**
   - 信心 = 距离中点的距离：`conf = |w - 0.5|`（非原始 w）

**统一返回格式**：
```python
{
    'dprime_A': float,
    'dprime_B': float,
    'dprime_team': float
}
```

#### 2. `run_simulation.py`（576 行）

**功能**：三个独立的参数扫描函数 + 可视化

**主要函数**：

```python
1. run_bahrami_sweep()
   - 固定：c_A = 0.7
   - 扫描：c_B ∈ [0.1, 0.9], step 0.1
   - 规则：CF, UW, DMC, DSS, BF
   - 输出：bahrami_sweep_final.csv, bahrami_sweep_plot.png

2. run_miscalibration_sweep()
   - 固定：c_A = c_B = 0.7, alpha_A = 1.2
   - 扫描：alpha_B ∈ [0.5, 1.5], step 0.1
   - 规则：WCS_Miscal, DMC_Miscal, DSS, CF
   - 输出：miscalibration_sweep.csv, miscalibration_plot.png

3. create_bahrami_plot()
   - 5条曲线 + y=1.0 参考线
   - X轴：Agent B Ability (c_B)
   - Y轴：Collective Benefit Ratio

4. create_rich_verification_plot()
   - DSS (simulated) vs Theory 对比
   - 左图：d'_team 绝对值
   - 右图：Collective Benefit Ratio

5. create_miscalibration_plot()
   - 4条曲线（2 miscal + 2 baseline）
   - X轴：Agent B Miscalibration (α_B)
   - Y轴：Collective Benefit Ratio
```

**关键设计**：
- 每个扫描独立初始化 RNGs（确保条件间独立）
- REM 参数固定：w=20, g=0.4, u=0.04, nSteps=5
- 使用 Hautus correction 计算 d'

#### 3. `bahrami_sweep_demo.ipynb`

**功能**：Jupyter notebook 包装器，三个独立分析部分

**结构**：
- **Part 1**：运行 Bahrami sweep，展示 5 规则比较
- **Part 2**：展示 Rich's 理论验证（DSS vs SDT prediction）
- **Part 3**：运行 miscalibration sweep，分析 Prelec weighting 效应

**每部分独立**：
- 使用不同变量名（`results`, `results_miscal`）
- 避免交叉污染
- 独立的可视化输出

### 理论创新点

#### 1. Prelec 概率权重函数

**数学形式**：
$$w(p) = \exp(-\beta \cdot (-\ln p)^\alpha)$$

**约束条件**：
$$\beta = (\ln 2)^{1-\alpha} \quad \Rightarrow \quad w(0.5) = 0.5$$

**参数解释**：
- α = 1：完美校准（w = p）
- α > 1：过度自信（极端值被夸大）
- α < 1：不够自信（极端值被压缩）

**实现细节**（LOCKED SEMANTICS）：
1. 输入：Log-odds L（非概率）
2. 转换：p = 1/(1 + exp(-L))
3. 应用 Prelec：w(p)
4. 返回：主观权重 w ∈ (0,1)

#### 2. 正交和理论（Orthogonal Sum）

**SDT 预测**（独立噪声假设）：
$$d'_{\text{optimal}} = \sqrt{d'_A^2 + d'_B^2}$$

**验证目标**：
- 检验 DSS（REM 模拟）是否恢复该上限
- 定量测量偏差：|d'_DSS - d'_theory|

#### 3. Collective Benefit Ratio

**定义**：
$$\text{CBR} = \frac{d'_{\text{team}}}{max(d'_A, d'_B)}$$

**解释**：
- CBR > 1：群体优于最佳个体（集体增益）
- CBR = 1：群体等于最佳个体
- CBR < 1：群体劣于最佳个体（集体损失）

### 实验设计矩阵

| 实验 | 固定参数 | 扫描参数 | 测试规则 | 研究问题 |
|------|----------|----------|----------|----------|
| Bahrami Sweep | c_A = 0.7 | c_B ∈ [0.1, 0.9] | CF, UW, DMC, DSS, BF | 能力异质性如何影响群体增益？ |
| Rich Verification | c_A = 0.7 | c_B ∈ [0.1, 0.9] | DSS only | DSS 是否达到理论上限？ |
| Miscalibration | c_A = c_B = 0.7, α_A = 1.2 | α_B ∈ [0.5, 1.5] | WCS_Miscal, DMC_Miscal, DSS, CF | 信心校准偏差如何影响整合？ |

### 输出文件

```
Simulations/
├── bahrami_sweep_final.csv       # 45 rows (9 c_B × 5 rules)
│   └── Columns: c_A, c_B, rule, dprime_A, dprime_B, dprime_team,
│                d_best, collective_benefit_ratio, dprime_theory, ratio_theory
│
├── miscalibration_sweep.csv      # 44 rows (11 α_B × 4 models)
│   └── Columns: alpha_A, alpha_B, model, dprime_A, dprime_B, dprime_team,
│                d_best, collective_benefit_ratio
│
├── bahrami_sweep_plot.png        # 5-curve comparison
├── rich_theory_verification.png  # DSS vs theory (2 subplots)
└── miscalibration_plot.png       # 4-model comparison
```

### 关键技术细节

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

#### D' 计算（Hautus Correction）

```python
HR = (hits + 0.5) / (S + 1)
FAR = (fas + 0.5) / (N + 1)
dprime = stats.norm.ppf(HR) - stats.norm.ppf(FAR)
```

### 验证清单

完成实现后验证的关键点：

- ✅ UW 计算原始 odds 平均（非 log 空间）
- ✅ BF 基于个体正确性更新分数
- ✅ 所有规则返回 d' 指标
- ✅ Bahrami sweep 生成 45 行（9 × 5）
- ✅ Prelec beta 确保 w(0.5) = 0.5
- ✅ DMC_Miscal 使用 |w - 0.5| 作为信心
- ✅ 三个分析部分完全独立
- ✅ CSV 包含所有必需列

---

## 导师指导记录

### Tim Pleskac 的邮件（模型空间）

Tim 建议系统比较以下模型：

> "I think it is worth considering the Bahrami models in particular the WCS and DSS models and formulate them with REM."

具体规则定义：
1. **CF (Coin Flip)**：分歧时随机
2. **BF (Behavior & Feedback)**：根据历史正确率决定听谁的
3. **WCS (Weighted Confidence Sharing)**：按信心加权
4. **DSS (Direct Signal Sharing)**：直接整合内部证据（理论上限）

### Rich Shiffrin 的邮件（方法论约束）

Rich 的关键判断：

> "BF needs analysis as testing continues, and involves output interference (OI). If we use a model without OI then what is learned will not likely be interpretable..."

**Rich 的指令解读**：

| 现在做 | 暂时不做 |
|--------|----------|
| ✅ CF / UM / WCS / DSS 静态规则比较 | ❌ Output Interference (OI) |
| ✅ 用 REM 生成个体证据 | ❌ Behavior & Feedback (BF) 学习 |
| ✅ 分开报告 Targets vs Foils | ❌ 跨 trial 的动态变化 |

**原因**：OI 会导致识别性能随测试下降，BF 会导致性能上升，两者不分开会导致解释混乱。

---

## 会议记录摘要（2024年12月）

### 会议参与者

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
3. **方法论**：先做静态规则比较，暂不做学习和 OI

---

## 各参与者观点与性格分析

### Rich Shiffrin

**观点**：
- 强调 REM 的长尾分布使直觉预测变得困难
- 关注个体差异的巨大影响
- 建议暂时搁置复杂因素（OI、BF），先做简单模型
- 对"社会因素如何影响记忆决策"比"是否有群体增益"更感兴趣

**性格特点**：
- **审慎**：不轻易相信直觉，坚持用模拟验证
- **系统性**：强调一次只改变一个因素
- **开放**：愿意改变研究方向（从效应检测到机制探索）
- **幽默**：称 Yiyan 为"沉默的成员"

### Tim Pleskac

**观点**：
- 强调不同决策规则（Bahrami 模型）的理论区分
- 关注 REM 如何与社会决策模型整合
- 提出用认知模型预测信心（而不是直接测量）

**性格特点**：
- **理论驱动**：总是从模型预测出发
- **桥梁角色**：连接不同领域（决策科学 + 记忆）
- **务实**：关注"下一步做什么"
- **支持性**：帮助 Yiyan 联系导师、发送视频

### Steve Clark

**观点**：
- 提供实验细节和被试行为观察
- 提出创新的预曝光实验设计
- 发现信心评分可能使任务"过于简单"

**性格特点**：
- **实验主义者**：亲自运行实验、观察被试
- **细节导向**：注意到信心校准的社会动态
- **创新**：提出"分开学习、一起测试"的设计
- **实用**：关注实验的"面效度"

### Suparna Rajaram

**观点**：
- 从协作抑制文献出发理解群体效应
- 强调"under the hood"的机制问题
- 建议操纵记忆强度（而不只是信心）
- 关注社会压力对判断的影响

**性格特点**：
- **综合者**：能把不同观点整合成连贯框架
- **好奇**：想知道"人们到底在做什么"
- **社会敏感**：注意到"不想显得像混蛋"的社会因素
- **高效**：会议密集但仍参与讨论

---

## 项目文件结构

```
Collective_Memory/
├── CLAUDE.md                          # 本文档（项目知识库）
├── papers/                            # 相关论文 PDF
│   ├── shiffrin_steyvers_1997_REM.pdf
│   ├── ernst_banks_2002_nature.pdf
│   ├── bahrami_2010_science.pdf
│   └── enright_2020.pdf
├── Simulations/                       # 模拟代码和数据（当前工作目录）
│   ├── rem_core.py                    # REM 引擎（READ-ONLY）
│   ├── group_rules.py                 # 7 种群体决策规则
│   ├── run_simulation.py              # 参数扫描主程序
│   ├── bahrami_sweep_demo.ipynb       # Jupyter notebook 包装器
│   ├── bahrami_sweep_final.csv        # Bahrami sweep 输出
│   ├── miscalibration_sweep.csv       # Miscalibration sweep 输出
│   ├── bahrami_sweep_plot.png         # 5 规则比较图
│   ├── rich_theory_verification.png   # DSS vs theory 验证图
│   └── miscalibration_plot.png        # Prelec weighting 效应图
├── data/                              # 模拟数据归档
├── experiments/                       # 实验设计文档
└── docs/                              # 其他文档
    ├── meeting_notes/                 # 会议记录
    └── email_threads/                 # 邮件往来
```

---

## 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| REM | Retrieving Effectively from Memory | Rich Shiffrin 的识别记忆模型 |
| Odds | 熟悉度比 | P(Old\|Data) / P(New\|Data) |
| Log-odds | 对数优势比 | ln(Odds)，REM 输出的自然形式 |
| d' | d-prime | SDT 中的敏感度指标（信号-噪音距离） |
| **决策规则** | | |
| CF | Coin Flip | 分歧时随机决策 |
| UW | Uniform Weighting | 原始 odds 算术平均 |
| DMC | Defer to Max Confidence | 听信心最大的人 |
| DSS | Direct Signal Sharing | log-odds 相加（理论上限） |
| BF | Behavior & Feedback | 基于个体历史正确率学习 |
| WCS | Weighted Confidence Sharing | 按信心加权整合 |
| WCS_Miscal | WCS + Miscalibration | WCS + Prelec 权重函数 |
| DMC_Miscal | DMC + Miscalibration | DMC + Prelec 权重函数 |
| **理论概念** | | |
| CBR | Collective Benefit Ratio | d'_team / max(d'_A, d'_B) |
| Prelec weighting | Prelec 概率权重 | w(p) = exp(-β(-ln p)^α) |
| α (alpha) | 校准参数 | α=1 校准，α>1 过度自信，α<1 不足自信 |
| Orthogonal Sum | 正交和 | d'_optimal = √(d'_A² + d'_B²) |
| Hautus Correction | Hautus 修正 | 避免 HR=1 或FAR=0 的 d' 计算方法 |
| **其他** | | |
| OI | Output Interference | 提取导致的干扰效应 |
| MLE | Maximum Likelihood Estimation | 最大似然估计 |
| AUC | Area Under Curve | ROC 曲线下面积 |
| SDT | Signal Detection Theory | 信号检测理论 |

---

## 常用命令

```bash
# 进入模拟目录
cd /Users/yiytan/Collective_Memory/Simulations/

# 运行 Bahrami Parameter Sweep（Python 脚本）
python run_simulation.py

# 运行 Jupyter Notebook（推荐）
jupyter notebook bahrami_sweep_demo.ipynb

# 在 Python 中交互运行
python
>>> import run_simulation
>>> df_bahrami = run_simulation.run_bahrami_sweep()
>>> df_miscal = run_simulation.run_miscalibration_sweep()

# 查看输出文件
ls -lh *.csv *.png
```

---

## 下一步任务

### 当前阶段（已完成 ✅）

1. ✅ **Bahrami Parameter Sweep**
   - 实现 5 种决策规则（CF, UW, DMC, DSS, BF）
   - 完成能力异质性参数扫描（c_B 0.1-0.9）
   - 生成 Collective Benefit Ratio 分析

2. ✅ **Rich's 理论验证**
   - DSS vs SDT 正交和理论对比
   - 定量验证模拟准确性

3. ✅ **信心校准分析**
   - 实现 Prelec 概率权重函数
   - WCS_Miscal 和 DMC_Miscal 规则
   - α_B 参数扫描（0.5-1.5）

### 近期任务（分析当前结果）

1. **解读 Bahrami sweep 结果**
   - 哪些规则在何时产生集体增益（CBR > 1）？
   - UW 和 DMC 的表现差异说明了什么？
   - BF 学习效应有多强？

2. **验证理论符合度**
   - DSS 与理论上限的偏差大小
   - 偏差的系统性（是否随 c_B 变化）
   - REM 非正态性的影响

3. **分析信心校准效应**
   - 匹配校准（α_A = α_B = 1.2）vs 错配（α_A ≠ α_B）
   - WCS_Miscal vs DMC_Miscal 的性能差异
   - 信心校准对不同规则的影响程度

### 未来方向（扩展研究）

1. **变更能力差异设置**
   - 两个专家（c_A = c_B = 0.8）
   - 两个新手（c_A = c_B = 0.3）
   - 极端异质性（c_A = 0.9, c_B = 0.1）

2. **群体规模扩展**
   - N = 3, 4, 5 人群体
   - 多数投票的临界点
   - DSS 随 N 增长的收益

3. **动态过程研究**（需与 Rich 讨论）
   - 加入 Output Interference (OI)
   - BF 的跨 trial 学习曲线
   - 信心校准的动态调整

4. **实验验证**
   - 设计行为实验测试模型预测
   - Steve 的"无信心评分"实验设计
   - 预曝光专长操纵实验

---

## 关键公式速查

### REM Odds

$$\text{Odds} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{c + (1-c)g}{g}\right)^{K_M^i} (1-c)^{K_N^i}$$

### d' 计算（Hautus Correction）

$$\text{HR} = \frac{\text{hits} + 0.5}{S + 1}, \quad \text{FAR} = \frac{\text{fas} + 0.5}{N + 1}$$

$$d' = \Phi^{-1}(\text{HR}) - \Phi^{-1}(\text{FAR})$$

### Collective Benefit Ratio

$$\text{CBR} = \frac{d'_{\text{team}}}{\max(d'_A, d'_B)}$$

### DSS 在 REM 下的实现

$$\text{Odds}_{\text{group}} = \prod_i \text{Odds}_i$$

或等价地：

$$\log \text{Odds}_{\text{group}} = \sum_i \log \text{Odds}_i$$

### UW 规则（原始 Odds 平均）

$$\text{Odds}_{\text{group}} = \frac{1}{N} \sum_{i=1}^{N} \text{Odds}_i$$

### Prelec 概率权重函数

$$w(p) = \exp(-\beta \cdot (-\ln p)^\alpha)$$

$$\beta = (\ln 2)^{1-\alpha} \quad \text{（确保 } w(0.5) = 0.5 \text{）}$$

### 正交和理论（SDT 独立噪声）

$$d'_{\text{theory}} = \sqrt{d'_A^2 + d'_B^2}$$

### Ernst MLE 加权

$$w_i = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}$$

---

## 注意事项

### 理论约束

1. **REM 的角色**：REM 只是"证据生成器"，不负责社会整合
2. **DSS 是上限**：人类无法真正"直接共享内部信号"，但它提供理论基准
3. **个体差异很大**：83 名被试从随机到完美呈线性分布（Rich 的数据）

### 实现细节（CRITICAL）

4. **UW 规则必须用原始 Odds**：不是 log-space 平均，是 `(exp(L_A) + exp(L_B)) / 2`
5. **BF 规则学习机制**：分数更新基于**个体**正确性，非群体决策正确性
6. **Prelec 函数约束**：β 必须使 w(0.5) = 0.5，否则不是交叉点
7. **DMC_Miscal 信心定义**：conf = |w - 0.5|（距离中点），不是原始 w

### RNG 独立性

8. **条件间独立**：每个参数扫描点重新初始化 RNGs
9. **个体间独立**：Agent A 和 B 使用不同的 seed（+1000, +2000）
10. **刺激驱动共享**：study list 和 test list 用同一个 rng_test

### 文件依赖

11. **rem_core.py 是 READ-ONLY**：不修改 REM 引擎，只导入使用
12. **三个分析部分独立**：Bahrami, Rich's verification, Miscalibration 互不影响
13. **暂不做 OI**：Rich 明确建议先做静态规则，避免混淆因素

---

## 联系方式

- **主要导师**：Tim Pleskac（UCR）
- **REM 专家**：Rich Shiffrin（Indiana）
- **识别记忆**：Steve Clark（UCR）
- **协作记忆**：Suparna Rajaram（Stony Brook）

---

## 更新日志

- **2025年1月18日**：完成 Bahrami Parameter Sweep 实现（3个分析模块：Bahrami sweep, Rich's verification, Prelec miscalibration）
- **2025年1月14日**：初始文档创建，整合项目背景和理论框架

---

*最后更新：2025年1月18日*
