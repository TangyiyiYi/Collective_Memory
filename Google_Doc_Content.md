# Collective Memory Project: Simulation Progress Report
## 群体记忆项目：模拟进展汇报

**For:** Lab Members (Rich, Tim, Steve, Suparna + others)
**Date:** February 2026
**Author:** Yiyan Tan

---

## Executive Summary

本文档汇报 Collective Memory 项目的模拟进展。核心发现：

1. **群体决策可以超越最佳个体**：当使用 DSS（贝叶斯最优）、DMC 或 UW 规则时，Collective Benefit Ratio > 1
2. **群体优势主要来自 Hit Rate 的提升**：群体能"救回"个体漏掉的 Old 项目
3. **信心校准偏差会削弱群体优势**：但即使有 miscalibration，群体仍优于随机
4. **下一步需要确定实验操纵方案**：如何把模拟中的 c（编码质量）映射到实际实验

---

## 1. 研究背景 (Background)

### 1.1 核心问题

当两个人各自记住同一个项目时，群体如何整合这些证据形成判断？

**三个子问题：**
- **Process Gain**: 什么时候 group > best individual？
- **Process Loss**: 什么时候 group < best individual？
- **机制**: 不同决策规则在什么条件下更优/更差？

### 1.2 为什么现在汇报

- REM + 群体规则的核心模拟已完成
- 与 Tim 讨论了实验设计方案
- 需要让组内其他成员了解进展，以便下一步实验设计

### 1.3 模拟的基本假设

- **群体大小**: N = 2（两人一组）
- **判断独立性**: 两人独立做出判断，然后按规则整合
- **证据生成**: 使用 REM 模型生成个体 log-odds
- **任务类型**: Old/New 识别任务

---

## 2. REM 模型简介 (For Non-Modelers)

### 2.1 核心思想

REM（Retrieving Effectively from Memory）是 Rich Shiffrin 1997 年提出的识别记忆模型。

**直觉类比**: 想象你的记忆像一个嘈杂的图书馆。当你看到一个测试词，你在图书馆里搜索——如果找到很多相似的书，你会觉得"这词我见过"（熟悉度高）。

**模型输出**:
- **Odds** = P(Old | 证据) / P(New | 证据)
- Odds > 1 → 判断 "Old"（见过）
- Odds < 1 → 判断 "New"（没见过）

### 2.2 关键参数

**c = 编码质量**（encoding accuracy）
- c 高 → 学得好，记得清楚
- c 低 → 学得差，记忆模糊

**这是我们在实验中可能要操纵的变量。**

### 2.3 REM 的角色

REM 只是"证据生成器"——它产生个体的内部证据（log-odds），**不负责群体整合**。群体整合由下面的决策规则实现。

---

## 3. 五种群体决策规则 (Group Decision Rules)

| 规则 | 全称 | 机制 | 预期表现 |
|------|------|------|---------|
| **DSS** | Direct Signal Sharing | log-odds 直接相加 | 理论最优（贝叶斯上限） |
| **DMC** | Defer to Max Confidence | 听最自信的人 | 中等偏好 |
| **UW** | Uniform Weighting | odds 算术平均 | 中等 |
| **BF** | Behavior & Feedback | 基于历史正确率学习 | 约等于最佳个体 |
| **CF** | Coin Flip | 分歧时随机猜 | 基线（floor） |

**直觉理解**:
- **DSS**: 两人把证据完全共享，像一个人有两套记忆
- **DMC**: 谁更自信听谁的
- **UW**: 民主投票，每人权重相同
- **BF**: 观察谁更准，逐渐学习信任谁
- **CF**: 抛硬币（完全忽略证据质量）

---

## 4. 模拟结果（四张图）

### Figure 1: Bahrami Parameter Sweep — Collective Benefit Ratio

[INSERT FIGURE 1 HERE]

**设定**:
- Agent A 能力固定: c_A = 0.7
- Agent B 能力扫描: c_B = 0.1 到 0.9
- Y轴: CBR = d'_team / max(d'_A, d'_B)

**核心发现**:
1. **DSS（蓝）、DMC（绿）、UW（红）都能达到 CBR > 1**——群体优于最佳个体
2. **能力相近时优势最大**: 当 c_B ≈ 0.7 时，CBR 最高（约 1.5-1.6）
3. **CF（紫）始终 < 1**——随机猜只会拖后腿
4. **BF（橙）≈ 1**——学习规则约等于最佳个体

**预测**: 如果人们采用类似 DMC 或 UW 的策略，我们应该能在实验中观察到群体增益。

---

### Figure 2: Hit Rate vs Correct Rejection 拆解

[INSERT FIGURE 2 HERE]

**左图**: Hit Rate = P(say "Old" | truly Old)
**右图**: Correct Rejection Rate = P(say "New" | truly New)

**核心发现**:
1. **UW 和 DMC 的 Hit Rate 明显高于 Oracle-Best**
2. **CR 的改善相对较小**
3. **群体优势主要来自 "hit rescue"**——把个体漏掉的 Old 项目救回来

**机制解释**: 当一个人"错过"了一个 Old 项目（Miss），另一个人可能记住了。群体规则让两人互补。

---

### Figure 3: Improvement over Oracle-Best (Δ)

[INSERT FIGURE 3 HERE]

**左图**: ΔHit Rate = Rule - Oracle-Best
**右图**: ΔCorrect Rejection = Rule - Oracle-Best

**核心发现**:
1. **UW/DMC 的 ΔHit 最高可达 +0.15**（显著提升）
2. **ΔCR 约 +0.02**（有改善但较小）
3. **CF 两边都是负的**——损失

**结论**: CBR > 1 主要由 Hit Rate 驱动，CR 改善是次要贡献。

---

### Figure 4: Confidence Miscalibration Sweep

[INSERT FIGURE 4 HERE]

**设定**:
- Agent A 固定 α_A = 1.2（过度自信）
- Agent B 扫描 α_B = 0.5 到 1.5
- α = 1 表示完美校准
- α > 1 表示过度自信
- α < 1 表示不够自信

**核心发现**:
1. **DSS（蓝）完全平坦**——不受 miscalibration 影响（它不用 confidence）
2. **UW_Miscal 和 DMC_Miscal 在 α < 1 时下降**——不够自信会损害群体表现
3. **但即使有 miscalibration，CBR 仍然 > 1**——群体优势存在但减少

**预测**: 如果被试的信心与准确性不完全一致（现实中很常见），群体优势会减少，但不会消失。

---

## 5. 与 Tim 的实验设计讨论 (Feb 5, 2026)

### 5.1 两种设计方案

**Design 1: 不操纵能力（事后分类）**
- 所有被试学习同样的材料、同样的时间
- 测试后，根据个体 d' 分类（高能力 vs 低能力）
- **优点**: 自然变异，ecological validity
- **缺点**: 无法确定因果关系

**Design 2: 操纵能力（实验控制）**
- 方式 A: 不同学习时长（如 2秒 vs 5秒 per item）
- 方式 B: 不同学习次数（如 1遍 vs 3遍）
- **优点**: 可以确定因果关系
- **缺点**: 需要防止被试发现能力差异

### 5.2 Tim 的核心关切

> "我们能不能把 simulation 里的 c（编码质量）映射到实验操纵上？"

这是下一步需要解决的关键问题。

---

## 6. Next Steps（具体行动项）

### 短期（1-2 周）
- [ ] **决定实验范式**: Design 1 or Design 2
- [ ] **设计 pilot study**: 测试操纵是否有效
- [ ] **确定 N=2 配对方式**: 随机配对 vs 能力匹配

### 中期（IRB 通过后）
- [ ] **开始数据收集**: 使用在线平台（已部署）
- [ ] **分析实际被试的决策规则**: 与模拟预测对比

### 开放问题
- 如何操纵 confidence 而不影响 accuracy？
- N > 2 时群体动态如何变化？
- 如何处理强制 disagreement vs 自然 disagreement？

---

## Appendix: 技术细节

- **REM 实现**: `/Users/yiytan/Collective_Memory/Simulations/src/rem_core.py`
- **群体规则**: `/Users/yiytan/Collective_Memory/Simulations/src/group_rules.py`
- **在线实验系统**: https://collective-memory-d3802.web.app
- **项目文档**: `/Users/yiytan/Collective_Memory/CLAUDE.md`

---

*Last updated: February 6, 2026*
