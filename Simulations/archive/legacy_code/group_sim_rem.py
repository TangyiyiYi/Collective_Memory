# -*- coding: utf-8 -*-
"""
Group Decision Simulation (Recognition Memory) —— 基础可运行脚本
===============================================================

目的（Purpose）
--------------
在**识别记忆（recognition memory）**任务中，先为每位组员生成**个体证据（individual evidence）**，
再按不同的**群体整合规则（group fusion rules）**将个体决策合成为**团队决策（team decision）**，
比较各规则的命中率/虚警率/ d′ 等指标。

⚠️ 说明（Beginner notes）：
1) 这里把 REM（Retrieving-Effectively-from-Memory）的试次证据近似成 SDT（信号检测）形式的
   “对数似然比/标准化证据 z（standardized evidence）”。这是 REM→SDT 的常见简化：
   old 试次采样 z ~ N(+d′/2, 1)，new 试次采样 z ~ N(−d′/2, 1)；阈值（criterion）默认为 0。
   —— 这样做能直接对接 Enright（UW/OW）、Bahrami（WCS/DSS）、Ernst & Banks（MLE）的公式。
2) 若要替换成“真正的 REM 证据生成”，只需要改 `sample_evidence_for_member()` 里
   的采样部分（把 z_i 换成REM 的 log-likelihood ratio 的标准化版本），其余群体规则都无需改动。
3) 术语均**中英对照**（bilingual），并在关键函数写了**行内注释**，方便初学者阅读。

群体规则（Group rules, 中英对照）
--------------------------------
- UW / UM：**均权加权（Uniform Weighting）**（Enright 论文常用基线），z_team = (z1 + ... + zm) / sqrt(m)；阈值 0。
- WCS：**加权信心共享（Weighted Confidence Sharing）**（Bahrami 模型），用“信心（confidence）”给成员分配权重；
        这里用 |z_i| 的平滑函数近似信心（也可改为你自己的置信度产生机制）。
- DSS：**直接信号共享（Direct Signal Sharing）**（Bahrami 上界；Bayesian optimal，如 Ernst & Banks 的 MLE 思想）：
        z_team = z1 + ... + zm；阈值 0（在独立、等方差下，对应 d′_team = sqrt(sum d′_i^2)）。
- CF：**掷硬币（Coin Flip）**：若意见不一致，随机听其中一人；一致则采纳一致意见（下界基线）。
- BF：**行为与反馈（Behavior & Feedback）**：若分歧，听“历史更准者”（rolling accuracy）。
       这里用 Beta(1,1) 先验的逐步更新（不包含**输出干扰 OI**；与 Rich 的建议一致：先不加 OI，后续数据分析再看时序变化）。

输出（Outputs）
---------------
- 每条规则下：命中率（Hit, 旧项判旧）、虚警率（False Alarm, 新项判旧）、d′（zPhi(H) − zPhi(FA)）
- 同时返回**个体成员**的指标，以及“best member vs. team”的增益比较。

快速使用（Quick Start）
----------------------
1) 直接运行本文件：
   $ python group_sim_rem.py
   将在控制台打印一个汇总表，并把 CSV 保存到 ./sim_summary.csv

2) 作为模块导入：
   >>> import group_sim_rem as gsm
   >>> res = gsm.run_simulation(seed=42)
   >>> print(res["summary_df"])
：
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# =========
# 基本数据结构（Data classes）
# =========

@dataclass
class MemberSpec:
    """单个成员（group member）的能力规格
    d_prime: 个人灵敏度 d′（越大越容易区分 old/new）
    name:    成员名（便于输出）
    """
    d_prime: float
    name: str


@dataclass
class SimSpec:
    """一次仿真的总体规格（Simulation spec）"""
    n_trials: int = 2000              # 试次数（每组员共享相同试次的真值 old/new）
    p_old: float = 0.5                # old 概率（目标/旧项比例）
    group_members: List[MemberSpec] = field(default_factory=lambda: [MemberSpec(1.2, "A"), MemberSpec(0.8, "B")])
    # 置信度→权重的平滑强度（越大越“偏向高自信者”）；WCS 会用到
    wcs_alpha: float = 1.0
    # BF（行为与反馈）估计准确率的遗忘/平滑系数（0=严格累计, 介于0~1内可做指数滑动平均）
    bf_smoothing: float = 0.0
    # 是否启用 DSS / WCS / UW / CF / BF 五种规则（可选择性关闭做消融）
    enable_rules: Dict[str, bool] = field(default_factory=lambda: {
        "UW": True, "WCS": True, "DSS": True, "CF": True, "BF": True
    })
    # 随机种子
    seed: Optional[int] = 42


# =========
# 工具函数（Utilities）
# =========

def z2p(z: float) -> float:
    """zPhi：把命中/虚警比率 clamp 到 (1e-6, 1-1e-6) 再做逆正态（避免无穷大）"""
    z = min(max(z, 1e-6), 1 - 1e-6)
    return norm.ppf(z)


def compute_dprime(hit: float, fa: float) -> float:
    """d′ = zPhi(H) − zPhi(FA)"""
    return z2p(hit) - z2p(1 - (1 - fa))  # 等价于 norm.ppf(hit) - norm.ppf(fa)


def sample_truth_labels(n_trials: int, p_old: float, rng: np.random.Generator) -> np.ndarray:
    """采样试次真值（1=old, 0=new）"""
    return rng.binomial(1, p_old, size=n_trials).astype(np.int32)


def sample_evidence_for_member(truth: np.ndarray, d_prime: float, rng: np.random.Generator) -> np.ndarray:
    """为某个成员采样 SDT/REM 近似证据（z；criterion=0 时 Old 决策= z>0）
    - old: z ~ N(+d′/2, 1)
    - new: z ~ N(−d′/2, 1)
    备注：这就是把 REM 产生的对数似然比做了标准化近似；若你以后有真正的 REM 证据，替换这里即可。
    """
    n = truth.size
    means = np.where(truth == 1, +d_prime / 2.0, -d_prime / 2.0)
    z = rng.normal(loc=means, scale=1.0, size=n)
    return z


def decisions_from_z(z: np.ndarray, criterion: float = 0.0) -> np.ndarray:
    """个体二分类决策（individual decision）：z > criterion → Old(1)，否则 New(0)"""
    return (z > criterion).astype(np.int32)


def wcs_weights_from_confidence(z: np.ndarray, alpha: float) -> np.ndarray:
    """把 |z| 当作“置信度（confidence）”的 proxy，做一个平滑权重。
    权重计算：w_i ∝ exp(alpha * |z_i|)；再做归一化。
    注：你也可以把置信度改成 trial-wise 的主观评分、或 REM 的后验概率等。
    """
    conf = np.abs(z)
    raw = np.exp(alpha * conf)
    w = raw / np.sum(raw)
    return w


# =========
# 五种群体规则（Group fusion rules）
# =========

def rule_UW(z_list: List[float]) -> int:
    """UW（均权）—— Enright：z_team = (sum z_i)/sqrt(m)；阈值 0"""
    m = len(z_list)
    z_team = np.sum(z_list) / math.sqrt(m)
    return int(z_team > 0.0)


def rule_DSS(z_list: List[float]) -> int:
    """DSS（直接信号共享，上界/最优整合影子）—— Bahrami：z_team = sum z_i；阈值 0"""
    z_team = np.sum(z_list)
    return int(z_team > 0.0)


def rule_WCS(z_list: List[float], alpha: float) -> int:
    """WCS（加权信心共享）—— Bahrami：按“置信度”分配权重，这里用 |z| 的平滑函数近似"""
    z = np.array(z_list, dtype=float)
    w = wcs_weights_from_confidence(z, alpha=alpha)  # 归一化权重
    z_team = float(np.sum(w * z))
    return int(z_team > 0.0)


def rule_CF(indiv_decisions: List[int], rng: random.Random) -> int:
    """CF（掷硬币）：若分歧，随机听一人；若一致，直接采纳一致意见"""
    if all(d == indiv_decisions[0] for d in indiv_decisions):
        return indiv_decisions[0]
    else:
        return rng.choice(indiv_decisions)


class BFState:
    """BF（行为与反馈）滚动准确率（rolling accuracy）状态"""
    def __init__(self, n_members: int, smoothing: float = 0.0):
        # 用 Beta(1,1) 先验的充分统计：success, total
        self.success = [0] * n_members
        self.total = [0] * n_members
        self.smoothing = smoothing  # 0=严格累计；>0 时做 EMA

        # 对 EMA 记录一份“平滑的准确率”，初始化为 0.5
        self.ema_acc = [0.5] * n_members

    def get_scores(self) -> List[float]:
        """返回每位成员当前的“更准度”分数（score）；分歧时选分数高者"""
        scores = []
        for i in range(len(self.success)):
            # 后验均值： (success+1)/(total+2)
            post = (self.success[i] + 1) / (self.total[i] + 2)
            if self.smoothing > 0.0:
                # EMA 融合一下（更稳一点）
                post = (1 - self.smoothing) * post + self.smoothing * self.ema_acc[i]
            scores.append(post)
        return scores

    def update(self, indiv_correct: List[bool]):
        """在试次反馈后更新每人的正确统计"""
        for i, ok in enumerate(indiv_correct):
            self.total[i] += 1
            if ok:
                self.success[i] += 1
            # 更新 EMA 视图
            post = (self.success[i] + 1) / (self.total[i] + 2)
            self.ema_acc[i] = 0.9 * self.ema_acc[i] + 0.1 * post


def rule_BF(indiv_decisions: List[int], truth_label: int, scores: List[float]) -> int:
    """BF（行为与反馈）：若分歧，选“当前估计更准”的成员；一致则直接采纳
    注：判完之后，外层会用真实反馈去更新各人的滚动准确率。
    """
    if all(d == indiv_decisions[0] for d in indiv_decisions):
        return indiv_decisions[0]
    else:
        # 选择 score 最大的人的决策（若并列，默认取第一个最大者）
        idx = int(np.argmax(scores))
        return indiv_decisions[idx]


# =========
# 主仿真（Main simulation）
# =========

def simulate_once(spec: SimSpec) -> Dict[str, pd.DataFrame]:
    """跑一轮仿真，返回：
    - trial_df：逐试次的个体与团队决策
    - summary_df：按规则汇总的指标（Hit/FA/d′）
    """
    rng = np.random.default_rng(spec.seed)
    py_rng = random.Random(spec.seed)

    # 1) 生成真值序列（old/new）
    truth = sample_truth_labels(spec.n_trials, spec.p_old, rng)  # 1=old, 0=new

    # 2) 为每个成员生成证据与个体决策
    member_Z = []      # List[np.ndarray], 每个 shape=(n_trials,)
    member_dec = []    # List[np.ndarray], 每个 shape=(n_trials,)
    for mem in spec.group_members:
        z = sample_evidence_for_member(truth, mem.d_prime, rng)
        d = decisions_from_z(z, 0.0)
        member_Z.append(z)
        member_dec.append(d)

    member_Z = np.vstack(member_Z)        # shape = (m, n_trials)
    member_dec = np.vstack(member_dec)    # shape = (m, n_trials)
    m, n = member_Z.shape

    # 3) 逐试次按不同规则生成团队决策
    bf_state = BFState(n_members=m, smoothing=spec.bf_smoothing)

    team_decisions = {rule: np.zeros(n, dtype=np.int32) for rule, on in spec.enable_rules.items() if on}

    for t in range(n):
        z_t = member_Z[:, t].tolist()
        dec_t = member_dec[:, t].tolist()

        if spec.enable_rules.get("UW", False):
            team_decisions["UW"][t] = rule_UW(z_t)

        if spec.enable_rules.get("DSS", False):
            team_decisions["DSS"][t] = rule_DSS(z_t)

        if spec.enable_rules.get("WCS", False):
            team_decisions["WCS"][t] = rule_WCS(z_t, alpha=spec.wcs_alpha)

        if spec.enable_rules.get("CF", False):
            team_decisions["CF"][t] = rule_CF(dec_t, py_rng)

        if spec.enable_rules.get("BF", False):
            scores = bf_state.get_scores()
            team_decisions["BF"][t] = rule_BF(dec_t, truth[t], scores)
            # 用真实反馈更新各人的滚动准确率（team 决策并不回写到个人统计；BF 用的是个人历史准确率）
            indiv_correct = [(dec_t[i] == truth[t]) for i in range(m)]
            bf_state.update(indiv_correct)

    # 4) 组织逐试次 DataFrame（individual + team）
    cols = {
        "trial": np.arange(n, dtype=int),
        "truth": truth,
    }
    for i, mem in enumerate(spec.group_members):
        cols[f"z_{mem.name}"] = member_Z[i]
        cols[f"dec_{mem.name}"] = member_dec[i]

    for rule, dec in team_decisions.items():
        cols[f"team_{rule}"] = dec

    trial_df = pd.DataFrame(cols)

    # 5) 计算汇总指标（Hit/FA/d′）—— 个人 + 团队
    records = []

    def _metrics_from_binary(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
        hit = float(np.mean(pred[truth == 1] == 1)) if np.any(truth == 1) else float("nan")
        fa  = float(np.mean(pred[truth == 0] == 1)) if np.any(truth == 0) else float("nan")
        dprime = compute_dprime(hit, fa)
        return hit, fa, dprime

    # 个人
    for i, mem in enumerate(spec.group_members):
        hit, fa, dprime = _metrics_from_binary(member_dec[i], truth)
        records.append({"who": f"Indiv_{mem.name}", "rule": "INDIV", "hit": hit, "fa": fa, "dprime": dprime})

    # 团队
    for rule, dec in team_decisions.items():
        hit, fa, dprime = _metrics_from_binary(dec, truth)
        records.append({"who": "Team", "rule": rule, "hit": hit, "fa": fa, "dprime": dprime})

    summary_df = pd.DataFrame.from_records(records)

    # 6) 附加：计算 “Team − BestMember” 增益
    best_indiv_dprime = summary_df.query("rule=='INDIV'")["dprime"].max()
    team_rows = summary_df.query("who=='Team'").copy()
    team_rows["gain_vs_best_member"] = team_rows["dprime"] - best_indiv_dprime

    # 合并回去（为了阅读直观，这里只在团队行显示增益）
    out_summary = pd.concat([summary_df.query("rule=='INDIV'"), team_rows], ignore_index=True)

    return {"trial_df": trial_df, "summary_df": out_summary.sort_values(["who", "rule"]).reset_index(drop=True)}


def run_simulation(
    n_trials: int = 4000,
    p_old: float = 0.5,
    dprimes: Tuple[float, ...] = (1.2, 0.8),
    wcs_alpha: float = 1.0,
    bf_smoothing: float = 0.0,
    enable_rules: Optional[Dict[str, bool]] = None,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """一键运行入口（One-click runner）"""
    if enable_rules is None:
        enable_rules = {"UW": True, "WCS": True, "DSS": True, "CF": True, "BF": True}

    members = [MemberSpec(d_prime=dp, name=chr(ord('A')+i)) for i, dp in enumerate(dprimes)]
    spec = SimSpec(
        n_trials=n_trials,
        p_old=p_old,
        group_members=members,
        wcs_alpha=wcs_alpha,
        bf_smoothing=bf_smoothing,
        enable_rules=enable_rules,
        seed=seed
    )
    return simulate_once(spec)


# =========
# 命令行入口（CLI）
# =========

def _print_summary(df: pd.DataFrame):
    """把关键结果打印成整洁表格（含 Team−BestMember 增益；仅团队行显示该列）"""
    show_cols = ["who", "rule", "hit", "fa", "dprime", "gain_vs_best_member"]
    # 对于个人行，没有增益列，展示为 NaN 更直观
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


if __name__ == "__main__":
    res = run_simulation(
        n_trials=6000,          # 多一点试次更稳定
        p_old=0.5,
        dprimes=(1.2, 0.8),     # 二人组（Dyad）；可改为 (1.2, 0.8, 0.6) 做三人组
        wcs_alpha=1.0,          # WCS 偏爱高置信度的强度
        bf_smoothing=0.0,       # BF 的平滑系数（0=严格累计）
        enable_rules={"UW": True, "WCS": True, "DSS": True, "CF": True, "BF": True},
        seed=42
    )

    summary = res["summary_df"]
    trial_df = res["trial_df"]

    # 保存 CSV
    summary.to_csv("sim_summary.csv", index=False)
    trial_df.to_csv("sim_trials.csv", index=False)

    print("\n=== Simulation Summary (识别记忆群体规则对比) ===")
    _print_summary(summary)

    print("\n已保存：sim_summary.csv 与 sim_trials.csv （当前工作目录）。")
    print("你可以用 pandas/R 进一步画图，或把本脚本集成到你的 REM 证据管线中。")
