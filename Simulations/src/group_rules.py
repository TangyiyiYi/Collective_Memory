"""
Group Decision Rules for Bahrami Parameter Sweep

Implements 5 fusion rules:
1. CF (Coin Flip) - agree → adopt, disagree → coin flip
2. UW (Uniform Weighting) - arithmetic mean of raw odds
3. DMC (Defer to Max Confidence) - defer to largest |L|
4. DSS (Direct Signal Sharing) - sum of log-odds (Bayesian optimal)
5. BF (Behavioral Feedback) - learn from individual correctness

Plus confidence miscalibration rules (Prelec weighting):
6. UW_Miscal - UW + Prelec (arithmetic mean of w, NOT Bahrami's WCS)
7. DMC_Miscal - Defer to Max Confidence with miscalibration (max |w - 0.5|)

All rules return d' metrics only.
"""

import numpy as np
from scipy import stats
from typing import Dict

EPSILON = 1e-10


def compute_hautus_correction(hits: int, misses: int, fas: int, crs: int) -> float:
    """
    Compute d' with Hautus correction.

    Returns d' (dprime).
    """
    S = hits + misses
    N = fas + crs

    HR = (hits + 0.5) / (S + 1)
    FAR = (fas + 0.5) / (N + 1)

    HR = np.clip(HR, EPSILON, 1 - EPSILON)
    FAR = np.clip(FAR, EPSILON, 1 - EPSILON)

    dprime = stats.norm.ppf(HR) - stats.norm.ppf(FAR)

    return dprime


def compute_dprime_from_decisions(decisions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute d' from binary decisions and labels.

    Parameters:
    -----------
    decisions : np.ndarray
        Binary decisions (1 = Old, 0 = New)
    labels : np.ndarray
        True labels (1 = Old/Target, 0 = New/Foil)

    Returns:
    --------
    dprime : float
    """
    target_mask = (labels == 1)
    foil_mask = (labels == 0)

    hits = np.sum((decisions == 1) & target_mask)
    misses = np.sum((decisions == 0) & target_mask)
    fas = np.sum((decisions == 1) & foil_mask)
    crs = np.sum((decisions == 0) & foil_mask)

    return compute_hautus_correction(hits, misses, fas, crs)


def coin_flip_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Coin Flip rule: agree → adopt decision, disagree → coin flip.

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    agree_mask = (D_A == D_B)
    D_team = np.zeros_like(D_A)

    # Agree: adopt decision
    D_team[agree_mask] = D_A[agree_mask]

    # Disagree: coin flip
    n_disagree = np.sum(~agree_mask)
    if n_disagree > 0:
        D_team[~agree_mask] = rng.choice([0, 1], size=n_disagree)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }


def uniform_weighting_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Uniform Weighting rule: arithmetic mean of raw odds.

    CRITICAL: This is NOT log-space averaging.
    mean_odds = (exp(L_A) + exp(L_B)) / 2
    decision: mean_odds > 1

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Arithmetic mean of RAW odds
    odds_A = np.exp(L_A)
    odds_B = np.exp(L_B)
    mean_odds = (odds_A + odds_B) / 2

    D_team = (mean_odds > 1).astype(int)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }


def defer_to_max_confidence(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Defer to Max Confidence rule: choose agent with larger |L|.
    Tie → coin flip.

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    conf_A = np.abs(L_A)
    conf_B = np.abs(L_B)

    D_team = np.zeros_like(D_A)

    # A has higher confidence
    D_team[conf_A > conf_B] = D_A[conf_A > conf_B]

    # B has higher confidence
    D_team[conf_B > conf_A] = D_B[conf_B > conf_A]

    # Tie: coin flip
    tie_mask = (conf_A == conf_B)
    n_tie = np.sum(tie_mask)
    if n_tie > 0:
        D_team[tie_mask] = rng.choice([0, 1], size=n_tie)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }


def direct_signal_sharing(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Direct Signal Sharing rule: sum of log-odds (Bayesian optimal).

    L_team = L_A + L_B
    decision: L_team > 0

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Sum log-odds
    L_team = L_A + L_B
    D_team = (L_team > 0).astype(int)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }


def behavior_feedback_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Behavioral Feedback rule: learn from INDIVIDUAL correctness only.

    CRITICAL: Scores update based on individual correctness,
    NOT group decision correctness.

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    n_trials = len(L_A)
    D_team = np.zeros(n_trials, dtype=int)

    score_A = 0
    score_B = 0

    for i in range(n_trials):
        # Decision based on current scores
        if score_A > score_B:
            D_team[i] = D_A[i]
        elif score_B > score_A:
            D_team[i] = D_B[i]
        else:
            # Tie (including trial 0) -> coin flip
            D_team[i] = D_A[i] if rng.random() < 0.5 else D_B[i]

        # Learning: update scores based on INDIVIDUAL correctness
        if D_A[i] == labels[i]:
            score_A += 1
        if D_B[i] == labels[i]:
            score_B += 1

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }


# ============================================================================
# CONFIDENCE MISCALIBRATION (PRELEC WEIGHTING)
# ============================================================================

def prelec_weighting(L: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply Prelec probability weighting to log-odds.

    LOCKED SEMANTICS:
    1. Accept log-odds L as input
    2. Convert L to probability p via sigmoid: p = 1/(1 + exp(-L))
    3. Compute beta = (log 2)^(1-alpha) to ensure w(0.5) = 0.5
    4. Apply Prelec: w(p) = exp(-beta * (-log p)^alpha)
    5. Return w in (0,1)

    Parameters:
    -----------
    L : np.ndarray
        Log-odds (REM output)
    alpha : float
        Miscalibration parameter
        alpha = 1: calibrated
        alpha > 1: overconfident
        alpha < 1: underconfident

    Returns:
    --------
    w : np.ndarray
        Weighted probability (subjective confidence)
    """
    # Convert log-odds to probability
    p = 1.0 / (1.0 + np.exp(-L))

    # Clip for numerical stability
    p = np.clip(p, 1e-9, 1 - 1e-9)

    # Compute beta to ensure w(0.5) = 0.5
    beta = np.power(np.log(2), 1 - alpha)

    # Apply Prelec weighting
    w = np.exp(-beta * np.power(-np.log(p), alpha))

    return w


def uw_miscal_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    alpha_A: float,
    alpha_B: float
) -> Dict[str, float]:
    """
    Uniform Weighting with miscalibration (Prelec weighting).

    This is UW + Prelec, NOT Bahrami's WCS. Renamed per Tim's model specification.

    Logic:
    ------
    1. Compute w_A, w_B via Prelec weighting
    2. Treat w as probability of OLD
    3. w_team = (w_A + w_B) / 2  (arithmetic mean, NOT confidence-weighted)
    4. Decision: OLD if w_team > 0.5 (strictly greater than)

    Parameters:
    -----------
    L_A, L_B : np.ndarray
        Log-odds for members A and B
    labels : np.ndarray
        True labels
    rng : np.random.Generator
        Random number generator (not used, for signature consistency)
    alpha_A, alpha_B : float
        Miscalibration parameters for A and B

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Apply Prelec weighting
    w_A = prelec_weighting(L_A, alpha_A)
    w_B = prelec_weighting(L_B, alpha_B)

    # Average weighted probabilities
    w_team = (w_A + w_B) / 2

    # Decision: w_team > 0.5 (indifference point)
    D_team = (w_team > 0.5).astype(int)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team
    }


def dmc_miscal_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    alpha_A: float,
    alpha_B: float
) -> Dict[str, float]:
    """
    Defer to Max Confidence with miscalibration (Prelec weighting).

    Logic:
    ------
    1. Compute w_A, w_B via Prelec weighting
    2. Define confidence magnitude: c = |w - 0.5|
    3. Choose agent with larger c
    4. Tie-breaking: coin flip using rng (trial-by-trial)

    CRITICAL: Confidence is distance from indifference (0.5), NOT raw w.

    Parameters:
    -----------
    L_A, L_B : np.ndarray
        Log-odds for members A and B
    labels : np.ndarray
        True labels
    rng : np.random.Generator
        Random number generator for tie-breaking
    alpha_A, alpha_B : float
        Miscalibration parameters for A and B

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team
    """
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Apply Prelec weighting
    w_A = prelec_weighting(L_A, alpha_A)
    w_B = prelec_weighting(L_B, alpha_B)

    # Confidence magnitude: distance from indifference
    conf_A = np.abs(w_A - 0.5)
    conf_B = np.abs(w_B - 0.5)

    D_team = np.zeros_like(D_A)

    # A has higher confidence
    D_team[conf_A > conf_B] = D_A[conf_A > conf_B]

    # B has higher confidence
    D_team[conf_B > conf_A] = D_B[conf_B > conf_A]

    # Tie: coin flip (trial-by-trial)
    tie_mask = (conf_A == conf_B)
    n_tie = np.sum(tie_mask)
    if n_tie > 0:
        D_team[tie_mask] = rng.choice([0, 1], size=n_tie)

    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team
    }


# ============================================================================
# RICH'S CONFLICT RESOLUTION MODEL
# ============================================================================

def rich_conflict_rule(L_A: np.ndarray, L_B: np.ndarray, labels: np.ndarray, rng: np.random.Generator, beta: float = 1.0):
    """
    Rich Shiffrin's Conflict Resolution Model.

    When agents disagree (Old vs New), the group chooses the agent with
    stronger evidence according to: P = ((1 + D) / (2 + D))^beta
    where D = |S_A - S_B| and S = max(φ', 1/φ'), φ' = φ^(1/11), φ = exp(L)

    Parameters:
    -----------
    L_A, L_B : np.ndarray
        Log-odds for agents A and B
    labels : np.ndarray
        True labels (1 = Old, 0 = New)
    rng : np.random.Generator
        Random number generator
    beta : float, default=1.0
        Exponent parameter in probability formula

    Returns:
    --------
    dict with keys:
        - dprime_A, dprime_B, dprime_team: sensitivity metrics
        - decisions: team decision array
        - conflict_mask: boolean array indicating conflict trials
        - strength_A, strength_B: strength values for each trial
        - D_values: strength difference |S_A - S_B| for each trial
    """
    # Individual decisions
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Convert to Odds
    odds_A = np.exp(L_A)
    odds_B = np.exp(L_B)

    # Fixed power scaling (1/11) - NOT a parameter
    power = 1.0 / 11.0
    odds_scaled_A = np.power(odds_A, power)
    odds_scaled_B = np.power(odds_B, power)

    # Strength: max(φ', 1/φ')
    S_A = np.maximum(odds_scaled_A, 1.0 / odds_scaled_A)
    S_B = np.maximum(odds_scaled_B, 1.0 / odds_scaled_B)

    # Identify conflict trials (strict definition)
    conflict_mask = ((L_A > 0) & (L_B < 0)) | ((L_A < 0) & (L_B > 0))

    # Compute difference D for ALL trials (needed for return metadata)
    D_diff = np.abs(S_A - S_B)

    # Initialize team decisions
    D_team = np.zeros_like(D_A)

    # Agreement trials: adopt agreed decision
    agree_mask = ~conflict_mask
    D_team[agree_mask] = D_A[agree_mask]

    # Conflict trials: use Rich's probability rule
    if np.sum(conflict_mask) > 0:
        # Probability of choosing stronger (with beta exponent)
        P_choose_stronger = np.power((1.0 + D_diff) / (2.0 + D_diff), beta)

        # Determine who is stronger
        A_stronger = S_A > S_B
        B_stronger = S_B > S_A

        # Generate random choices for conflict trials
        conflict_indices = np.where(conflict_mask)[0]
        for idx in conflict_indices:
            p = P_choose_stronger[idx]

            if A_stronger[idx]:
                # A is stronger, choose A with probability p
                choose_A = rng.random() < p
                D_team[idx] = D_A[idx] if choose_A else D_B[idx]
            elif B_stronger[idx]:
                # B is stronger, choose B with probability p
                choose_B = rng.random() < p
                D_team[idx] = D_B[idx] if choose_B else D_A[idx]
            else:
                # Equal strength (D = 0), random choice
                D_team[idx] = D_A[idx] if rng.random() < 0.5 else D_B[idx]

    # Compute d' metrics
    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team,
        'conflict_mask': conflict_mask,
        'strength_A': S_A,
        'strength_B': S_B,
        'D_values': D_diff
    }


def best_odds_deterministic_rule(
    L_A: np.ndarray,
    L_B: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Best Odds Deterministic rule: deterministic baseline for conflict resolution.

    On conflict trials: ALWAYS choose the agent with larger |L| (stronger evidence).
    On agreement trials: adopt the agreed decision.

    This represents the "ceiling" for conflict-based decision approaches.

    Parameters:
    -----------
    L_A, L_B : np.ndarray
        Log-odds from agents A and B
    labels : np.ndarray
        True labels (1 = Old, 0 = New)
    rng : np.random.Generator
        Not used (kept for interface consistency)

    Returns:
    --------
    dict with dprime_A, dprime_B, dprime_team, decisions
    """
    # Convert to binary decisions
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)

    # Detect conflict
    conflict_mask = (D_A != D_B)

    # Initialize with agreement
    D_team = D_A.copy()

    # On conflict: deterministically choose max |L|
    if np.sum(conflict_mask) > 0:
        conflict_indices = np.where(conflict_mask)[0]
        for idx in conflict_indices:
            if np.abs(L_A[idx]) > np.abs(L_B[idx]):
                D_team[idx] = D_A[idx]
            else:
                D_team[idx] = D_B[idx]

    # Compute d' metrics
    dprime_A = compute_dprime_from_decisions(D_A, labels)
    dprime_B = compute_dprime_from_decisions(D_B, labels)
    dprime_team = compute_dprime_from_decisions(D_team, labels)

    return {
        'dprime_A': dprime_A,
        'dprime_B': dprime_B,
        'dprime_team': dprime_team,
        'decisions': D_team
    }
