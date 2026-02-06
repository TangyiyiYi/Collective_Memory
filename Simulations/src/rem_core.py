"""
REM Core Engine

This module implements the REM model's core likelihood ratio computation
with value-dependent feature rarity, following Shiffrin & Steyvers (1997).

Key features:
- λ_v formula with feature value v dependence
- Log-space operations for numerical stability
- Two-level log computation (per-trace → item-level)
- Geometric distribution P(v) = g(1-g)^(v-1)

Author: YIYAN 
Date: 2025
"""

import numpy as np
import warnings
from typing import Tuple

# Numerical stability constants
MIN_P = 1e-300  # Floor for extremely small P(v) to prevent overflow
EPSILON = 1e-10


def compute_lambda(g: float, c: float, v: int) -> float:
    """
    Compute feature-level likelihood ratio λ_v with value-dependent rarity.

    Formula:
        P(v) = g(1-g)^(v-1)  [geometric distribution]
        λ_v = [c + (1-c)P(v)] / P(v)

    Rarer features (smaller P(v)) yield larger λ_v (stronger evidence).

    Parameters:
    -----------
    g : float
        Geometric distribution parameter (0 < g < 1)
    c : float
        Copy correctness probability (0 ≤ c ≤ 1)
    v : int
        Feature value (v ≥ 1, natural numbers)

    Returns:
    --------
    lambda_v : float
        Likelihood ratio for matched feature value v

    Notes:
    ------
    - For mismatched features, λ_v = (1-c)
    - For unstored features (v=0 in trace), λ_v = 1 → log(λ_v) = 0 (neutral)
    - P(v) is floored at MIN_P to prevent numerical overflow
    """
    # Assertions for parameter validity
    assert 0 < g < 1, f"g must be in (0,1), got {g}"
    assert 0 <= c <= 1, f"c must be in [0,1], got {c}"
    assert isinstance(v, (int, np.integer)) and v >= 1, f"v must be integer ≥1, got {v}"

    # Compute P(v) with numerical floor
    P_v = g * ((1 - g) ** (v - 1))
    P_v = max(P_v, MIN_P)  # Floor to prevent division by tiny numbers

    # Compute λ_v
    lambda_v = (c + (1 - c) * P_v) / P_v

    return lambda_v


def generate_traces(
    study_list: np.ndarray,
    u: float,
    c: float,
    g: float,
    nSteps: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate memory traces for a study list via noisy storage process.

    Process:
    --------
    1. Initialize fixed-size trace array (n_items, w) filled with zeros
    2. Over nSteps encoding opportunities:
       - Each feature has probability u of being stored
       - Stored features are copied correctly with probability c
       - Incorrectly copied features drawn from geometric distribution g
    3. Once a feature is stored (non-zero), it is not overwritten

    Parameters:
    -----------
    study_list : np.ndarray
        Study items, shape (n_items, w) where w is number of features
    u : float
        Per-feature storage probability (0 ≤ u ≤ 1)
    c : float
        Correct copy probability (0 ≤ c ≤ 1)
    g : float
        Geometric distribution parameter for error features (0 < g < 1)
    nSteps : int
        Number of encoding steps (≥ 1)
    rng : np.random.Generator
        Independent random number generator for this member

    Returns:
    --------
    traces : np.ndarray
        Memory traces, shape (n_items, w)
        - 0 indicates unstored feature
        - v > 0 indicates stored value

    Notes:
    ------
    - Expected fill rate: p_filled = 1 - (1-u)^nSteps
    - Expected correct-copy rate: p_correct ≈ p_filled * c
    - Expected error-write rate: p_error ≈ p_filled * (1-c)
    """
    # Assertions
    assert 0 <= u <= 1, f"u must be in [0,1], got {u}"
    assert 0 <= c <= 1, f"c must be in [0,1], got {c}"
    assert 0 < g < 1, f"g must be in (0,1), got {g}"
    assert isinstance(nSteps, int) and nSteps >= 1, f"nSteps must be int ≥1, got {nSteps}"
    assert study_list.ndim == 2, f"study_list must be 2D, got shape {study_list.shape}"

    n_items, w = study_list.shape
    traces = np.zeros((n_items, w), dtype=np.int32)

    # Iterative storage process
    for step in range(nSteps):
        for k, item in enumerate(study_list):
            # Which features to store at this step?
            store = rng.choice([0, 1], size=w, p=[1 - u, u])

            # Will stored features be copied correctly?
            copy_correct = rng.choice([0, 1], size=w, p=[1 - c, c])

            # Current trace for this item
            trace = traces[k, :]

            # Update trace: only fill empty slots (trace == 0)
            # Correct copy: store == 1 & copy_correct == 1 & trace == 0 → item[:]
            # Error copy: store == 1 & copy_correct == 0 & trace == 0 → geometric(g)
            # Already filled: keep current value

            # Correct copies
            correct_mask = (store == 1) & (copy_correct == 1) & (trace == 0)
            traces[k, correct_mask] = item[correct_mask]

            # Error copies (draw from geometric)
            error_mask = (store == 1) & (copy_correct == 0) & (trace == 0)
            if np.any(error_mask):
                traces[k, error_mask] = rng.geometric(g, size=np.sum(error_mask))

    return traces


def compute_log_odds(
    probe: np.ndarray,
    traces: np.ndarray,
    c: float,
    g: float
) -> float:
    """
    Compute item-level log-odds L = log(Φ) for a probe against stored traces.

    Two-level log computation:
    ---------------------------
    1. Per-trace level: Σ log(λ_v) across features
       - Unstored (trace[f] == 0): log(λ) = 0 (neutral evidence)
       - Matched (trace[f] == probe[f]): log(λ_v) using compute_lambda
       - Mismatched: log(λ) = log(1-c)

    2. Item level: log of arithmetic mean of λ across traces
       L = log(Φ) = log(mean(λ_i)) = logsumexp(log λ_i) - log(N_traces)

    Parameters:
    -----------
    probe : np.ndarray
        Test item, shape (w,) where w is number of features
    traces : np.ndarray
        Memory traces, shape (n_traces, w)
    c : float
        Copy correctness probability (0 ≤ c ≤ 1)
    g : float
        Geometric distribution parameter (0 < g < 1)

    Returns:
    --------
    L : float
        Log-odds (log Φ). Decision rule: L > 0 → "Old", else "New"

    Notes:
    ------
    - Use scipy.special.logsumexp for numerical stability
    - All operations in log-space to prevent overflow/underflow
    """
    from scipy.special import logsumexp

    # Assertions
    assert probe.ndim == 1, f"probe must be 1D, got shape {probe.shape}"
    assert traces.ndim == 2, f"traces must be 2D, got shape {traces.shape}"
    assert traces.shape[1] == len(probe), "Feature dimension mismatch"
    assert 0 <= c <= 1, f"c must be in [0,1], got {c}"
    assert 0 < g < 1, f"g must be in (0,1), got {g}"

    n_traces, w = traces.shape
    log_lambdas = np.zeros(n_traces, dtype=np.float64)

    # Compute per-trace log(λ)
    for j, trace in enumerate(traces):
        log_lambda_trace = 0.0

        for f in range(w):
            if trace[f] == 0:
                # Unstored feature: λ = 1 → log(λ) = 0 (neutral)
                continue
            elif trace[f] == probe[f]:
                # Matched feature: use value-dependent λ_v
                lambda_v = compute_lambda(g, c, int(trace[f]))
                log_lambda_trace += np.log(lambda_v)
            else:
                # Mismatched feature: λ = (1-c)
                log_lambda_trace += np.log(max(1 - c, EPSILON))  # Avoid log(0)

        log_lambdas[j] = log_lambda_trace

    # Item-level: L = log(mean(λ)) = logsumexp(log λ) - log(N)
    L = logsumexp(log_lambdas) - np.log(n_traces)

    return L


# ============================================================================
# VECTORIZED IMPLEMENTATIONS (High Performance)
# ============================================================================

def compute_log_lambda_vectorized(
    values: np.ndarray,
    c: float,
    g: float,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Vectorized computation of log(λ_v) for matched feature values.

    Parameters:
    -----------
    values : np.ndarray, shape (n_traces, w)
        Trace values (only values where valid_mask is True will be processed)
    c : float
        Copy correctness probability
    g : float
        Geometric distribution parameter
    valid_mask : np.ndarray, shape (n_traces, w)
        Boolean mask indicating which positions to compute

    Returns:
    --------
    log_lambdas : np.ndarray, shape (n_traces, w)
        Log-likelihood ratios (0 where valid_mask is False)
    """
    result = np.zeros_like(values, dtype=np.float64)

    # Only compute for valid positions (matched & non-zero)
    if not np.any(valid_mask):
        return result

    v = values[valid_mask].astype(np.float64)

    # P(v) = g * (1-g)^(v-1) with numerical floor
    P_v = g * np.power(1 - g, v - 1)
    P_v = np.maximum(P_v, MIN_P)

    # λ_v = (c + (1-c)*P(v)) / P(v)
    lambda_v = (c + (1 - c) * P_v) / P_v

    result[valid_mask] = np.log(lambda_v)

    return result


def compute_log_odds_vectorized(
    probe: np.ndarray,
    traces: np.ndarray,
    c: float,
    g: float
) -> float:
    """
    Vectorized computation of item-level log-odds L = log(Φ).

    This is a high-performance version that eliminates nested Python loops
    using NumPy broadcasting. Mathematically equivalent to compute_log_odds().

    Parameters:
    -----------
    probe : np.ndarray, shape (w,)
        Test item features
    traces : np.ndarray, shape (n_traces, w)
        Memory traces
    c : float
        Copy correctness probability
    g : float
        Geometric distribution parameter

    Returns:
    --------
    L : float
        Log-odds (log Φ). Decision rule: L > 0 → "Old", else "New"
    """
    from scipy.special import logsumexp

    n_traces, w = traces.shape

    # Step 1: Create boolean masks for all conditions
    # Shape: (n_traces, w)
    zero_mask = (traces == 0)                      # Unstored features
    match_mask = (traces == probe) & ~zero_mask   # Matched & stored
    mismatch_mask = ~zero_mask & ~match_mask       # Stored but mismatched

    # Step 2: Compute log-likelihood contributions
    # Initialize with zeros (neutral evidence for unstored features)
    log_contributions = np.zeros((n_traces, w), dtype=np.float64)

    # For matched features: compute log(λ_v) using vectorized helper
    if np.any(match_mask):
        log_contributions += compute_log_lambda_vectorized(traces, c, g, match_mask)

    # For mismatched features: log(1-c)
    if np.any(mismatch_mask):
        log_mismatch = np.log(max(1 - c, EPSILON))
        log_contributions[mismatch_mask] = log_mismatch

    # Step 3: Sum across features for each trace
    log_lambdas = np.sum(log_contributions, axis=1)  # Shape: (n_traces,)

    # Step 4: Item-level log-odds: L = logsumexp(log λ) - log(N)
    L = logsumexp(log_lambdas) - np.log(n_traces)

    return L


def print_storage_diagnostics(u: float, c: float, nSteps: int, traces: np.ndarray):
    """
    Print storage rate diagnostics for verification.

    Parameters:
    -----------
    u : float
        Storage probability
    c : float
        Copy correctness probability
    nSteps : int
        Number of storage steps
    traces : np.ndarray
        Generated traces (for empirical estimation)
    """
    # Theoretical fill rate
    p_filled_theory = 1 - (1 - u) ** nSteps

    # Empirical fill rate
    p_filled_empirical = np.mean(traces > 0)

    # Expected rates (theory)
    p_correct_theory = p_filled_theory * c
    p_error_theory = p_filled_theory * (1 - c)

    print("\n=== REM Storage Diagnostics ===")
    print(f"Expected write-rate (any storage): {p_filled_theory:.3f} (u={u}, nSteps={nSteps})")
    print(f"Empirical fill rate: {p_filled_empirical:.3f}")
    print(f"Expected correct-copy rate: {p_correct_theory:.3f}")
    print(f"Expected error-write rate: {p_error_theory:.3f}")

    # Warning if fill rate is too extreme
    if p_filled_empirical < 0.10:
        warnings.warn(f"Fill rate very low ({p_filled_empirical:.1%}), may compress evidence discriminability")
    elif p_filled_empirical > 0.70:
        warnings.warn(f"Fill rate very high ({p_filled_empirical:.1%}), may saturate evidence")

    print("=" * 40 + "\n")


if __name__ == "__main__":
    # Quick sanity test
    print("REM Core Engine - Sanity Test")
    print("=" * 50)

    # Test parameters
    g = 0.4
    c = 0.7
    u = 0.04
    nSteps = 5
    w = 20
    n_study = 50

    # Generate study list
    rng = np.random.default_rng(42)
    study_list = rng.geometric(g, size=(n_study, w))

    # Generate traces
    traces = generate_traces(study_list, u, c, g, nSteps, rng)
    print_storage_diagnostics(u, c, nSteps, traces)

    # Test log-odds computation
    probe_old = study_list[0]  # Target
    probe_new = rng.geometric(g, size=w)  # Foil

    L_old = compute_log_odds(probe_old, traces, c, g)
    L_new = compute_log_odds(probe_new, traces, c, g)

    print(f"Log-odds for Target: L = {L_old:.4f} ({'Old' if L_old > 0 else 'New'})")
    print(f"Log-odds for Foil: L = {L_new:.4f} ({'Old' if L_new > 0 else 'New'})")

    # Verify vectorized implementation matches original
    print("\n--- Vectorized Implementation Verification ---")
    L_old_vec = compute_log_odds_vectorized(probe_old, traces, c, g)
    L_new_vec = compute_log_odds_vectorized(probe_new, traces, c, g)

    print(f"Vectorized Target: L = {L_old_vec:.4f}")
    print(f"Vectorized Foil: L = {L_new_vec:.4f}")

    # Check equivalence
    assert np.isclose(L_old, L_old_vec, rtol=1e-10), f"Target mismatch: {L_old} vs {L_old_vec}"
    assert np.isclose(L_new, L_new_vec, rtol=1e-10), f"Foil mismatch: {L_new} vs {L_new_vec}"

    print("✓ Vectorized matches original (rtol=1e-10)")

    # Performance comparison
    import time
    n_probes = 100
    probes = rng.geometric(g, size=(n_probes, w))

    start = time.time()
    for p in probes:
        compute_log_odds(p, traces, c, g)
    original_time = time.time() - start

    start = time.time()
    for p in probes:
        compute_log_odds_vectorized(p, traces, c, g)
    vectorized_time = time.time() - start

    print(f"\nPerformance ({n_probes} probes):")
    print(f"  Original:   {original_time:.3f}s")
    print(f"  Vectorized: {vectorized_time:.3f}s")
    print(f"  Speedup:    {original_time/vectorized_time:.1f}x")

    print("\n✓ rem_core.py sanity test passed")
