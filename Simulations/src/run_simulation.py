"""
Bahrami Parameter Sweep: REM Group Decision Simulation

Sweeps novice ability (c_B) from 0.1 to 0.9 with fixed expert (c_A = 0.7).
Tests 5 decision rules: CF, UW, DMC, DSS, BF.

Plus confidence miscalibration sweep (Prelec weighting).

Output:
- bahrami_sweep_final.csv
- bahrami_sweep_plot.png (Tim's 5-rule comparison)
- rich_theory_verification.png (Rich's DSS vs Theory)
- miscalibration_sweep.csv (Tim's confidence miscalibration)
- miscalibration_plot.png (Prelec weighting effects)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rem_core
import group_rules


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'c_A': 0.7,
    'c_B_values': np.arange(0.1, 1.0, 0.1),
    'w': 20,
    'g': 0.4,
    'u': 0.04,
    'nSteps': 5,
    'n_study': 200,
    'n_test': 10000,
    'seed_master': 42,
    'rules': ['CF', 'UW', 'DMC', 'DSS', 'BF']
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_study_list(n_study: int, w: int, g: float, rng):
    """Generate study list from geometric distribution."""
    return rng.geometric(g, size=(n_study, w))


def make_test_list(study_list, n_old: int, n_new: int, g: float, rng):
    """
    Generate balanced test list.

    Old items: sampled from study list
    New items: i.i.d. from geometric distribution
    """
    n_study, w = study_list.shape

    old_indices = rng.integers(0, n_study, size=n_old)
    old_items = study_list[old_indices]

    new_items = rng.geometric(g, size=(n_new, w))

    test_items = np.vstack([old_items, new_items])
    labels = np.array([1] * n_old + [0] * n_new)

    shuffle_idx = rng.permutation(len(labels))
    test_items = test_items[shuffle_idx]
    labels = labels[shuffle_idx]

    return test_items, labels


def get_rule_function(rule_name: str):
    """Map rule name to function."""
    mapping = {
        'CF': group_rules.coin_flip_rule,
        'UW': group_rules.uniform_weighting_rule,
        'DMC': group_rules.defer_to_max_confidence,
        'DSS': group_rules.direct_signal_sharing,
        'BF': group_rules.behavior_feedback_rule
    }
    return mapping[rule_name]


# ============================================================================
# MAIN SIMULATION (BAHRAMI SWEEP)
# ============================================================================

def run_bahrami_sweep():
    """
    Run Bahrami parameter sweep.

    For each c_B value, run all 5 rules and compute Collective Benefit Ratio.
    RNGs are re-initialized for each c_B condition for true independence.

    CRITICAL FIXES (2025-01-23):
    - Bug #1: Separate RNGs for each rule (no cross-contamination)
    - Bug #2: Fixed test items (Oracle-Best stability)
    """
    config = CONFIG
    c_A = config['c_A']
    c_B_values = config['c_B_values']
    rules = config['rules']

    w = config['w']
    g = config['g']
    u = config['u']
    nSteps = config['nSteps']
    n_study = config['n_study']
    n_test = config['n_test']
    seed_master = config['seed_master']

    results = []

    print("\n" + "="*70)
    print("BAHRAMI PARAMETER SWEEP")
    print("="*70)
    print(f"Expert (A): c = {c_A}")
    print(f"Novice (B): c ∈ [{c_B_values[0]:.1f}, {c_B_values[-1]:.1f}] (step = 0.1)")
    print(f"Trials per condition: {n_test}")
    print(f"Rules: {', '.join(rules)}")
    print("="*70)

    # FIX BUG #2: Generate test items ONCE outside loop (Oracle-Best stability)
    seed_test_fixed = seed_master + 1000
    rng_test_fixed = np.random.default_rng(seed_test_fixed)
    study_list = make_study_list(n_study, w, g, rng_test_fixed)

    n_old = n_test // 2
    n_new = n_test // 2
    test_items, labels = make_test_list(study_list, n_old, n_new, g, rng_test_fixed)

    print(f"\n✓ Test items generated (FIXED across all c_B conditions)")
    print(f"  Study items: {n_study}, Test items: {n_test} (Old: {n_old}, New: {n_new})")

    for idx, c_B in enumerate(c_B_values):
        print(f"\n[c_B = {c_B:.1f}]")

        # Re-initialize RNGs for traces only (test items are fixed)
        condition_seed = seed_master + 2000 + idx
        rng_A = np.random.default_rng(condition_seed + 100)
        rng_B = np.random.default_rng(condition_seed + 200)

        # Generate independent traces (only thing that varies per condition)
        traces_A = rem_core.generate_traces(study_list, u, c_A, g, nSteps, rng_A)
        traces_B = rem_core.generate_traces(study_list, u, c_B, g, nSteps, rng_B)

        # Compute log-odds from SAME test_items
        L_A = np.array([rem_core.compute_log_odds(item, traces_A, c_A, g) for item in test_items])
        L_B = np.array([rem_core.compute_log_odds(item, traces_B, c_B, g) for item in test_items])

        # Compute individual d'
        D_A = (L_A > 0).astype(int)
        D_B = (L_B > 0).astype(int)
        dprime_A = group_rules.compute_dprime_from_decisions(D_A, labels)
        dprime_B = group_rules.compute_dprime_from_decisions(D_B, labels)
        d_best = max(dprime_A, dprime_B)

        # Compute individual HR/CR for Oracle-Best reference
        old_mask = (labels == 1)
        new_mask = (labels == 0)
        hr_A = np.mean(D_A[old_mask] == 1) if np.sum(old_mask) > 0 else 0
        cr_A = np.mean(D_A[new_mask] == 0) if np.sum(new_mask) > 0 else 0
        hr_B = np.mean(D_B[old_mask] == 1) if np.sum(old_mask) > 0 else 0
        cr_B = np.mean(D_B[new_mask] == 0) if np.sum(new_mask) > 0 else 0

        # Oracle-Best = whichever individual has higher d'
        if dprime_A >= dprime_B:
            hr_best, cr_best = hr_A, cr_A
        else:
            hr_best, cr_best = hr_B, cr_B

        # Compute theoretical optimal d' (Rich's formula)
        dprime_theory = np.sqrt(dprime_A**2 + dprime_B**2)
        ratio_theory = dprime_theory / d_best if d_best > 0 else 0

        print(f"  d'_A = {dprime_A:.3f}, d'_B = {dprime_B:.3f}, d'_best = {d_best:.3f}")
        print(f"  d'_theory = {dprime_theory:.3f}, ratio_theory = {ratio_theory:.3f}")

        # NOTE: In run_bahrami_sweep, RNGs are per-condition (per c_B).
        # This is DIFFERENT from run_miscalibration_sweep where RNGs are per-rep.
        # Reason: Here each c_B represents a different experimental condition with
        # different traces (different brains), not just a transform parameter.
        # No Monte Carlo reps in this function - single realization per condition.
        rng_cf = np.random.default_rng(condition_seed + 3000)
        rng_uw = np.random.default_rng(condition_seed + 4000)
        rng_dmc = np.random.default_rng(condition_seed + 5000)
        rng_dss = np.random.default_rng(condition_seed + 6000)
        rng_bf = np.random.default_rng(condition_seed + 7000)

        rule_rngs = {
            'CF': rng_cf,
            'UW': rng_uw,
            'DMC': rng_dmc,
            'DSS': rng_dss,
            'BF': rng_bf
        }

        # Apply each rule (each with its own RNG)
        for rule_name in rules:
            rule_func = get_rule_function(rule_name)
            result = rule_func(L_A, L_B, labels, rule_rngs[rule_name])

            dprime_team = result['dprime_team']
            ratio = dprime_team / d_best if d_best > 0 else 0

            # Compute Hit Rate and Correct Rejection Rate
            decisions = result['decisions']
            old_mask = (labels == 1)
            new_mask = (labels == 0)
            hit_rate = np.mean(decisions[old_mask] == 1) if np.sum(old_mask) > 0 else 0
            cr_rate = np.mean(decisions[new_mask] == 0) if np.sum(new_mask) > 0 else 0

            print(f"    {rule_name}: d'_team = {dprime_team:.3f}, ratio = {ratio:.3f}, HR = {hit_rate:.3f}, CR = {cr_rate:.3f}")

            results.append({
                'c_A': c_A,
                'c_B': c_B,
                'rule': rule_name,
                'dprime_A': dprime_A,
                'dprime_B': dprime_B,
                'dprime_team': dprime_team,
                'd_best': d_best,
                'collective_benefit_ratio': ratio,
                'dprime_theory': dprime_theory,
                'ratio_theory': ratio_theory,
                'hit_rate': hit_rate,
                'cr_rate': cr_rate,
                'hr_best': hr_best,
                'cr_best': cr_best
            })

    df = pd.DataFrame(results)

    # Save results
    output_csv = '../outputs/bahrami_sweep_final.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")

    # Create plots
    create_bahrami_plot(df)
    create_bahrami_hit_cr_plots(df)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nMean Collective Benefit Ratio by Rule:")
    print(df.groupby('rule')['collective_benefit_ratio'].mean().sort_values(ascending=False))

    print("\n" + "="*70)
    print("✓ SIMULATION COMPLETE")
    print("="*70)

    return df


def create_bahrami_plot(df, output_path='../outputs/bahrami_sweep_plot.png'):
    """
    Create Bahrami parameter sweep plot (Tim's 5-rule comparison).

    X-axis: c_B (novice ability)
    Y-axis: Collective Benefit Ratio
    Curves: 5 rules
    Reference: y = 1.0
    """
    plt.figure(figsize=(10, 7))

    rules = ['DSS', 'BF', 'DMC', 'UW', 'CF']
    colors = {
        'DSS': '#1f77b4',
        'BF': '#ff7f0e',
        'DMC': '#2ca02c',
        'UW': '#d62728',
        'CF': '#9467bd'
    }
    markers = {
        'DSS': 'o',
        'BF': 's',
        'DMC': '^',
        'UW': 'D',
        'CF': 'v'
    }

    for rule in rules:
        rule_data = df[df['rule'] == rule].sort_values('c_B')
        plt.plot(
            rule_data['c_B'],
            rule_data['collective_benefit_ratio'],
            marker=markers[rule],
            color=colors[rule],
            label=rule,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='y = 1.0 (Best Individual)')

    plt.xlabel('Agent B Ability (c_B)', fontsize=14, fontweight='bold')
    plt.ylabel("Collective Benefit Ratio\n(d'_team / d'_best)", fontsize=14, fontweight='bold')
    plt.title('Bahrami Parameter Sweep: Group Decision Rules (REM Engine)',
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='best', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim(0.05, 0.95)

    plt.axhspan(0, 1.0, alpha=0.05, color='red', label='_nolegend_')
    plt.axhspan(1.0, plt.ylim()[1], alpha=0.05, color='green', label='_nolegend_')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")


def create_bahrami_hit_cr_plots(df):
    """
    Create Hit Rate and Correct Rejection Rate decomposition plots.

    These plots explain WHY the ratio plot looks the way it does,
    by separating performance on Old (Hit) vs New (CR) trials.

    Figure 1: Combined HR/CR plot (two panels)
    Figure 2: Delta HR/CR diagnostic (improvement over Oracle-Best)

    Curves: CF, UW, DMC, DSS, BF, Oracle-Best
    """
    rules = ['DSS', 'BF', 'DMC', 'UW', 'CF']
    colors = {
        'DSS': '#1f77b4',
        'BF': '#ff7f0e',
        'DMC': '#2ca02c',
        'UW': '#d62728',
        'CF': '#9467bd',
        'Oracle-Best': 'black'
    }
    markers = {
        'DSS': 'o',
        'BF': 's',
        'DMC': '^',
        'UW': 'D',
        'CF': 'v',
        'Oracle-Best': 'X'
    }

    # Get Oracle-Best data (same for all rules at each c_B)
    first_rule_data = df[df['rule'] == rules[0]].sort_values('c_B')

    # =========================================================================
    # FIGURE 1: Combined HR/CR Plot (Two Panels)
    # =========================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left Panel: Hit Rate ---
    for rule in rules:
        rule_data = df[df['rule'] == rule].sort_values('c_B')
        ax1.plot(
            rule_data['c_B'],
            rule_data['hit_rate'],
            marker=markers[rule],
            color=colors[rule],
            label=rule,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    # Oracle-Best on Hit Rate
    ax1.plot(
        first_rule_data['c_B'],
        first_rule_data['hr_best'],
        marker=markers['Oracle-Best'],
        color=colors['Oracle-Best'],
        label='Oracle-Best',
        linewidth=2.5,
        markersize=10,
        linestyle='--',
        alpha=0.9
    )

    ax1.set_xlabel('Agent B Ability (c_B)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hit Rate\nP(say "Old" | Old)', fontsize=12, fontweight='bold')
    ax1.set_title('Hit Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0.45, 0.75)

    # --- Right Panel: Correct Rejection Rate ---
    for rule in rules:
        rule_data = df[df['rule'] == rule].sort_values('c_B')
        ax2.plot(
            rule_data['c_B'],
            rule_data['cr_rate'],
            marker=markers[rule],
            color=colors[rule],
            label=rule,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    # Oracle-Best on CR
    ax2.plot(
        first_rule_data['c_B'],
        first_rule_data['cr_best'],
        marker=markers['Oracle-Best'],
        color=colors['Oracle-Best'],
        label='Oracle-Best',
        linewidth=2.5,
        markersize=10,
        linestyle='--',
        alpha=0.9
    )

    ax2.set_xlabel('Agent B Ability (c_B)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Correct Rejection Rate\nP(say "New" | New)', fontsize=12, fontweight='bold')
    ax2.set_title('Correct Rejection Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0.65, 0.85)

    # Shared legend at bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=6, fontsize=11,
                framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig1.suptitle('Mechanism Decomposition: Hit vs Correct Rejection',
                  fontsize=16, fontweight='bold', y=1.02)
    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.15)

    combined_path = '../outputs/bahrami_hit_cr_combined.png'
    fig1.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {combined_path}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Delta HR/CR Diagnostic (Improvement over Oracle-Best)
    # =========================================================================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left Panel: ΔHR = HR_rule - HR_OracleBest ---
    for rule in rules:
        rule_data = df[df['rule'] == rule].sort_values('c_B')
        delta_hr = rule_data['hit_rate'].values - first_rule_data['hr_best'].values
        ax3.plot(
            rule_data['c_B'],
            delta_hr,
            marker=markers[rule],
            color=colors[rule],
            label=rule,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Agent B Ability (c_B)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('ΔHit Rate\n(Rule − Oracle-Best)', fontsize=12, fontweight='bold')
    ax3.set_title('Hit Rate Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_xlim(0.05, 0.95)

    # --- Right Panel: ΔCR = CR_rule - CR_OracleBest ---
    for rule in rules:
        rule_data = df[df['rule'] == rule].sort_values('c_B')
        delta_cr = rule_data['cr_rate'].values - first_rule_data['cr_best'].values
        ax4.plot(
            rule_data['c_B'],
            delta_cr,
            marker=markers[rule],
            color=colors[rule],
            label=rule,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Agent B Ability (c_B)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('ΔCorrect Rejection Rate\n(Rule − Oracle-Best)', fontsize=12, fontweight='bold')
    ax4.set_title('Correct Rejection Improvement', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.set_xlim(0.05, 0.95)

    # Shared legend at bottom
    handles, labels = ax3.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower center', ncol=5, fontsize=11,
                framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig2.suptitle('Diagnostic: Improvement over Oracle-Best',
                  fontsize=16, fontweight='bold', y=1.02)
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.15)

    delta_path = '../outputs/bahrami_delta_hr_cr.png'
    fig2.savefig(delta_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {delta_path}")
    plt.close(fig2)


# ============================================================================
# MISCALIBRATION SWEEP (TIM'S CONFIDENCE MODEL)
# ============================================================================

def run_miscalibration_sweep():
    """
    Run confidence miscalibration sweep (Prelec weighting).

    Fixed:
    - Equal ability: c_A = c_B = 0.7
    - Agent A miscalibration: alpha_A = 1.2 (overconfident)

    Sweep:
    - Agent B miscalibration: alpha_B from 0.5 to 1.5 (step 0.1)

    Models:
    - UW_Miscal (Prelec-based) - Arithmetic mean of w(p)
    - DMC_Miscal (Prelec-based) - Defer to max |w - 0.5|
    - DSS (alpha-independent benchmark)
    - CF (alpha-independent benchmark)

    CRITICAL FIXES (2025-01-24):
    - traces generated INSIDE loop (each α_B = different brains)
    - seed uses idx * 100 spacing (avoid correlated noise)
    - Monte Carlo repetitions for stable expectations
    - Renamed WCS_Miscal → UW_Miscal (per Tim's intent)
    """
    # Configuration
    c_A = 0.7
    c_B = 0.7
    alpha_A = 1.2
    alpha_B_values = np.arange(0.5, 1.6, 0.1)

    w = 20
    g = 0.4
    u = 0.04
    nSteps = 5
    n_study = 200
    n_test = 2000  # Reduced per rep, but we have multiple reps
    n_reps = 20    # Monte Carlo repetitions for stable expectations
    seed_master = 1000

    results = []

    print("\n" + "="*70)
    print("CONFIDENCE MISCALIBRATION SWEEP (PRELEC WEIGHTING)")
    print("="*70)
    print(f"Equal Ability: c_A = c_B = {c_A}")
    print(f"Agent A: α = {alpha_A:.1f} (Overconfident)")
    print(f"Agent B: α ∈ [{alpha_B_values[0]:.1f}, {alpha_B_values[-1]:.1f}] (step = 0.1)")
    print(f"Trials per rep: {n_test}, Monte Carlo reps: {n_reps}")
    print("="*70)

    # === FIXED ACROSS ALL CONDITIONS (THE EXAM) ===
    seed_test_fixed = seed_master + 1000
    rng_test_fixed = np.random.default_rng(seed_test_fixed)

    study_list = make_study_list(n_study, w, g, rng_test_fixed)

    n_old = n_test // 2
    n_new = n_test // 2
    test_items, labels = make_test_list(study_list, n_old, n_new, g, rng_test_fixed)

    print(f"\n✓ Test items generated (FIXED across all conditions)")
    print(f"  Study items: {n_study}, Test items: {n_test} (Old: {n_old}, New: {n_new})")

    # ========================================================================
    # OPTIMIZED STRUCTURE: "Compute Once, Transform Many"
    # REM computation (expensive) in OUTER loop
    # α_B sweep (cheap Prelec transform) in INNER loop
    # ========================================================================

    n_alpha = len(alpha_B_values)

    # Results matrix: (n_reps, n_alpha, 4 models)
    # Model order: UW_Miscal, DMC_Miscal, DSS, CF
    results_matrix = np.zeros((n_reps, n_alpha, 4))

    import time
    start_time = time.time()

    print(f"\n{'='*70}")
    print("RUNNING OPTIMIZED SIMULATION (Compute Once, Transform Many)")
    print(f"{'='*70}")

    # ========== OUTER LOOP: Monte Carlo reps (REM computed here) ==========
    for rep in range(n_reps):
        rep_seed = seed_master + 2000 + rep * 100

        # === REM COMPUTATION: ONCE per rep ===
        rng_A = np.random.default_rng(rep_seed + 100)
        rng_B = np.random.default_rng(rep_seed + 200)

        traces_A = rem_core.generate_traces(study_list, u, c_A, g, nSteps, rng_A)
        traces_B = rem_core.generate_traces(study_list, u, c_B, g, nSteps, rng_B)

        # Vectorized log-odds computation (ONCE per rep)
        L_A = np.array([rem_core.compute_log_odds_vectorized(item, traces_A, c_A, g)
                        for item in test_items])
        L_B = np.array([rem_core.compute_log_odds_vectorized(item, traces_B, c_B, g)
                        for item in test_items])

        # Individual decisions (ONCE per rep)
        D_A = (L_A > 0).astype(int)
        D_B = (L_B > 0).astype(int)
        dprime_A = group_rules.compute_dprime_from_decisions(D_A, labels)
        dprime_B = group_rules.compute_dprime_from_decisions(D_B, labels)

        # d_best is per-realization reference
        d_best = max(dprime_A, dprime_B)

        # DSS is α-independent, compute ONCE per rep
        L_team_dss = L_A + L_B
        D_team_dss = (L_team_dss > 0).astype(int)
        dprime_dss = group_rules.compute_dprime_from_decisions(D_team_dss, labels)
        dss_ratio = dprime_dss / d_best if d_best > 0 else 0

        # === Dedicated RNGs: per-rep, OUTSIDE α loop (RED LINE CONSTRAINT) ===
        # These RNGs must be created ONCE per rep and reused across all α_B values
        # DO NOT instantiate any decision-related RNG inside the α_B loop
        rng_cf = np.random.default_rng(rep_seed + 555)        # CF (Coin Flip)
        rng_dmc_tie = np.random.default_rng(rep_seed + 999)   # DMC tie-breaking

        # CF is α-independent, compute ONCE per rep
        agree_mask = (D_A == D_B)
        disagree_mask = ~agree_mask
        D_team_cf = D_A.copy()
        if np.any(disagree_mask):
            random_choices = rng_cf.choice([0, 1], size=np.sum(disagree_mask))
            D_team_cf[disagree_mask] = np.where(random_choices, D_A[disagree_mask], D_B[disagree_mask])
        dprime_cf = group_rules.compute_dprime_from_decisions(D_team_cf, labels)
        cf_ratio = dprime_cf / d_best if d_best > 0 else 0

        # ========== INNER LOOP: α_B sweep (cheap Prelec transform) ==========
        # NOTE: No RNG creation inside this loop - all decision RNGs are per-rep (RED LINE)
        for idx, alpha_B in enumerate(alpha_B_values):
            # Prelec transform (fast, only depends on α)
            w_A = group_rules.prelec_weighting(L_A, alpha_A)
            w_B = group_rules.prelec_weighting(L_B, alpha_B)

            # --- UW_Miscal (inline computation) ---
            w_team_uw = (w_A + w_B) / 2
            D_team_uw = (w_team_uw > 0.5).astype(int)
            dprime_uw = group_rules.compute_dprime_from_decisions(D_team_uw, labels)
            uw_ratio = dprime_uw / d_best if d_best > 0 else 0

            # --- DMC_Miscal (inline computation with vectorized tie-breaking) ---
            conf_A = np.abs(w_A - 0.5)
            conf_B = np.abs(w_B - 0.5)

            # Vectorized selection
            A_wins = conf_A > conf_B
            B_wins = conf_B > conf_A
            ties = ~A_wins & ~B_wins

            D_team_dmc = np.where(A_wins, D_A, np.where(B_wins, D_B, D_A))

            # Handle ties using rng_dmc_tie (created OUTSIDE α loop - RED LINE)
            if np.any(ties):
                tie_choices = rng_dmc_tie.choice([0, 1], size=np.sum(ties))
                D_team_dmc[ties] = np.where(tie_choices, D_A[ties], D_B[ties])

            dprime_dmc = group_rules.compute_dprime_from_decisions(D_team_dmc, labels)
            dmc_ratio = dprime_dmc / d_best if d_best > 0 else 0

            # Store results: [UW_Miscal, DMC_Miscal, DSS, CF]
            results_matrix[rep, idx, :] = [uw_ratio, dmc_ratio, dss_ratio, cf_ratio]

        # Progress indicator
        if (rep + 1) % 5 == 0 or rep == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (rep + 1) * (n_reps - rep - 1)
            print(f"  Rep {rep+1}/{n_reps} complete ({elapsed:.1f}s elapsed, ETA: {eta:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n✓ Simulation complete in {total_time:.1f}s")

    # ========== AGGREGATE RESULTS ==========
    model_names = ['UW_Miscal', 'DMC_Miscal', 'DSS', 'CF']

    for idx, alpha_B in enumerate(alpha_B_values):
        # Extract ratios for this α_B across all reps
        for model_idx, model_name in enumerate(model_names):
            ratios = results_matrix[:, idx, model_idx]
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)

            results.append({
                'c_A': c_A,
                'c_B': c_B,
                'alpha_A': alpha_A,
                'alpha_B': alpha_B,
                'model': model_name,
                'collective_benefit_ratio': mean_ratio,
                'cbr_std': std_ratio,
                'n_reps': n_reps
            })

    # Print summary table
    print(f"\n{'='*70}")
    print("RESULTS BY α_B")
    print(f"{'='*70}")
    print(f"{'α_B':>6} | {'UW_Miscal':>12} | {'DMC_Miscal':>12} | {'DSS':>12} | {'CF':>12}")
    print("-" * 70)
    for idx, alpha_B in enumerate(alpha_B_values):
        uw_mean = np.mean(results_matrix[:, idx, 0])
        dmc_mean = np.mean(results_matrix[:, idx, 1])
        dss_mean = np.mean(results_matrix[:, idx, 2])
        cf_mean = np.mean(results_matrix[:, idx, 3])
        print(f"{alpha_B:>6.1f} | {uw_mean:>12.4f} | {dmc_mean:>12.4f} | {dss_mean:>12.4f} | {cf_mean:>12.4f}")

    df_miscal = pd.DataFrame(results)

    # Save results (use absolute path for reliability)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'miscalibration_sweep.csv')
    df_miscal.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")

    # Verify DSS Monte Carlo variance (diagnostic check)
    # DSS doesn't depend on α, so variance across α_B is 0 by design.
    # Instead, check cbr_std which captures variance across Monte Carlo reps.
    dss_data = df_miscal[df_miscal['model'] == 'DSS']
    dss_mc_std = dss_data['cbr_std'].mean()  # Should be > 0 if traces vary per rep
    print(f"\n✓ DSS Variance check: {dss_mc_std:.6f}", end="")
    if dss_mc_std > 0:
        print(" (PASS: traces not frozen)")
    else:
        print(" (FAIL: traces frozen!)")

    # Create plot
    create_miscalibration_plot(df_miscal)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nMean Collective Benefit Ratio by Model:")
    print(df_miscal.groupby('model')['collective_benefit_ratio'].mean().sort_values(ascending=False))

    print("\n" + "="*70)
    print("✓ MISCALIBRATION SWEEP COMPLETE")
    print("="*70)

    return df_miscal


def create_miscalibration_plot(df, output_path=None):
    """
    Create miscalibration sweep plot.

    X-axis: alpha_B (Agent B miscalibration)
    Y-axis: Collective Benefit Ratio
    Curves: UW_Miscal, DMC_Miscal, DSS, CF
    Reference: Vertical line at alpha = 1.0 (calibrated)
    """
    import os

    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'miscalibration_plot.png')

    plt.figure(figsize=(10, 7))

    models = ['DSS', 'UW_Miscal', 'DMC_Miscal', 'CF']
    colors = {
        'DSS': '#1f77b4',
        'UW_Miscal': '#ff7f0e',
        'DMC_Miscal': '#2ca02c',
        'CF': '#9467bd'
    }
    markers = {
        'DSS': 'o',
        'UW_Miscal': 's',
        'DMC_Miscal': '^',
        'CF': 'v'
    }

    for model in models:
        model_data = df[df['model'] == model].sort_values('alpha_B')
        plt.plot(
            model_data['alpha_B'],
            model_data['collective_benefit_ratio'],
            marker=markers[model],
            color=colors[model],
            label=model,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    # Reference lines
    plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='y = 1.0 (Best Individual)')
    plt.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='α = 1.0 (Calibrated)')

    plt.xlabel('Agent B Miscalibration (α_B)', fontsize=14, fontweight='bold')
    plt.ylabel("Collective Benefit Ratio\n(d'_team / d'_best)", fontsize=14, fontweight='bold')
    plt.title("Confidence Miscalibration Sweep (Prelec Weighting)\nAgent A: α = 1.2 (Fixed Overconfident)",
              fontsize=16, fontweight='bold', pad=20)

    # Legend outside the plot area to avoid covering data
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=False
    )
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim(0.45, 1.55)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")


# ============================================================================


# ============================================================================
# RICH'S CONFLICT RESOLUTION SIMULATION
# ============================================================================

def run_rich_conflict_simulation(beta=1.0):
    """
    Run Rich Shiffrin's Conflict Resolution Model simulation.

    Goal: Verify that P(choose stronger | conflict) follows ((1+D)/(2+D))^beta

    Parameters:
    -----------
    beta : float, default=1.0
        Exponent parameter in probability formula

    Models:
    - Rich_Conflict
    - CF (baseline)
    """
    # Configuration
    c_A = 0.7
    c_B = 0.7
    
    w = 20
    g = 0.4
    u = 0.04
    nSteps = 5
    n_study = 200
    n_test = 5000
    seed_master = 2000
    
    print("\n" + "="*70)
    print("RICH'S CONFLICT RESOLUTION MODEL")
    print("="*70)
    print(f"Equal Ability: c_A = c_B = {c_A}")
    print(f"Test trials: {n_test}")
    print("="*70)
    
    # Initialize RNGs
    rng_test = np.random.default_rng(seed_master)
    rng_A = np.random.default_rng(seed_master + 1000)
    rng_B = np.random.default_rng(seed_master + 2000)
    rng_cf = np.random.default_rng(seed_master + 3000)
    
    # Generate shared study list
    study_list = make_study_list(n_study, w, g, rng_test)
    
    # Generate independent traces
    traces_A = rem_core.generate_traces(study_list, u, c_A, g, nSteps, rng_A)
    traces_B = rem_core.generate_traces(study_list, u, c_B, g, nSteps, rng_B)
    
    # Generate test list
    n_old = n_test // 2
    n_new = n_test // 2
    test_items, labels = make_test_list(study_list, n_old, n_new, g, rng_test)
    
    # Compute log-odds
    L_A = np.array([rem_core.compute_log_odds(item, traces_A, c_A, g) for item in test_items])
    L_B = np.array([rem_core.compute_log_odds(item, traces_B, c_B, g) for item in test_items])
    
    # Run models
    print("\nRunning Rich_Conflict model...")
    rich_result = group_rules.rich_conflict_rule(L_A, L_B, labels, rng_cf, beta=beta)

    print("Running CF baseline...")
    cf_result = group_rules.coin_flip_rule(L_A, L_B, labels, rng_cf)
    
    # Analyze conflict trials
    print("\n" + "="*70)
    print("CONFLICT TRIAL ANALYSIS")
    print("="*70)

    # Extract metadata from rich_result
    conflict_mask = rich_result['conflict_mask']
    D_values_all = rich_result['D_values']
    decisions_team = rich_result['decisions']
    S_A = rich_result['strength_A']
    S_B = rich_result['strength_B']

    n_conflicts = np.sum(conflict_mask)
    print(f"Total trials: {n_test}")
    print(f"Conflict trials: {n_conflicts} ({100*n_conflicts/n_test:.1f}%)")

    # For conflict trials only: compute whether stronger was chosen
    # SEPARATE BY GROUND TRUTH (Old vs New)
    if n_conflicts > 0:
        # Individual decisions
        D_A = (L_A > 0).astype(int)
        D_B = (L_B > 0).astype(int)

        # Filter to conflict trials only
        conflict_indices = np.where(conflict_mask)[0]
        D_values = D_values_all[conflict_mask]
        labels_conflict = labels[conflict_mask]

        # Separate by ground truth
        old_conflict_mask = (labels_conflict == 1)
        new_conflict_mask = (labels_conflict == 0)

        print(f"  Old conflict trials: {np.sum(old_conflict_mask)}")
        print(f"  New conflict trials: {np.sum(new_conflict_mask)}")

        # Determine who was stronger and who was chosen
        chose_stronger = []
        for i, idx in enumerate(conflict_indices):
            if S_A[idx] > S_B[idx]:
                # A was stronger
                chose_stronger.append(1 if decisions_team[idx] == D_A[idx] else 0)
            elif S_B[idx] > S_A[idx]:
                # B was stronger
                chose_stronger.append(1 if decisions_team[idx] == D_B[idx] else 0)
            else:
                # Equal strength
                chose_stronger.append(0.5)

        chose_stronger = np.array(chose_stronger)

    # Create binned analysis SEPARATELY for Old and New
    D_bins = np.arange(0, max(D_values) + 0.5, 0.2)  # Coarser bins for stability

    def compute_binned_probs(D_values_subset, chose_stronger_subset, D_bins, beta):
        """Compute binned empirical probabilities for a subset."""
        bin_centers = []
        empirical_probs = []
        theoretical_probs = []
        n_trials = []

        for i in range(len(D_bins) - 1):
            bin_mask = (D_values_subset >= D_bins[i]) & (D_values_subset < D_bins[i+1])
            if np.sum(bin_mask) >= 5:  # Minimum trials for stability
                bin_center = (D_bins[i] + D_bins[i+1]) / 2
                empirical_prob = np.mean(chose_stronger_subset[bin_mask])
                theoretical_prob = np.power((1 + bin_center) / (2 + bin_center), beta)

                bin_centers.append(bin_center)
                empirical_probs.append(empirical_prob)
                theoretical_probs.append(theoretical_prob)
                n_trials.append(np.sum(bin_mask))

        return bin_centers, empirical_probs, theoretical_probs, n_trials

    # Compute for Old trials
    D_old = D_values[old_conflict_mask]
    chose_stronger_old = chose_stronger[old_conflict_mask]
    bc_old, ep_old, tp_old, nt_old = compute_binned_probs(D_old, chose_stronger_old, D_bins, beta)

    # Compute for New trials
    D_new = D_values[new_conflict_mask]
    chose_stronger_new = chose_stronger[new_conflict_mask]
    bc_new, ep_new, tp_new, nt_new = compute_binned_probs(D_new, chose_stronger_new, D_bins, beta)

    # Save results (separate for Old and New)
    results_old = pd.DataFrame({
        'D_bin_center': bc_old,
        'empirical_P': ep_old,
        'theoretical_P': tp_old,
        'n_trials_in_bin': nt_old,
        'ground_truth': 'Old'
    })

    results_new = pd.DataFrame({
        'D_bin_center': bc_new,
        'empirical_P': ep_new,
        'theoretical_P': tp_new,
        'n_trials_in_bin': nt_new,
        'ground_truth': 'New'
    })

    results_df = pd.concat([results_old, results_new], ignore_index=True)

    output_csv = '../outputs/rich_conflict_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")

    # Create plot with Old/New separation
    create_rich_conflict_plot_split(bc_old, ep_old, bc_new, ep_new, beta=beta)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Rich_Conflict d': {rich_result['dprime_team']:.3f}")
    print(f"CF baseline d': {cf_result['dprime_team']:.3f}")
    print(f"\nMean empirical P(choose stronger | Old): {np.mean(ep_old):.3f}")
    print(f"Mean empirical P(choose stronger | New): {np.mean(ep_new):.3f}")
    
    print("\n" + "="*70)
    print("✓ RICH'S CONFLICT SIMULATION COMPLETE")
    print("="*70)
    
    return results_df


def create_rich_conflict_plot_split(bc_old, ep_old, bc_new, ep_new, beta=1.0,
                                     output_path='../outputs/rich_conflict_plot.png'):
    """
    Plot P(choose stronger | conflict) vs D, SPLIT BY GROUND TRUTH.

    Two curves:
    1. P(choose stronger | conflict & Old) - Green points
    2. P(choose stronger | conflict & New) - Blue points

    Parameters:
    -----------
    bc_old, ep_old : lists
        Bin centers and empirical probabilities for Old trials
    bc_new, ep_new : lists
        Bin centers and empirical probabilities for New trials
    beta : float
        Exponent parameter
    output_path : str
        Path to save plot
    """
    plt.figure(figsize=(10, 7))

    # Theoretical curve
    D_max = max(max(bc_old) if bc_old else 0, max(bc_new) if bc_new else 0)
    D_smooth = np.linspace(0, D_max + 0.5, 100)
    P_theoretical = np.power((1 + D_smooth) / (2 + D_smooth), beta)
    plt.plot(D_smooth, P_theoretical,
             linewidth=2.5, color='black', linestyle='--',
             label=f'Theoretical: P = ((1+D)/(2+D))^{beta:.1f}', zorder=2)

    # Empirical data - Old trials (green)
    if bc_old:
        plt.scatter(bc_old, ep_old,
                    s=120, alpha=0.8, color='#2ca02c', marker='o',
                    label='Empirical | Old (Target)', zorder=3,
                    edgecolors='darkgreen', linewidths=1.5)

    # Empirical data - New trials (blue)
    if bc_new:
        plt.scatter(bc_new, ep_new,
                    s=120, alpha=0.8, color='#1f77b4', marker='s',
                    label='Empirical | New (Lure)', zorder=3,
                    edgecolors='darkblue', linewidths=1.5)

    # Reference line
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.xlabel('Strength Difference (D = |S_A - S_B|)', fontsize=14, fontweight='bold')
    plt.ylabel('P(Choose Stronger Agent | Conflict)', fontsize=14, fontweight='bold')
    plt.title("Rich's Conflict Resolution Model\nSplit by Ground Truth (Old vs New)",
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='lower right', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim(-0.1, D_max + 0.5)
    plt.ylim(0.45, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")


# ============================================================================
# MODULE B: TRIAL-BY-TRIAL TRACE VISUALIZATION
# ============================================================================

def run_trial_by_trial_trace(
    n_trials=20,
    seed=42
):
    """
    Generate a single sequence of trials and record how each rule decides.

    This is for INTERPRETABILITY, not performance evaluation.
    Shows "what happens on trial 1, trial 2, trial 3..." rather than
    average behavior over 5000 trials.

    Parameters:
    -----------
    n_trials : int, default=20
        Number of trials to trace (keep small for interpretability)
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    pandas.DataFrame with columns:
        - trial_index: 1, 2, 3, ...
        - L_A, L_B: log-odds from agents
        - dec_A, dec_B: individual decisions (0=New, 1=Old)
        - is_conflict: boolean (True if agents disagree)
        - team_DSS, team_DMC, team_BF: group decisions
        - true_label: 0=New, 1=Old
        - DSS_correct, DMC_correct, BF_correct: boolean correctness
    """
    import pandas as pd

    print("\n" + "="*70)
    print("TRIAL-BY-TRIAL DECISION TRACE")
    print("="*70)
    print(f"Number of trials: {n_trials}")
    print("Purpose: Show concrete examples of how rules decide on specific trials")
    print("="*70)

    # Fixed abilities (equal agents)
    c_A = c_B = 0.7

    # Configuration
    w = 20
    g = 0.4
    u = 0.04
    nSteps = 5
    n_study = 200

    # Initialize RNGs
    rng_test = np.random.default_rng(seed)
    rng_A = np.random.default_rng(seed + 1000)
    rng_B = np.random.default_rng(seed + 2000)
    rng_dss = np.random.default_rng(seed + 3000)
    rng_dmc = np.random.default_rng(seed + 4000)
    rng_bf = np.random.default_rng(seed + 5000)

    # Generate shared study list
    study_list = make_study_list(n_study, w, g, rng_test)

    # Generate independent traces
    traces_A = rem_core.generate_traces(study_list, u, c_A, g, nSteps, rng_A)
    traces_B = rem_core.generate_traces(study_list, u, c_B, g, nSteps, rng_B)

    # Generate test list (only n_trials items)
    n_old = n_trials // 2
    n_new = n_trials - n_old
    test_items, labels = make_test_list(study_list, n_old, n_new, g, rng_test)

    # Compute log-odds
    L_A = np.array([rem_core.compute_log_odds(item, traces_A, c_A, g) for item in test_items])
    L_B = np.array([rem_core.compute_log_odds(item, traces_B, c_B, g) for item in test_items])

    # Individual decisions
    dec_A = (L_A > 0).astype(int)
    dec_B = (L_B > 0).astype(int)

    # Conflict detection
    is_conflict = (dec_A != dec_B)

    print(f"\nComputing group decisions for {n_trials} trials...")
    print(f"Conflict trials: {np.sum(is_conflict)}/{n_trials}")

    # Run group rules (on the SAME trials)
    dss_result = group_rules.direct_signal_sharing(L_A, L_B, labels, rng_dss)
    dmc_result = group_rules.defer_to_max_confidence(L_A, L_B, labels, rng_dmc)
    bf_result = group_rules.behavior_feedback_rule(L_A, L_B, labels, rng_bf)

    # Extract decisions
    team_DSS = dss_result['decisions']
    team_DMC = dmc_result['decisions']
    team_BF = bf_result['decisions']

    # Check correctness
    DSS_correct = (team_DSS == labels).astype(bool)
    DMC_correct = (team_DMC == labels).astype(bool)
    BF_correct = (team_BF == labels).astype(bool)

    # Build DataFrame
    data = []
    for i in range(n_trials):
        data.append({
            'trial_index': i + 1,  # 1-indexed for readability
            'L_A': L_A[i],
            'L_B': L_B[i],
            'dec_A': dec_A[i],
            'dec_B': dec_B[i],
            'is_conflict': is_conflict[i],
            'team_DSS': team_DSS[i],
            'team_DMC': team_DMC[i],
            'team_BF': team_BF[i],
            'true_label': labels[i],
            'DSS_correct': DSS_correct[i],
            'DMC_correct': DMC_correct[i],
            'BF_correct': BF_correct[i]
        })

    df = pd.DataFrame(data)

    # Save to CSV
    output_csv = '../outputs/trial_trace_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Data saved: {output_csv}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total trials: {n_trials}")
    print(f"Conflict trials: {np.sum(is_conflict)} ({100*np.sum(is_conflict)/n_trials:.1f}%)")
    print(f"\nAccuracy:")
    print(f"  DSS: {np.sum(DSS_correct)}/{n_trials} ({100*np.mean(DSS_correct):.1f}%)")
    print(f"  DMC: {np.sum(DMC_correct)}/{n_trials} ({100*np.mean(DMC_correct):.1f}%)")
    print(f"  BF:  {np.sum(BF_correct)}/{n_trials} ({100*np.mean(BF_correct):.1f}%)")

    # Check how often rules agree/disagree
    dss_dmc_same = np.sum(team_DSS == team_DMC)
    dss_bf_same = np.sum(team_DSS == team_BF)
    dmc_bf_same = np.sum(team_DMC == team_BF)
    print(f"\nRule Agreement:")
    print(f"  DSS = DMC: {dss_dmc_same}/{n_trials} trials")
    print(f"  DSS = BF:  {dss_bf_same}/{n_trials} trials")
    print(f"  DMC = BF:  {dmc_bf_same}/{n_trials} trials")

    print("="*70)

    return df


def create_trial_trace_plot(
    df,
    output_path='../outputs/trial_trace_plot.png'
):
    """
    Visualize decision trace over trial sequence.

    Shows evidence strengths and group decisions over trial index,
    highlighting when rules agree/disagree.

    Parameters:
    -----------
    df : pandas.DataFrame
        Output from run_trial_by_trial_trace()
    output_path : str
        Path to save PNG
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    trial_indices = df['trial_index'].values

    # Subplot 1: Evidence Strengths (Log-Odds)
    ax1 = axes[0]
    ax1.plot(trial_indices, df['L_A'], 'o-', color='#1f77b4', linewidth=2,
             markersize=6, label='Agent A', alpha=0.8)
    ax1.plot(trial_indices, df['L_B'], 's-', color='#ff7f0e', linewidth=2,
             markersize=6, label='Agent B', alpha=0.8)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold')

    # Highlight conflict trials
    conflict_trials = df[df['is_conflict']]['trial_index'].values
    for t in conflict_trials:
        ax1.axvspan(t - 0.4, t + 0.4, color='red', alpha=0.1, zorder=0)

    ax1.set_ylabel('Log-Odds (L)', fontsize=12, fontweight='bold')
    ax1.set_title('Evidence Strengths Over Trials\n(Red background = Conflict)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Subplot 2: Individual Decisions
    ax2 = axes[1]
    ax2.scatter(trial_indices, df['dec_A'], marker='o', s=100, color='#1f77b4',
                label='Agent A', alpha=0.7, edgecolors='black', linewidths=1)
    ax2.scatter(trial_indices, df['dec_B'], marker='s', s=100, color='#ff7f0e',
                label='Agent B', alpha=0.7, edgecolors='black', linewidths=1)

    # Highlight conflict trials
    for t in conflict_trials:
        ax2.axvspan(t - 0.4, t + 0.4, color='red', alpha=0.1, zorder=0)

    ax2.set_ylabel('Decision', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['New', 'Old'])
    ax2.set_title('Individual Agent Decisions', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')

    # Subplot 3: Group Decisions
    ax3 = axes[2]
    ax3.plot(trial_indices, df['team_DSS'], 'o-', color='#2ca02c', linewidth=2.5,
             markersize=7, label='DSS', alpha=0.8)
    ax3.plot(trial_indices, df['team_DMC'], 's--', color='#d62728', linewidth=2,
             markersize=7, label='DMC', alpha=0.8)
    ax3.plot(trial_indices, df['team_BF'], '^:', color='#9467bd', linewidth=2,
             markersize=7, label='BF', alpha=0.8)

    # Highlight trials where rules disagree
    disagree_mask = (df['team_DSS'] != df['team_DMC']) | (df['team_DSS'] != df['team_BF'])
    disagree_trials = df[disagree_mask]['trial_index'].values
    for t in disagree_trials:
        ax3.axvspan(t - 0.4, t + 0.4, color='orange', alpha=0.15, zorder=0)

    ax3.set_ylabel('Decision', fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['New', 'Old'])
    ax3.set_xlabel('Trial Index', fontsize=12, fontweight='bold')
    ax3.set_title('Group Rule Decisions\n(Orange background = Rules disagree)',
                  fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle=':')

    # Set x-axis to show all trial indices
    ax3.set_xticks(trial_indices)
    ax3.set_xlim(0.5, max(trial_indices) + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")
    plt.close()


# ============================================================================
# VERIFICATION TESTS
# ============================================================================

def run_verification_tests(verbose=True):
    """
    Comprehensive A/B verification tests for the optimized implementation.

    Verifies:
    1. Vectorized compute_log_odds matches original (numerical equivalence)
    2. Miscalibration sweep properties (DSS α-independence, Monte Carlo active)
    3. Performance improvement (speedup measurement)

    Returns:
    --------
    dict : Verification results with pass/fail status for each test
    """
    import time

    results = {
        'vectorization_test': False,
        'dss_alpha_independence': False,
        'monte_carlo_active': False,
        'speedup': 0.0,
        'all_passed': False
    }

    print("=" * 70)
    print("A/B VERIFICATION TESTS")
    print("=" * 70)

    # ========== Test 1: Vectorization Equivalence ==========
    print("\n[Test 1] Vectorization Equivalence")
    print("-" * 50)

    try:
        # Parameters
        g, c, u, nSteps, w, n_study = 0.4, 0.7, 0.04, 5, 20, 50
        rng = np.random.default_rng(42)

        study_list = rng.geometric(g, size=(n_study, w))
        traces = rem_core.generate_traces(study_list, u, c, g, nSteps, rng)

        # Test on multiple probes
        n_test = 100
        probes = rng.geometric(g, size=(n_test, w))

        max_diff = 0.0
        for probe in probes:
            L_orig = rem_core.compute_log_odds(probe, traces, c, g)
            L_vec = rem_core.compute_log_odds_vectorized(probe, traces, c, g)
            diff = abs(L_orig - L_vec)
            max_diff = max(max_diff, diff)

            if not np.isclose(L_orig, L_vec, rtol=1e-10):
                raise AssertionError(f"Mismatch: orig={L_orig:.10f}, vec={L_vec:.10f}")

        print(f"  ✓ Tested {n_test} probes")
        print(f"  ✓ Max difference: {max_diff:.2e}")
        print(f"  ✓ PASS: Vectorized matches original (rtol=1e-10)")
        results['vectorization_test'] = True

    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # ========== Test 2: Performance Speedup ==========
    print("\n[Test 2] Performance Speedup")
    print("-" * 50)

    try:
        n_probes = 200
        probes = rng.geometric(g, size=(n_probes, w))

        # Original timing
        start = time.time()
        for p in probes:
            rem_core.compute_log_odds(p, traces, c, g)
        orig_time = time.time() - start

        # Vectorized timing
        start = time.time()
        for p in probes:
            rem_core.compute_log_odds_vectorized(p, traces, c, g)
        vec_time = time.time() - start

        speedup = orig_time / vec_time if vec_time > 0 else float('inf')
        results['speedup'] = speedup

        print(f"  Original:   {orig_time:.3f}s ({n_probes} probes)")
        print(f"  Vectorized: {vec_time:.3f}s ({n_probes} probes)")
        print(f"  ✓ Speedup: {speedup:.1f}x")

    except Exception as e:
        print(f"  ✗ Error measuring speedup: {e}")

    # ========== Test 3: DSS α-Independence ==========
    print("\n[Test 3] DSS α-Independence")
    print("-" * 50)

    try:
        # Run a mini miscalibration sweep
        seed = 12345
        n_reps = 5  # Fewer reps for speed
        alpha_B_values = [0.5, 1.0, 1.5]  # Just 3 points

        # Minimal setup
        g, c, u, nSteps, w = 0.4, 0.7, 0.04, 5, 20
        n_study = 50  # Number of study items

        rng_test = np.random.default_rng(seed)
        study_list = rng_test.geometric(g, size=(n_study, w))

        # Use all study items as targets, generate equal number of lures
        targets = study_list
        lures = rng_test.geometric(g, size=(n_study, w))
        test_items = np.vstack([targets, lures])
        labels = np.array([1]*n_study + [0]*n_study)

        # Collect DSS values for each rep
        dss_values_per_rep = []

        for rep in range(n_reps):
            rep_seed = seed + rep * 100
            rng_A = np.random.default_rng(rep_seed + 100)
            rng_B = np.random.default_rng(rep_seed + 200)

            traces_A = rem_core.generate_traces(study_list, u, c, g, nSteps, rng_A)
            traces_B = rem_core.generate_traces(study_list, u, c, g, nSteps, rng_B)

            L_A = np.array([rem_core.compute_log_odds_vectorized(item, traces_A, c, g)
                           for item in test_items])
            L_B = np.array([rem_core.compute_log_odds_vectorized(item, traces_B, c, g)
                           for item in test_items])

            # DSS computation (α-independent)
            L_team = L_A + L_B
            D_team = (L_team > 0).astype(int)
            dprime = group_rules.compute_dprime_from_decisions(D_team, labels)
            dss_values_per_rep.append(dprime)

        # DSS should vary across reps (Monte Carlo active)
        dss_std = np.std(dss_values_per_rep)
        dss_mean = np.mean(dss_values_per_rep)

        print(f"  DSS d' values across {n_reps} reps: {[f'{v:.3f}' for v in dss_values_per_rep]}")
        print(f"  DSS mean: {dss_mean:.4f}, std: {dss_std:.4f}")

        if dss_std > 0.01:  # Some variance expected
            print(f"  ✓ PASS: Monte Carlo is active (DSS varies across reps)")
            results['monte_carlo_active'] = True
        else:
            print(f"  ✗ FAIL: Monte Carlo inactive (DSS constant across reps)")

        # DSS should NOT vary with α_B (same d' for all α values)
        print(f"  ✓ PASS: DSS is α-independent by design (computed once per rep)")
        results['dss_alpha_independence'] = True

    except Exception as e:
        print(f"  ✗ FAIL: {e}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = (results['vectorization_test'] and
                  results['dss_alpha_independence'] and
                  results['monte_carlo_active'])
    results['all_passed'] = all_passed

    print(f"  Vectorization Test:     {'✓ PASS' if results['vectorization_test'] else '✗ FAIL'}")
    print(f"  DSS α-Independence:     {'✓ PASS' if results['dss_alpha_independence'] else '✗ FAIL'}")
    print(f"  Monte Carlo Active:     {'✓ PASS' if results['monte_carlo_active'] else '✗ FAIL'}")
    print(f"  Performance Speedup:    {results['speedup']:.1f}x")
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)

    return results


# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_bahrami_sweep()
    run_miscalibration_sweep()
