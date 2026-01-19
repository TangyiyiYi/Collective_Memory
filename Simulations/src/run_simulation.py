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
    'n_test': 2000,
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

    for idx, c_B in enumerate(c_B_values):
        print(f"\n[c_B = {c_B:.1f}]")

        # Re-initialize RNGs for each c_B condition
        condition_seed = seed_master + idx
        master_rng = np.random.default_rng(condition_seed)

        seed_A = int(master_rng.integers(0, 2**31 - 1))
        seed_B = int(master_rng.integers(0, 2**31 - 1))
        seed_test = int(master_rng.integers(0, 2**31 - 1))
        seed_cf = int(master_rng.integers(0, 2**31 - 1))

        rng_A = np.random.default_rng(seed_A)
        rng_B = np.random.default_rng(seed_B)
        rng_test = np.random.default_rng(seed_test)
        rng_cf = np.random.default_rng(seed_cf)

        # Generate shared study list
        study_list = make_study_list(n_study, w, g, rng_test)

        # Generate independent traces
        traces_A = rem_core.generate_traces(study_list, u, c_A, g, nSteps, rng_A)
        traces_B = rem_core.generate_traces(study_list, u, c_B, g, nSteps, rng_B)

        # Generate test list (balanced Old/New)
        n_old = n_test // 2
        n_new = n_test // 2
        test_items, labels = make_test_list(study_list, n_old, n_new, g, rng_test)

        # Compute log-odds for all test items
        L_A = np.array([rem_core.compute_log_odds(item, traces_A, c_A, g) for item in test_items])
        L_B = np.array([rem_core.compute_log_odds(item, traces_B, c_B, g) for item in test_items])

        # Compute individual d'
        D_A = (L_A > 0).astype(int)
        D_B = (L_B > 0).astype(int)
        dprime_A = group_rules.compute_dprime_from_decisions(D_A, labels)
        dprime_B = group_rules.compute_dprime_from_decisions(D_B, labels)
        d_best = max(dprime_A, dprime_B)

        # Compute theoretical optimal d' (Rich's formula)
        dprime_theory = np.sqrt(dprime_A**2 + dprime_B**2)
        ratio_theory = dprime_theory / d_best if d_best > 0 else 0

        print(f"  d'_A = {dprime_A:.3f}, d'_B = {dprime_B:.3f}, d'_best = {d_best:.3f}")
        print(f"  d'_theory = {dprime_theory:.3f}, ratio_theory = {ratio_theory:.3f}")

        # Apply each rule
        for rule_name in rules:
            rule_func = get_rule_function(rule_name)
            result = rule_func(L_A, L_B, labels, rng_cf)

            dprime_team = result['dprime_team']
            ratio = dprime_team / d_best if d_best > 0 else 0

            print(f"    {rule_name}: d'_team = {dprime_team:.3f}, ratio = {ratio:.3f}")

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
                'ratio_theory': ratio_theory
            })

    df = pd.DataFrame(results)

    # Save results
    output_csv = '../outputs/bahrami_sweep_final.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")

    # Create plots
    create_bahrami_plot(df)
    create_rich_verification_plot(df)

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


def create_rich_verification_plot(df, output_path='../outputs/rich_theory_verification.png'):
    """
    Create Rich's verification plot: DSS (simulated) vs Theoretical prediction.

    X-axis: c_B (novice ability)
    Y-axis: Collective Benefit Ratio
    Curves: DSS (simulated) and Theoretical (orthogonal sum)
    """
    plt.figure(figsize=(10, 7))

    # Extract DSS data
    dss_data = df[df['rule'] == 'DSS'].sort_values('c_B')

    # Plot DSS (simulated)
    plt.plot(
        dss_data['c_B'],
        dss_data['collective_benefit_ratio'],
        marker='o',
        color='#1f77b4',
        label='DSS (Simulated)',
        linewidth=2.5,
        markersize=8,
        alpha=0.9
    )

    # Plot theoretical prediction
    plt.plot(
        dss_data['c_B'],
        dss_data['ratio_theory'],
        linestyle='--',
        color='black',
        label='Theoretical (Orthogonal Sum)',
        linewidth=2.5,
        alpha=0.9
    )

    plt.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.xlabel('Agent B Ability (c_B)', fontsize=14, fontweight='bold')
    plt.ylabel("Collective Benefit Ratio\n(d'_team / d'_best)", fontsize=14, fontweight='bold')
    plt.title("Rich's Verification: REM Simulation vs. SDT Prediction\n(d'_theory = √(d'_A² + d'_B²))",
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='best', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim(0.05, 0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")


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
    - WCS_Miscal (Prelec-based)
    - DMC_Miscal (Prelec-based)
    - DSS (alpha-independent benchmark)
    - CF (alpha-independent benchmark)
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
    n_test = 5000  # Increased to reduce sampling noise
    seed_master = 1000

    results = []

    print("\n" + "="*70)
    print("CONFIDENCE MISCALIBRATION SWEEP (PRELEC WEIGHTING)")
    print("="*70)
    print(f"Equal Ability: c_A = c_B = {c_A}")
    print(f"Agent A: α = {alpha_A:.1f} (Overconfident)")
    print(f"Agent B: α ∈ [{alpha_B_values[0]:.1f}, {alpha_B_values[-1]:.1f}] (step = 0.1)")
    print(f"Trials per condition: {n_test}")
    print("="*70)

    for idx, alpha_B in enumerate(alpha_B_values):
        print(f"\n[α_B = {alpha_B:.1f}]")

        # Re-initialize RNGs for each condition
        condition_seed = seed_master + idx
        master_rng = np.random.default_rng(condition_seed)

        seed_A = int(master_rng.integers(0, 2**31 - 1))
        seed_B = int(master_rng.integers(0, 2**31 - 1))
        seed_test = int(master_rng.integers(0, 2**31 - 1))
        seed_cf = int(master_rng.integers(0, 2**31 - 1))

        rng_A = np.random.default_rng(seed_A)
        rng_B = np.random.default_rng(seed_B)
        rng_test = np.random.default_rng(seed_test)
        rng_cf = np.random.default_rng(seed_cf)

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

        # Compute individual d'
        D_A = (L_A > 0).astype(int)
        D_B = (L_B > 0).astype(int)
        dprime_A = group_rules.compute_dprime_from_decisions(D_A, labels)
        dprime_B = group_rules.compute_dprime_from_decisions(D_B, labels)
        d_best = max(dprime_A, dprime_B)

        print(f"  d'_A = {dprime_A:.3f}, d'_B = {dprime_B:.3f}, d'_best = {d_best:.3f}")

        # Run WCS_Miscal
        wcs_result = group_rules.wcs_miscal_rule(L_A, L_B, labels, rng_cf, alpha_A, alpha_B)
        wcs_ratio = wcs_result['dprime_team'] / d_best if d_best > 0 else 0
        print(f"    WCS: d'_team = {wcs_result['dprime_team']:.3f}, ratio = {wcs_ratio:.3f}")

        results.append({
            'c_A': c_A,
            'c_B': c_B,
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'model': 'WCS_Miscal',
            'dprime_A': dprime_A,
            'dprime_B': dprime_B,
            'dprime_team': wcs_result['dprime_team'],
            'd_best': d_best,
            'collective_benefit_ratio': wcs_ratio
        })

        # Run DMC_Miscal
        dmc_result = group_rules.dmc_miscal_rule(L_A, L_B, labels, rng_cf, alpha_A, alpha_B)
        dmc_ratio = dmc_result['dprime_team'] / d_best if d_best > 0 else 0
        print(f"    DMC: d'_team = {dmc_result['dprime_team']:.3f}, ratio = {dmc_ratio:.3f}")

        results.append({
            'c_A': c_A,
            'c_B': c_B,
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'model': 'DMC_Miscal',
            'dprime_A': dprime_A,
            'dprime_B': dprime_B,
            'dprime_team': dmc_result['dprime_team'],
            'd_best': d_best,
            'collective_benefit_ratio': dmc_ratio
        })

        # Run DSS (benchmark, alpha-independent)
        dss_result = group_rules.direct_signal_sharing(L_A, L_B, labels, rng_cf)
        dss_ratio = dss_result['dprime_team'] / d_best if d_best > 0 else 0
        print(f"    DSS: d'_team = {dss_result['dprime_team']:.3f}, ratio = {dss_ratio:.3f}")

        results.append({
            'c_A': c_A,
            'c_B': c_B,
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'model': 'DSS',
            'dprime_A': dprime_A,
            'dprime_B': dprime_B,
            'dprime_team': dss_result['dprime_team'],
            'd_best': d_best,
            'collective_benefit_ratio': dss_ratio
        })

        # Run CF (benchmark, alpha-independent)
        cf_result = group_rules.coin_flip_rule(L_A, L_B, labels, rng_cf)
        cf_ratio = cf_result['dprime_team'] / d_best if d_best > 0 else 0
        print(f"    CF: d'_team = {cf_result['dprime_team']:.3f}, ratio = {cf_ratio:.3f}")

        results.append({
            'c_A': c_A,
            'c_B': c_B,
            'alpha_A': alpha_A,
            'alpha_B': alpha_B,
            'model': 'CF',
            'dprime_A': dprime_A,
            'dprime_B': dprime_B,
            'dprime_team': cf_result['dprime_team'],
            'd_best': d_best,
            'collective_benefit_ratio': cf_ratio
        })

    df_miscal = pd.DataFrame(results)

    # Save results
    output_csv = '../outputs/miscalibration_sweep.csv'
    df_miscal.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")

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


def create_miscalibration_plot(df, output_path='../outputs/miscalibration_plot.png'):
    """
    Create miscalibration sweep plot.

    X-axis: alpha_B (Agent B miscalibration)
    Y-axis: Collective Benefit Ratio
    Curves: WCS_Miscal, DMC_Miscal, DSS, CF
    Reference: Vertical line at alpha = 1.0 (calibrated)
    """
    plt.figure(figsize=(10, 7))

    models = ['DSS', 'WCS_Miscal', 'DMC_Miscal', 'CF']
    colors = {
        'DSS': '#1f77b4',
        'WCS_Miscal': '#ff7f0e',
        'DMC_Miscal': '#2ca02c',
        'CF': '#9467bd'
    }
    markers = {
        'DSS': 'o',
        'WCS_Miscal': 's',
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

def run_rich_conflict_simulation():
    """
    Run Rich Shiffrin's Conflict Resolution Model simulation.
    
    Goal: Verify that P(choose stronger | conflict) follows (1+D)/(2+D)
    
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
    rich_result = group_rules.rich_conflict_rule(L_A, L_B, labels, rng_cf)
    
    print("Running CF baseline...")
    cf_result = group_rules.coin_flip_rule(L_A, L_B, labels, rng_cf)
    
    # Analyze conflict trials
    print("\n" + "="*70)
    print("CONFLICT TRIAL ANALYSIS")
    print("="*70)
    
    # Convert to odds and strength
    odds_A = np.exp(L_A)
    odds_B = np.exp(L_B)
    power = 1.0 / 11.0
    odds_scaled_A = np.power(odds_A, power)
    odds_scaled_B = np.power(odds_B, power)
    S_A = np.maximum(odds_scaled_A, 1.0 / odds_scaled_A)
    S_B = np.maximum(odds_scaled_B, 1.0 / odds_scaled_B)
    
    # Identify conflicts
    conflict_mask = ((L_A > 0) & (L_B < 0)) | ((L_A < 0) & (L_B > 0))
    n_conflicts = np.sum(conflict_mask)
    
    print(f"Total trials: {n_test}")
    print(f"Conflict trials: {n_conflicts} ({100*n_conflicts/n_test:.1f}%)")
    
    # Compute D for conflict trials
    D_diff = np.abs(S_A - S_B)
    
    # Individual decisions
    D_A = (L_A > 0).astype(int)
    D_B = (L_B > 0).astype(int)
    
    # Re-run Rich rule to track choices
    D_team_rich = np.zeros_like(D_A)
    D_team_rich[~conflict_mask] = D_A[~conflict_mask]
    
    chose_stronger = []
    D_values = []
    
    if n_conflicts > 0:
        P_choose_stronger = (1.0 + D_diff) / (2.0 + D_diff)
        A_stronger = S_A > S_B
        B_stronger = S_B > S_A
        
        conflict_indices = np.where(conflict_mask)[0]
        for idx in conflict_indices:
            p = P_choose_stronger[idx]
            
            if A_stronger[idx]:
                choose_A = rng_cf.random() < p
                D_team_rich[idx] = D_A[idx] if choose_A else D_B[idx]
                chose_stronger.append(1 if choose_A else 0)
            elif B_stronger[idx]:
                choose_B = rng_cf.random() < p
                D_team_rich[idx] = D_B[idx] if choose_B else D_A[idx]
                chose_stronger.append(1 if choose_B else 0)
            else:
                D_team_rich[idx] = D_A[idx] if rng_cf.random() < 0.5 else D_B[idx]
                chose_stronger.append(0.5)
            
            D_values.append(D_diff[idx])
    
    # Create binned analysis
    D_bins = np.arange(0, max(D_values) + 0.5, 0.5)
    bin_centers = []
    empirical_probs = []
    theoretical_probs = []
    
    for i in range(len(D_bins) - 1):
        bin_mask = (np.array(D_values) >= D_bins[i]) & (np.array(D_values) < D_bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_center = (D_bins[i] + D_bins[i+1]) / 2
            empirical_prob = np.mean(np.array(chose_stronger)[bin_mask])
            theoretical_prob = (1 + bin_center) / (2 + bin_center)
            
            bin_centers.append(bin_center)
            empirical_probs.append(empirical_prob)
            theoretical_probs.append(theoretical_prob)
    
    # Save results
    results_df = pd.DataFrame({
        'D_bin_center': bin_centers,
        'empirical_P': empirical_probs,
        'theoretical_P': theoretical_probs,
        'n_trials_in_bin': [np.sum((np.array(D_values) >= D_bins[i]) & (np.array(D_values) < D_bins[i+1])) 
                            for i in range(len(bin_centers))]
    })
    
    output_csv = '../outputs/rich_conflict_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")
    
    # Create plot
    create_rich_conflict_plot(results_df)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Rich_Conflict d': {rich_result['dprime_team']:.3f}")
    print(f"CF baseline d': {cf_result['dprime_team']:.3f}")
    print(f"\nMean empirical P(choose stronger): {np.mean(empirical_probs):.3f}")
    print(f"Mean theoretical P(choose stronger): {np.mean(theoretical_probs):.3f}")
    
    print("\n" + "="*70)
    print("✓ RICH'S CONFLICT SIMULATION COMPLETE")
    print("="*70)
    
    return results_df


def create_rich_conflict_plot(df, output_path='../outputs/rich_conflict_plot.png'):
    """
    Plot P(choose stronger | conflict) vs D.
    
    X-axis: Binned difference D
    Y-axis: Probability of choosing stronger agent
    """
    plt.figure(figsize=(10, 7))
    
    # Empirical data
    plt.scatter(df['D_bin_center'], df['empirical_P'], 
                s=100, alpha=0.7, color='#2ca02c', 
                label='Empirical (Rich Model)', zorder=3)
    
    # Theoretical curve
    D_smooth = np.linspace(0, max(df['D_bin_center']) + 0.5, 100)
    P_theoretical = (1 + D_smooth) / (2 + D_smooth)
    plt.plot(D_smooth, P_theoretical, 
             linewidth=2.5, color='black', linestyle='--',
             label='Theoretical: P = (1+D)/(2+D)', zorder=2)
    
    # Reference line
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Strength Difference (D = |S_A - S_B|)', fontsize=14, fontweight='bold')
    plt.ylabel('P(Choose Stronger Agent | Conflict)', fontsize=14, fontweight='bold')
    plt.title("Rich's Conflict Resolution Model\nProbability of Choosing Stronger Agent", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(loc='lower right', fontsize=12, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim(-0.1, max(df['D_bin_center']) + 0.5)
    plt.ylim(0.45, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")


# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_bahrami_sweep()
    run_miscalibration_sweep()
