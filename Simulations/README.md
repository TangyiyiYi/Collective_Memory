# REM Group Decision Simulations

This directory contains simulations for collective memory research using the REM (Retrieving Effectively from Memory) model.

## Quick Start

### Run Python Script
```bash
# Navigate to src directory
cd src

# Run the simulation
python run_simulation.py
```

### Run Jupyter Notebook (Recommended)
```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook bahrami_sweep_demo.ipynb
```

## Directory Structure

```
Simulations/
├── src/                    # Core simulation code
│   ├── rem_core.py         # REM engine (READ-ONLY)
│   ├── group_rules.py      # 7 decision rules (CF, UW, DMC, DSS, BF, WCS_Miscal, DMC_Miscal)
│   └── run_simulation.py   # Main simulation runner
│
├── notebooks/              # Interactive analysis
│   ├── bahrami_sweep_demo.ipynb
│   └── exports/            # PDF/HTML exports
│
├── outputs/               # Generated results (gitignored)
│   ├── bahrami_sweep_final.csv
│   ├── bahrami_sweep_plot.png
│   ├── rich_theory_verification.png
│   ├── miscalibration_sweep.csv
│   └── miscalibration_plot.png
│
└── archive/               # Legacy code and results
    ├── legacy_code/
    ├── legacy_results/
    └── legacy_docs/
```

## What This Simulates

Three independent parameter sweeps:

1. **Bahrami Sweep**: How does ability heterogeneity affect group performance?
   - Fixed expert (c_A = 0.7)
   - Sweep novice ability (c_B from 0.1 to 0.9)
   - Test 5 decision rules: CF, UW, DMC, DSS, BF

2. **Rich's Verification**: Does DSS match theoretical SDT prediction?
   - Compare simulated DSS to orthogonal sum: d'_theory = √(d'_A² + d'_B²)

3. **Confidence Miscalibration**: How does Prelec weighting affect integration?
   - Equal ability (c_A = c_B = 0.7)
   - Fixed Agent A overconfident (α_A = 1.2)
   - Sweep Agent B miscalibration (α_B from 0.5 to 1.5)
   - Test 4 models: WCS_Miscal, DMC_Miscal, DSS, CF

## Key Metric

**Collective Benefit Ratio (CBR)** = d'_team / max(d'_A, d'_B)

- CBR > 1: Group outperforms best individual
- CBR = 1: Group equals best individual
- CBR < 1: Group underperforms

## Outputs

All outputs are generated in `outputs/` directory and are gitignored (regenerate by running the simulation).

## Notes

- `rem_core.py` is READ-ONLY (REM engine, do not modify)
- Results are reproducible (fixed random seeds)
- Archive contains legacy code for reference only
