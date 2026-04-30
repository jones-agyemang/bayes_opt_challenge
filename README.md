# Project Overview
This project tackles a black-box optimisation challenge with eight unknown objective functions. Each function accepts continuous inputs in `[0, 1]^d` and returns a scalar score. The goal is to propose one strong next point per function under a tight evaluation budget.

The primary strategy is now a HEBO-style `q = 1` optimiser inspired by the Huawei Noah's Ark NeurIPS 2020 winning solution. The baseline GP proposer remains in the codebase for comparison, but HEBO is the default.

# Current Strategy
The main proposal pipeline combines four ideas that were missing from the earlier single-acquisition GP setup:

1. Output power transforms to reduce skew and heteroscedasticity.
2. Kumaraswamy input warping to handle non-stationary behaviour over `[0, 1]^d`.
3. An additive `Linear + Matern32` Gaussian process surrogate.
4. A MACE proposer that searches a Pareto front over `logEI`, `PI`, and `UCB` with NSGA-II, then picks one balanced candidate.

This matters because the challenge data are not well behaved. The logged outputs already vary from near-zero scales to values above `1000`, and several functions appear noisy or non-stationary. A vanilla GP with a single acquisition can become overconfident and collapse onto brittle local optima.

# Proposal Modes
The configuration surface is split into two paths:

- `proposal.mode = "hebo"`: default path using warped GP + NSGA-II MACE.
- `proposal.mode = "baseline"`: previous single-acquisition GP path using either gradient search or random candidate sampling.

The default configuration is defined in `utils/cycle_parameters.py` and is shared across all eight functions. Per-function overrides are still possible if you want ablations or comparisons.

# Running
Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate one proposal per function for a cycle:

```bash
python3 bayesian_optimiser.py --cycle 7
python3 -u bayesian_optimiser.py --cycle 7 --population-size 12 --generations 4 --warp-multistarts 1 --warp-maxiter 5 --gp-restarts 0
```

Run the replay sanity comparison on a shared candidate pool:

```bash
python3 replay_sanity.py --cycle 7 --pool-size 5000
```

Run tests:

```bash
pytest -q
```

# Research Basis
- Rasmussen and Williams (2006): GP regression and uncertainty-aware surrogate modelling.
- Huawei Noah's Ark Lab, NeurIPS 2020 BBO winner: warped GP surrogate, stochastic mean, and multi-objective acquisition search.

The Huawei paper directly motivated the current design choices:
- power transformation for heteroscedastic outputs,
- input warping for non-stationary functions,
- `Linear + Matern32` kernel for mixed global/local structure,
- MACE over multiple acquisitions instead of committing to one acquisition globally.
