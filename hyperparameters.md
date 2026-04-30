# BO Hyperparameters

The optimiser now has two public control surfaces: `surrogate` for the warped GP model and `proposal` for the HEBO-style MACE search. Baseline GP settings are still available under `baseline` for comparison.

| Hyperparameter | Default value | Typical values | Priority (1-5) | Why it matters |
|---|---:|---|---:|---|
| `proposal.mode` | `hebo` | `hebo`, `baseline` | 5 | Selects the entire proposal pipeline. |
| `surrogate.output_transform` | `auto` | `auto` | 5 | Chooses Box-Cox or Yeo-Johnson to stabilise variance and reduce skew. |
| `surrogate.input_warping` | `True` | `True`, `False` | 5 | Handles non-stationarity over `[0, 1]^d`; disabling it usually weakens the model. |
| `surrogate.kernel` | `linear_plus_matern32` | `linear`, `rbf`, `matern32`, `matern52`, `linear_plus_matern32`, `linear_plus_matern52` | 3 | Selects the surrogate inductive bias; additive linear-plus-Matern remains the default. |
| `surrogate.stochastic_mean_xi` | `1.0` | `0.0-2.0` | 4 | Adds noise-aware exploration pressure to the posterior mean. |
| `proposal.objectives` | `["log_ei", "pi", "ucb"]` | Subsets or reordered variants | 5 | Defines the acquisition ensemble searched by MACE. |
| `proposal.population_size` | `96` | `48-256` | 4 | Controls NSGA-II search breadth. |
| `proposal.generations` | `60` | `20-120` | 4 | Controls optimisation depth of the Pareto search. |
| `proposal.kappa` | `2.0` | `1-5` | 3 | Sets exploration strength in UCB. |
| `proposal.xi` | `0.0` | `0-0.1` | 3 | Sets the improvement threshold for `PI` and `logEI`. |
| `proposal.min_distance` | `1e-3` | `1e-4-1e-2` | 4 | Prevents the proposer from collapsing back onto previously observed points. |
| `proposal.dedupe_tol` | `1e-6` | `1e-7-1e-5` | 3 | Defines when a candidate counts as a duplicate. |
| `baseline.acquisition.strategy` | `ucb` | `ucb`, `ei` | 2 | Only relevant when `proposal.mode="baseline"`. |
| `baseline.proposer` | `propose_next` | `propose_next`, `propose_next_rnd_sampling` | 2 | Baseline-only search style. |

## Tuning Order
1. `surrogate.input_warping`
2. `surrogate.output_transform`
3. `proposal.population_size` and `proposal.generations`
4. `surrogate.stochastic_mean_xi`
5. `proposal.kappa` and `proposal.xi`

The highest-value knobs are the ones that control how well the surrogate handles heteroscedasticity and non-stationarity. In the Huawei ablations, input warping and power transformation were the most significant contributors, so those should be treated as first-order design choices rather than cosmetic tuning.
