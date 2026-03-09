# BO hyperparameters

Here the hyperparameters in this codebase, with typical values and tuning priority.

| Hyperparameter | Current value in project | Typical values | Tune priority (1-5) | Why tune it |
|---|---:|---|---:|---|
| `acquisition.strategy` | Cycle-dependent: mostly `ucb`; cycle 5 uses `ei` | `ucb`, `ei` (sometimes `pi`) | 4 | Controls exploration style; wrong strategy can stall progress on multimodal functions. |
| `params.kappa` (UCB) | `2.0` (base cycles), `25` (cycle 6) | `1-5` common; `>10` very exploratory | 5 | Directly sets exploration strength in UCB. `25` is aggressive and can over-explore. |
| `params.xi` (EI) | `25` (cycle 5) | `1e-4` to `0.1` (occasionally up to `1`) | 5 | In EI, large `xi` heavily penalizes improvement threshold; `25` is usually too large unless targets are similarly scaled. |
| `RBF.length_scale` (initial) | `0.3` | `0.05-1.0` after input normalization | 4 | Governs smoothness assumption; bad length-scale hurts surrogate fit and acquisition quality. |
| `RBF.length_scale_bounds` | `(1e-3, 1e3)` | Problem-dependent, often narrower (e.g. `1e-2` to `10`) | 3 | Very wide bounds can slow GP optimization or allow pathological fits. |
| `ConstantKernel` amplitude init/bounds | `1.0`, `(1e-3, 1e3)` | Init near output variance; broad bounds are common | 3 | Sets prior signal scale; affects uncertainty calibration and acquisition behavior. |
| `alpha` noise model | `max(1e-6, 1e-4 * var(y))` | `1e-8` to `1e-2` or data-driven scaling | 5 | Noise level strongly impacts GP confidence; underestimating noise makes overconfident bad suggestions. |
| `normalize_y` | `True` | Usually `True` | 3 | Important for stability when output scales vary across functions. |
| `n_restarts_optimizer` (GP) | `10` | `5-20` | 2 | Improves kernel fit robustness but increases runtime. |
| Acquisition optimizer multistarts (`propose_next`) | `10` starts | `5-30` | 3 | More starts reduce local-minima risk in acquisition optimization. |
| Random candidate pool (`propose_next_rnd_sampling`) | `50,000` | `5,000-200,000` (depends on dim/runtime) | 3 | Higher gives better approximation of acquisition optimum but costs more compute. |
| Domain bounds | `[1e-6, 0.999999]^d` | `[0,1]^d` or transformed bounds | 2 | Mostly fixed by problem; tune only if boundary behavior matters. |
| RNG seed for random sampling | `seed = cycle*NUMBER_OF_FUNCTIONS + func_id` | Fixed per run for reproducibility | 2 | Helps reproducibility vs diversity tradeoff across cycles/functions. |


## Tuning Priority Order
1. `kappa` / `xi` (acquisition aggressiveness)
2. GP noise `alpha` scaling
3. GP kernel length-scale behavior (init + bounds)

Tune priority (1-5) is a ranking of expected impact vs effort/risk; with 1 being less important and 5 being super-important.
|Priority order|Importance|Description|
|-:|-|-|
|5 |**Critical**| High leverage on BO performance; likely to change results materially.|
|4 |**High**| Important, but slightly less universal/urgent than 5.|
|3 |**Medium**| Moderate gains or more context-dependent.|
|2 |**Low**| Mostly refinement, robustness, or runtime tradeoff.|
|1 |**Ignore**| Minimal expected benefit for your current setup.|