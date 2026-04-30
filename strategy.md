# Cycle 5

Adjustments Log
- 
  - Currently using a gradient-based minimizer with 10 multi-starts. Switching to randomly sampling sampling a larger space and submitting the best candidate after evaluating all `n` candidate. This change ensures that the GP generates `n` number of possible candidates(`X_next`) in the given search space $[0, 1]^d$. The acquisition point is evaluated at the generated candidate points, with the highest acquisition value being chosen as the final value
  This is computationally intensive but, it should help to reach our maximisation goal faster. 

## Function 1
<img src="./evaluations/signed_log_outputs_1.png" alt="Function1-perf_graph" width="50%" height="50%">

## Function 2
<img src="./evaluations/signed_log_outputs_2.png" alt="Function2-perf_graph" width="50%" height="50%">

## Function 3
<img src="./evaluations/signed_log_outputs_3.png" alt="Function3-perf_graph" width="50%" height="50%">

## Function 4
<img src="./evaluations/signed_log_outputs_4.png" alt="Function4-perf_graph" width="50%" height="50%">

## Function 5
<img src="./evaluations/signed_log_outputs_5.png" alt="Function5-perf_graph" width="50%" height="50%">

## Function 6
<img src="./evaluations/signed_log_outputs_6.png" alt="Function6-perf_graph" width="50%" height="50%">

## Function 7
<img src="./evaluations/signed_log_outputs_7.png" alt="Function7-perf_graph" width="50%" height="50%">

## Function 8
<img src="./evaluations/signed_log_outputs_8.png" alt="Function8-perf_graph" width="50%" height="50%">

# Cycle 6
Adjustment Log
- introduced function-specific acquisition hyperparameter tuning. 
- dedicating this cycle to addressing the lack of use of acquisition hyperparameter usage to obtain proposed input values. I'm addressing this change as prescribed in a previous sentence. This is problematic; it means we have been using values with limited exploration search space. Hopefully this cycle should address that and give us a much better understanding of which regions are profitable and which regions are not worth exploring. After that, we can dive deeper into exploitation. 
- Improve GP noise/normalisation robustness
  - Issue: GP uses fixed tiny noise `alpha=1e-6` with no target normalisation. 
  - Impact: Outputs have shown to span extreme scales. Surrogate model can become overconfident/ill-conditioned, distorting EI/UCB and exploration-exploitation balance.
  - Proposed fix: Enable output normalisation, tune/increase `alpha` (or use a noise kernel) with bounded hyperparameters.
  - Validation: verify best found traces before/after for all functions


# Cycle 7
Using the NeurisIP 2020 winning team's idea. Implemented a HEBO-style setup. Running with default values for all functions. 
Hyperparameters:
- acquisition strategy: UCB
- kappa: 2

# Cycle 8
Refactored the hardcoded HEBO kernel. Before this fix, the kernel configuration surface had no effect on the surrogate, so the intended per-function tuning was effectively ignored. Cycle 8 is the first pass where those HEBO configuration choices genuinely change the fitted surrogate and therefore the proposed point.

## Cycle 8 Rationale
* Crafted with AI support(ChatGPT-5.4; Reasoning Level: High)
- Cycle 7 used one shared HEBO default across all functions. That was a reasonable grounding step, but it left too much performance on the table because each black-box function may have a different level of smoothness, non-stationarity, and local ruggedness.
- Cycle 8 therefore switches from one default HEBO setup to a diversified HEBO portfolio. The goal is not to claim exact knowledge of each objective, but to spread risk across multiple plausible surrogate assumptions now that `surrogate.kernel` is active.
- The highest-confidence HEBO choices remain unchanged across all functions:
  - `surrogate.output_transform = "auto"` to stabilise skewed or heteroscedastic targets
  - `surrogate.input_warping = True` to better model non-stationary structure over `[0, 1]^d`
  - `proposal.mode = "hebo"` with the standard MACE objective set `["log_ei", "pi", "ucb"]`
- The main cycle-8 tuning levers are:
  - `surrogate.kernel` to change inductive bias
  - `proposal.population_size` and `proposal.generations` to change the quality of Pareto acquisition search
  - `surrogate.stochastic_mean_xi`, `proposal.kappa`, and `proposal.xi` to control exploration pressure
  - `warp_multistarts`, `warp_maxiter`, and `gp_restarts` to improve surrogate fit robustness on harder functions

## Portfolio Logic
- Functions 1, 5, and 8 use smoother kernels such as `linear_plus_matern52` with medium-to-longer length scales. These were assigned more balanced or mildly exploitative settings because they represent the case where the response surface may contain broad structure that should not be over-fragmented by an overly rough kernel.
- Functions 2 and 6 use `matern32` with shorter length scales and higher exploration pressure. These are the rough-surface hypotheses in the portfolio, where the risk of missing narrow high-value regions is larger.
- Function 3 combines `linear_plus_matern32` with short length scale, high restart counts, and a deeper acquisition search. It is treated as one of the hardest cases in the portfolio and therefore receives one of the most exploratory and compute-heavy configurations.
- Function 4 is the most exploitative configuration. It uses `matern52`, a longer length scale, and lower `kappa` and `stochastic_mean_xi`. This represents the smooth-surface hypothesis where aggressive exploration is less likely to be efficient.
- Function 7 is the widest-search configuration. It receives the highest population size, strong exploration pressure, and the largest minimum distance from previous observations. This slot is intended to guard against premature local collapse when the search landscape is especially deceptive.

## Why This Is Defensible
- The cycle-8 settings are heuristic, but they are not arbitrary. They are designed around the highest-value HEBO controls already documented in `hyperparameters.md`, plus the fact that cycle 8 is the first cycle where kernel selection is actually operational.
- This is a portfolio strategy rather than a claim of perfect per-function identification. Under a tight evaluation budget, diversified inductive bias is safer than assuming the same surrogate shape is optimal for all eight functions.
- The next validation step is empirical rather than conceptual: compare cycle-8 replay performance and proposed maxima against the cycle-7 uniform-default setup and prune the weaker configurations in later cycles.


# Cycle 9
## Function 1
Still haven't found a solution that exceeds the maximum of initial data set. Expanding the value of kappa for the HEBO process to explore nearby promising region.

## Function 2
Increasing the value of kappa from 3.0 to 5.0 to expand the search space.

## Function 3
Tweaking EI from 0.01 to 0.05. Expecting there to be better improvements around this nearly discovered peak

### Function 4
Kappa from 1.5 to 2.5

## Function 5
Kappa from 1.75 to 2.0
xi from 0.1 to 0.5

## Function 6
kappa from 2.5 to 3.0
xi from 0.01 to 0.09

## Function 7
population_size 192 -> 300
generations 90 -> 180
kappa 3.0 -> 5.0
xi 0.02 -> 0.7

# Function 8
population_size 128 -> 256
kappa 2.0 -> 2.5
xi 0.01 -> 0.03