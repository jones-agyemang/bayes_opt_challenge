# Week 5

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

# Week 6
Adjustment Log
- introduced function-specific acquisition hyperparameter tuning. 
- dedicating this cycle to addressing the lack of use of acquisition hyperparameter usage to obtain proposed input values. I'm addressing this change as prescribed in a previous sentence. This is problematic; it means we have been using values with limited exploration search space. Hopefully this cycle should address that and give us a much better understanding of which regions are profitable and which regions are not worth exploring. After that, we can dive deeper into exploitation. 
- Improve GP noise/normalisation robustness
  - Issue: GP uses fixed tiny noise `alpha=1e-6` with no target normalisation. 
  - Impact: Outputs have shown to span extreme scales. Surrogate model can become overconfident/ill-conditioned, distorting EI/UCB and exploration-exploitation balance.
  - Proposed fix: Enable output normalisation, tune/increase `alpha` (or use a noise kernel) with bounded hyperparameters.
  - Validation: verify best found traces before/after for all functions
- 