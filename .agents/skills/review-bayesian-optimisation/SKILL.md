---
name: review-bayesian-optimisation
description: Review Bayesian optimisation code and design changes for mathematical correctness and objective consistency. Use when evaluating pull requests, diffs, or proposals involving Gaussian-process surrogates, acquisition functions (EI/UCB/PI/LCB), sign conventions, exploration-exploitation parameters, numerical stability, and acquisition optimisation for maximisation or minimisation tasks.
---

# Review Bayesian Optimisation

Run a math-first review focused on preserving the optimisation objective and statistical validity of the BO loop.

## Review Workflow

1. Establish the optimisation contract
- Confirm the true objective direction (maximise or minimise).
- Trace sign conventions end-to-end: objective values, surrogate target, acquisition value, optimiser call (`argmax` vs `minimize`), and selected candidate.
- Flag any silent objective flips or mixed conventions.

2. Audit the Gaussian-process surrogate
- Check whether kernel assumptions match expected smoothness and noise.
- Verify hyperparameter fitting strategy (bounds, restarts, and sane defaults).
- Validate noise modelling (`alpha` or explicit noise kernel) and feature/target scaling.
- Require stability guards for near-singular covariance matrices.

3. Validate acquisition math
- Verify EI/UCB/PI/LCB formula direction matches the objective.
- If LCB appears in a maximisation setup, require explicit mathematical equivalence (for example, optimise negative LCB with minimisation).
- Check behaviour when predictive standard deviation is near zero.
- Inspect exploration parameters (`kappa`, `xi`) for justification against iteration budget.

4. Inspect acquisition optimisation
- Confirm bounds are respected and candidate search is sufficiently diverse (random sampling and/or multi-start local search).
- Check optimiser choices and stopping conditions for brittleness.
- Flag fixed candidate budgets that do not scale with dimensionality.

5. Assess numerical and computational risk
- Verify jitter/regularisation and finite-value guards around GP and acquisition calculations.
- Check for negative variances or unstable decompositions.
- Note computational growth (`O(n^3)` GP fitting) and whether data growth controls are present.

6. Produce findings with concrete fixes
- Prioritise issues by severity.
- Connect each issue to mathematical impact on optimisation behaviour.
- Propose the smallest correct code change and a validation test.

## Output Format

For each finding, include:
- `Severity`: critical, high, medium, or low
- `Location`: file path and line number
- `Issue`: the mathematical or algorithmic defect
- `Impact`: expected behavioural regression
- `Fix`: minimal corrective change
- `Validation`: experiment or test that confirms the fix

## Reference

Load [references/review_checklist.md](references/review_checklist.md) for deeper checks and canonical links.
