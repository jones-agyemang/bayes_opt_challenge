# Bayesian Optimisation Review Checklist

Use this checklist when a review needs deeper scrutiny than the core workflow in `SKILL.md`.

## Objective Direction and Sign Conventions

- Confirm whether the project optimises for maximisation or minimisation.
- Verify one consistent convention across:
  - observed target values
  - GP training target
  - acquisition score definition
  - acquisition optimiser objective (`argmax` vs `minimize`)
  - selected candidate update rule
- Reject implicit sign flips without explicit comments and tests.
- Treat LCB in maximisation as suspicious unless transformed correctly.

## Surrogate Model Integrity

- Check kernel family choice against expected function behaviour.
  - RBF: smooth functions.
  - Matern: rougher functions and tunable smoothness.
  - RationalQuadratic: multiple smoothness scales.
- Verify hyperparameter bounds and optimiser restarts are reasonable.
- Confirm noise modelling is explicit (`alpha` or noise kernel).
- Verify preprocessing consistency (feature scaling and any target transforms).
- Ensure numerical guardrails exist for ill-conditioned covariance matrices.

## Acquisition Function Correctness

- Verify formula and direction for the chosen acquisition.
  - EI: improvement should be in the objective's preferred direction.
  - UCB/LCB: confidence bound sign should match objective direction.
  - PI: improvement threshold should be aligned to maximise/minimise mode.
- Check edge handling when `sigma -> 0` to avoid invalid division and NaNs.
- Verify exploration parameters:
  - `kappa` for UCB/LCB
  - `xi` for EI/PI
- Require justification when exploration parameters are changed.

## Acquisition Optimisation Procedure

- Confirm candidate search respects parameter bounds.
- Check for adequate search diversity:
  - random candidate sampling
  - multi-start local optimisation
- Validate optimizer method compatibility (for example, `L-BFGS-B` with bounds).
- Flag static candidate count in higher-dimensional settings.

## Numerical Stability

- Check for jitter/regularisation in GP fitting.
- Guard against negative predictive variances from floating-point noise.
- Confirm stable Cholesky or fallback behaviour where needed.
- Verify finite checks around acquisition outputs before `argmax`/`argmin`.

## Compute and Scaling

- Note that exact GP fitting scales roughly `O(n^3)`.
- Check whether loop design controls dataset growth or downsampling.
- Flag expensive configurations with no runtime safeguards.

## Minimum Evidence in PR Review

Request all of the following when behaviour changes:

- unit tests for sign convention and objective direction
- reproducible seed or deterministic test mode
- before/after objective traces across BO iterations
- rationale for kernel and acquisition parameter updates

## Useful References

- SciPy `minimize`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- SciPy L-BFGS-B: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
- scikit-learn Gaussian Processes: https://scikit-learn.org/stable/modules/gaussian_process.html
- scikit-learn Matern kernel: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
- GPML textbook (Rasmussen and Williams): http://www.gaussianprocess.org/gpml/
