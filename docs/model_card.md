# Model Card: HEBO-Style Bayesian Optimiser for the BBO Challenge

## Overview

- **Approach name:** HEBO-style Bayesian optimiser for black-box optimisation.
- **Model type:** Sequential Bayesian optimisation pipeline using Gaussian-process surrogate modelling and acquisition-based candidate proposal.
- **Version:** Current repository version, documented through cycle 9 with logged submitted outputs in `processed_data/`.
- **Primary objective:** Maximise eight hidden scalar objective functions over bounded continuous domains `[0, 1]^d`.

This is not a predictive model intended for deployment on new labelled datasets. It is an optimisation strategy that proposes candidate inputs under a tight query budget.

## Intended Use

Suitable uses:

- Proposing one next query point per hidden black-box function.
- Comparing Bayesian optimisation strategies under limited observations.
- Studying surrogate modelling, acquisition functions, input warping, and output transformation in small-data optimisation.
- Replaying candidate choices against recorded challenge outputs for sanity checking.

Uses to avoid:

- Treating the optimiser as a general-purpose supervised-learning model.
- Using the observed challenge data as evidence about real-world systems.
- Applying the method without modification to safety-critical, expensive, or irreversible real-world experiments.
- Assuming the same hyperparameters will transfer unchanged to objectives with different noise, constraints, dimensionality, or cost structures.

## Details

The strategy evolved across the challenge from a simpler Gaussian-process optimiser into a more robust HEBO-style pipeline.

### Round-by-Round Strategy

The repository contains exact output logs for submitted rounds and detailed written strategy notes from later cycles. Earlier rounds are represented by logged inputs and outputs, but their full design rationale is less completely documented.

1. **Initial baseline phase:** Started from the challenge-provided seed observations for eight hidden functions.
2. **Early GP proposals:** Used Gaussian-process surrogate modelling to estimate promising regions under limited data.
3. **Single-acquisition search:** Relied on standard acquisition-driven proposal logic, mainly exploitation/exploration trade-offs such as UCB-style selection.
4. **Candidate-space exploration:** Began moving away from purely local gradient proposals toward broader candidate search.
5. **Random candidate sampling:** Switched to sampling a larger candidate pool and choosing the point with the best acquisition value, reducing dependence on local optimiser starts.
6. **Function-specific acquisition tuning:** Introduced per-function acquisition hyperparameter changes and improved robustness concerns around noise and target normalisation.
7. **HEBO-style default:** Adopted ideas from the NeurIPS 2020 HEBO-style winning approach, including warped GP modelling and multi-objective acquisition search.
8. **Active kernel configuration:** Fixed the HEBO kernel configuration path so surrogate kernel choices actually affected proposed points; introduced a diversified portfolio across functions.
9. **Exploration-pressure tuning:** Increased `kappa`, `xi`, population size, and generations for selected functions where previous rounds suggested under-exploration.
10. **Late-round exploitation/exploration balance:** Continued using logged optimiser proposals, with strong boundary-seeking behaviour in several functions and mixed final-round returns.

### Techniques Used

- **Gaussian-process surrogate modelling:** Used to estimate both objective value and uncertainty from sparse observations.
- **Output power transforms:** Applied to reduce skew and stabilise heteroscedastic outputs.
- **Kumaraswamy input warping:** Used to model non-stationary behaviour over bounded `[0, 1]^d` inputs.
- **Configurable kernels:** Tested linear, Matérn, and additive linear-plus-Matérn assumptions.
- **MACE-style acquisition search:** Searched over multiple acquisition objectives, including `logEI`, `PI`, and `UCB`.
- **NSGA-II candidate search:** Used multi-objective evolutionary search through `pymoo` to produce candidate proposals.
- **Minimum-distance and deduplication checks:** Reduced repeated or near-duplicate candidate submissions.

## Performance

Performance is summarised using maximisation metrics:

- **Initial best:** Best objective value in the provided seed data.
- **Best submitted:** Best value observed among the first ten logged submitted rounds.
- **Improvement:** `best submitted - initial best`.
- **Round found:** First submitted round where the best submitted value was reached.
- **Beat initial best:** Whether the first ten submitted rounds improved on the seed-data best.

| Function | Initial Best | Best Submitted, First 10 Rounds | Improvement | Round Found | Beat Initial Best |
|---|---:|---:|---:|---:|---|
| 1 | `~0` | `3.636e-36` | `-7.711e-16` | 3 | No |
| 2 | `0.611205` | `0.680981` | `0.069776` | 9 | Yes |
| 3 | `-0.034835` | `-0.017925` | `0.016911` | 8 | Yes |
| 4 | `-4.025542` | `0.235585` | `4.261128` | 2 | Yes |
| 5 | `1088.859618` | `6192.712543` | `5103.852925` | 9 | Yes |
| 6 | `-0.714265` | `-0.310277` | `0.403988` | 5 | Yes |
| 7 | `1.364968` | `2.038457` | `0.673489` | 3 | Yes |
| 8 | `9.598482` | `9.956942` | `0.358460` | 5 | Yes |

Across the first ten logged submitted rounds, the optimiser improved over the initial best on **7 of 8 functions**. The largest absolute gain was on function 5, where the best logged submitted value increased from `1088.859618` to `6192.712543`.

The repository currently contains twelve submitted-output rows in `processed_data/outputs.txt`. If all twelve logged rows are included, function 1 also improves over its initial best, giving improvements on **8 of 8 functions**. The ten-round summary above is used because the brief asks for performance across ten rounds.

## Assumptions And Limitations

Core assumptions:

- The hidden objectives are deterministic or have low enough noise for GP-based modelling to remain useful.
- Objective values are comparable within each function, but not necessarily across functions.
- The bounded input domains are correctly specified as `[0, 1]^d`.
- Smoothness, locality, and uncertainty estimates from GP kernels are informative enough to guide proposals.
- Output transforms and input warping improve modelling stability rather than introducing harmful distortion.

Constraints and failure modes:

- The data are extremely small, so surrogate uncertainty can be miscalibrated.
- Boundary-seeking proposals may overfit optimiser artefacts rather than genuine objective structure.
- Kernel and acquisition choices are heuristic and can fail when the true function is discontinuous, highly noisy, or adversarial.
- Per-function tuning improves flexibility but increases the risk of overfitting to sparse feedback.
- Strong performance on logged challenge functions does not prove transfer to unrelated black-box optimisation problems.
- Some earlier-round rationale is less complete than later-cycle documentation, limiting retrospective auditability.

## Ethical Considerations

Transparency supports reproducibility because future reviewers can inspect:

- the raw initial data in `initial_data/`,
- submitted candidate traces in `processed_data/inputs.txt`,
- returned objective values in `processed_data/outputs.txt`,
- strategy changes in `strategy.md`,
- modelling rationale in `README.md`, `hyperparameters.md`, and `bbo_design_rationale.md`.

This matters for real-world adaptation. Bayesian optimisation is often used for expensive experiments, automated tuning, or resource allocation. In those settings, undocumented search choices can hide brittle assumptions, failed trials, or unsafe extrapolation. Recording the optimisation strategy, transformations, hyperparameters, and observed results makes it easier to reproduce the work, diagnose failure modes, and decide whether the method is appropriate before adapting it to higher-stakes domains.
