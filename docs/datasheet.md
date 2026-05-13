# Datasheet

## Motivation

This dataset supports a black-box Bayesian optimisation challenge. The task is to propose one high-value next query point for each of eight unknown objective functions.

Each objective maps continuous inputs in `[0, 1]^d` to a scalar score. The optimisation goal is to maximise the returned value under a tight evaluation budget.

## Composition

The dataset contains initial observed inputs and outputs for eight separate black-box functions.

- Format: NumPy `.npy` arrays.
- Location: `initial_data/`.
- Per-function files: `initial_inputs.npy` and `initial_outputs.npy`.
- Missing values: no missing or NaN values were found in the input or output arrays.

| Function | Input Shape | Output Shape | Dimension | Output Range |
|---|---:|---:|---:|---:|
| 1 | `(10, 2)` | `(10,)` | 2 | `-0.003606` to `~0` |
| 2 | `(10, 2)` | `(10,)` | 2 | `-0.065624` to `0.611205` |
| 3 | `(15, 3)` | `(15,)` | 3 | `-0.398926` to `-0.034835` |
| 4 | `(30, 4)` | `(30,)` | 4 | `-32.6257` to `-4.02554` |
| 5 | `(20, 4)` | `(20,)` | 4 | `0.11294` to `1088.86` |
| 6 | `(20, 5)` | `(20,)` | 5 | `-2.57117` to `-0.714265` |
| 7 | `(30, 6)` | `(30,)` | 6 | `0.002701` to `1.36497` |
| 8 | `(40, 8)` | `(40,)` | 8 | `5.59219` to `9.59848` |

Known gaps:

- The objective-function definitions are hidden.
- Feature dimensions have no semantic labels.
- The data are small by design, so coverage of each search space is sparse.
- The noise process, if any, is unknown.

## Collection Process

The initial observations were supplied as challenge seed data. Follow-up query candidates are generated using Bayesian optimisation.

The current strategy uses a HEBO-style pipeline:

- output power transforms to reduce skew and heteroscedasticity,
- Kumaraswamy input warping to handle non-stationarity over `[0, 1]^d`,
- Gaussian-process surrogate modelling,
- multi-objective acquisition search over `logEI`, `PI`, and `UCB`.

Earlier cycles used simpler Gaussian-process acquisition strategies. Later cycles diversified kernels and exploration settings by function. The repository records optimisation cycles 5-9, but exact calendar collection dates are not documented.

## Preprocessing And Uses

The raw challenge data are preserved as `.npy` arrays. Additional flattened text exports exist in `processed_data/`.

Modelling-time transformations include output power transforms and input warping. These transformations are used by the optimiser and should not be treated as replacements for the raw data.

Intended uses:

- Bayesian optimisation experiments.
- Surrogate-model comparison.
- Acquisition-function evaluation.
- Replay sanity checks under limited data.

Inappropriate uses:

- Supervised-learning benchmarking as if the data were independent, identically distributed, and richly sampled.
- Drawing scientific conclusions about real-world systems.
- Extrapolating outside `[0, 1]^d`.
- Assuming feature dimensions have interpretable meanings.

## Distribution And Maintenance

The dataset is available locally in this repository under `initial_data/`.

Terms of use are not explicitly stated in the repository, however, please treat the data as course or challenge material and avoid redistribution.

Future maintenance should record newly submitted query points, returned outputs, cycle number, optimiser configuration, and timestamp for reproducibility.
Currently maintained by https://github.com/jones-agyemang
