*What is the main technical justification for your current BBO approach? Which aspect of prior research or established methods supports your choice?*
The current approach is a HEBO-style Bayesian optimisation pipeline rather than a plain GP plus one acquisition. The main justification is robustness: the observed outputs across the eight challenge functions vary sharply in scale, skewness, and apparent noise level, so a vanilla GP with a single EI or UCB search is too brittle. I now use output power transforms to stabilise variance, Kumaraswamy input warping to handle non-stationarity over `[0, 1]^d`, an additive `Linear + Matern32` GP to capture both broad trends and local structure, and a Pareto-style proposer that balances `logEI`, `PI`, and `UCB` instead of trusting one acquisition everywhere.

*Which academic papers have you used to guide your design? Which ideas or techniques from the literature are most relevant, and how do they strengthen your project?*
Two sources are central. First, Rasmussen and Williams (2006) provides the GP foundation for surrogate modelling, uncertainty estimation, and kernel-based inductive bias. Second, the Huawei Noah's Ark NeurIPS 2020 BBO winner paper provides the design pattern that most directly shaped this implementation. The most relevant ideas from that paper are:
- power transformation for heteroscedastic outputs,
- input warping for non-stationary search spaces,
- an additive linear-plus-Matern surrogate kernel,
- stochastic mean exploration,
- multi-objective acquisition search with NSGA-II over `logEI`, `PI`, and `UCB`.

These ideas strengthen the project because they directly target the failure modes shown in my logged challenge data: unstable output scales, noisy regions, and conflicting recommendations from different acquisitions.

*Which third-party libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach? Why are these the right choices compared with possible alternatives?*
The central stack is `scikit-learn`, `SciPy`, `NumPy`, and `pymoo`.
- `scikit-learn` provides `GaussianProcessRegressor`, `PowerTransformer`, and kernel primitives needed for the warped GP surrogate.
- `SciPy` provides the optimisation routines used to fit Kumaraswamy warp parameters by maximising GP log-marginal likelihood.
- `NumPy` supports vectorised acquisition scoring, warping, distance checks, and candidate post-processing.
- `pymoo` provides the NSGA-II implementation needed for the MACE proposer.

This combination is appropriate because it keeps the project lightweight and traceable while still supporting the key HEBO mechanisms. I did not replace the optimiser with the official HEBO package because I wanted the reasoning and implementation details to remain explicit inside my own project.

*How do you plan to document and present these justifications in your GitHub repository so that peers, facilitators and future employers can clearly understand your reasoning?*
I will make the reasoning traceable across the repo:
- `README.md` explains the new default HEBO-style architecture and why it replaced the old single-acquisition setup.
- `hebo.py` contains the surrogate, transforms, and MACE proposer so the main research ideas are visible in one place.
- `utils/cycle_parameters.py` exposes the public config surface for `proposal.mode`, HEBO controls, and the retained baseline path.
- `hyperparameters.md` documents the highest-leverage controls and why they matter.
- `replay_sanity.py` gives a reproducible comparison between baseline and HEBO on a shared candidate pool.

*Looking ahead, what additional sources (research, benchmarks, software) might you consult to continue refining your strategy?*
The next places to push are more faithful warped-GP implementations, benchmark replays, and stronger ablations. I would look at the official HEBO repository, Bayesmark-style benchmarks, and recent work on trust-region BO and robust batch acquisition selection. The next technical questions are whether the stochastic mean coefficient should adapt per function and whether the warp fit should be regularised more aggressively for very small data regimes.
