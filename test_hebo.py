import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, Matern

from hebo import (
    KumaraswamyWarp,
    build_hebo_kernel,
    fit_hebo_surrogate,
    log_expected_improvement,
    probability_of_improvement,
    propose_next_hebo,
    select_output_transform,
    upper_confidence_bound,
)
from utils.loader import load_input_data, load_output_data


def _flatten_kernel_tree(kernel):
    kernels = [kernel]
    flattened = []
    while kernels:
        current = kernels.pop()
        flattened.append(current)
        for attr in ("k1", "k2"):
            child = getattr(current, attr, None)
            if child is not None:
                kernels.append(child)
    return flattened


def test_selects_box_cox_for_positive_targets():
    transform = select_output_transform(np.array([1.0, 2.0, 3.0]))
    assert transform.method == "box-cox"
    restored = transform.inverse_transform(transform.transform(np.array([1.0, 2.0, 3.0])))
    assert np.allclose(restored, np.array([1.0, 2.0, 3.0]), atol=1e-6)


def test_selects_box_cox_for_negative_targets():
    values = np.array([-3.0, -2.0, -1.0])
    transform = select_output_transform(values)
    assert transform.method == "box-cox"
    assert transform.offset > 0.0
    restored = transform.inverse_transform(transform.transform(values))
    assert np.allclose(restored, values, atol=1e-6)


def test_selects_yeo_johnson_for_mixed_sign_targets():
    values = np.array([-2.0, 0.0, 3.5])
    transform = select_output_transform(values)
    assert transform.method == "yeo-johnson"
    restored = transform.inverse_transform(transform.transform(values))
    assert np.allclose(restored, values, atol=1e-6)


def test_kumaraswamy_warp_is_monotone_and_reversible():
    warp = KumaraswamyWarp(a=np.array([1.8, 0.7]), b=np.array([0.9, 2.3]))
    X = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.4, 0.6],
            [0.8, 0.9],
        ]
    )
    warped = warp.transform(X)
    restored = warp.inverse_transform(warped)

    assert np.all(np.diff(warped[:, 0]) > 0.0)
    assert np.all(np.diff(warped[:, 1]) > 0.0)
    assert np.allclose(restored, X, atol=1e-6)


def test_build_hebo_kernel_uses_requested_kernel_type():
    kernel = build_hebo_kernel(
        2,
        {
            "type": "matern52",
            "length_scale": [0.4, 0.8],
            "length_scale_bounds": (1e-3, 1e1),
        },
    )
    kernels = _flatten_kernel_tree(kernel)

    assert not any(isinstance(part, DotProduct) for part in kernels)
    matern_parts = [part for part in kernels if isinstance(part, Matern)]
    assert len(matern_parts) == 1
    assert matern_parts[0].nu == 2.5
    assert np.allclose(matern_parts[0].length_scale, np.array([0.4, 0.8]))


def test_acquisitions_remain_finite_under_tiny_variance():
    mu = np.array([0.1, 0.2, 0.3])
    sigma = np.array([0.0, 1e-14, 1e-9])
    y_best = 0.25

    log_ei = log_expected_improvement(mu, sigma, y_best, xi=0.0)
    pi = probability_of_improvement(mu, sigma, y_best, xi=0.0)
    ucb = upper_confidence_bound(mu, sigma, kappa=2.0)

    assert np.all(np.isfinite(log_ei))
    assert np.all(np.isfinite(pi))
    assert np.all(np.isfinite(ucb))


def test_stochastic_mean_shifts_gp_mean():
    X = load_input_data(2)
    y = load_output_data(2)

    low_noise_surrogate = fit_hebo_surrogate(
        X,
        y,
        surrogate_cfg={
            "output_transform": "auto",
            "input_warping": True,
            "stochastic_mean_xi": 0.0,
            "warp_multistarts": 1,
            "warp_maxiter": 10,
            "gp_restarts": 0,
        },
        random_state=2,
    )
    high_noise_surrogate = fit_hebo_surrogate(
        X,
        y,
        surrogate_cfg={
            "output_transform": "auto",
            "input_warping": True,
            "stochastic_mean_xi": 1.0,
            "warp_multistarts": 1,
            "warp_maxiter": 10,
            "gp_restarts": 0,
        },
        random_state=2,
    )
    probe = np.array([[0.5, 0.5]])

    mu_without_noise, _ = low_noise_surrogate.predict(probe)
    mu_with_noise, _ = high_noise_surrogate.predict(probe)

    assert mu_with_noise[0] > mu_without_noise[0]


def test_negative_targets_preserve_maximisation_ordering():
    X = np.array([[0.1], [0.5], [0.9]])
    y = np.array([-3.0, -2.0, -1.0])

    surrogate = fit_hebo_surrogate(
        X,
        y,
        surrogate_cfg={
            "output_transform": "auto",
            "input_warping": False,
            "gp_restarts": 0,
        },
        random_state=0,
    )

    best_original_index = int(np.argmax(y))
    best_transformed_index = int(np.argmax(surrogate.y_transformed))

    assert best_transformed_index == best_original_index


def test_fit_hebo_surrogate_passes_kernel_config_to_gp():
    X = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.9], [0.9, 0.3]])
    y = np.array([0.2, 0.5, 0.7, 0.4])

    surrogate = fit_hebo_surrogate(
        X,
        y,
        surrogate_cfg={
            "output_transform": "auto",
            "input_warping": False,
            "kernel": {
                "type": "matern52",
                "length_scale": [0.25, 0.75],
                "length_scale_bounds": (1e-3, 1e1),
            },
            "gp_restarts": 0,
        },
        random_state=0,
    )
    kernels = _flatten_kernel_tree(surrogate.gp.kernel_)

    assert not any(isinstance(part, DotProduct) for part in kernels)
    matern_parts = [part for part in kernels if isinstance(part, Matern)]
    assert len(matern_parts) == 1
    assert matern_parts[0].nu == 2.5


def test_hebo_smoke_proposes_one_bounded_non_duplicate_point_per_function():
    proposal_cfg = {
        "mode": "hebo",
        "optimizer": "mace_nsga2",
        "objectives": ["log_ei", "pi", "ucb"],
        "population_size": 24,
        "generations": 8,
        "min_distance": 1e-3,
        "dedupe_tol": 1e-6,
        "selection": "balanced_pareto",
        "xi": 0.0,
        "kappa": 2.0,
    }
    surrogate_cfg = {
        "output_transform": "auto",
        "input_warping": True,
        "kernel": "linear_plus_matern32",
        "stochastic_mean_xi": 1.0,
        "warp_multistarts": 1,
        "warp_maxiter": 10,
        "gp_restarts": 0,
    }

    for func_id in range(1, 9):
        X = load_input_data(func_id)
        y = load_output_data(func_id)
        _, candidate = propose_next_hebo(
            X,
            y,
            surrogate_cfg=surrogate_cfg,
            proposal_cfg=proposal_cfg,
            seed=func_id,
        )

        assert candidate.shape == (X.shape[1],)
        assert np.all(candidate >= 1e-6)
        assert np.all(candidate <= 1.0 - 1e-6)

        duplicate_mask = np.max(np.abs(X - candidate), axis=1) <= proposal_cfg["dedupe_tol"]
        assert not np.any(duplicate_mask)
