"""Tests for configurable priors in inference coordinates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayesian import log_posterior, parameter_prior
from bayesian.mc_sampling import emcee as emcee_sampler


def physical_prior_config() -> dict:
    return {
        "physical_posterior_transform": {
            "components": [
                {
                    "name": "Lres",
                    "steps": [
                        {"kind": "multiply", "value": np.pi / 2},
                        {"kind": "tan"},
                        {"kind": "power", "value": 3},
                    ],
                },
                {
                    "name": "kappa_sc",
                    "steps": [
                        {"kind": "multiply", "value": 0.3},
                        {"kind": "add", "value": 0.3},
                    ],
                },
            ]
        },
        "prior": {
            "type": "uniform_physical",
            "bounds": {
                "Lres": [0.0, 20.0],
                "kappa_sc": [0.3, 0.6],
            },
        },
    }


def test_uniform_physical_prior_includes_jacobian_and_bounds() -> None:
    config = physical_prior_config()
    u20 = 2 / np.pi * np.arctan(20 ** (1 / 3))
    samples = np.array(
        [
            [0.25, 0.5],
            [0.50, 0.5],
            [u20, 0.5],
            [u20 + 1.0e-4, 0.5],
        ]
    )
    values = parameter_prior.log_prior(samples, config)

    assert np.all(np.isfinite(values[:3]))
    assert values[1] > values[0]
    assert values[2] > values[1]
    assert values[3] == -np.inf


def test_uniform_physical_prior_normalizes_in_unit_coordinate() -> None:
    config = physical_prior_config()
    u20 = 2 / np.pi * np.arctan(20 ** (1 / 3))
    grid = np.linspace(1.0e-7, u20, 200_000)
    samples = np.column_stack((grid, np.full_like(grid, 0.5)))
    density = np.exp(parameter_prior.log_prior(samples, config))

    assert np.trapz(density, grid) == pytest.approx(1.0, rel=2.0e-4)


def test_initial_walkers_respect_physical_prior_support() -> None:
    config = physical_prior_config()
    positions = emcee_sampler._draw_initial_positions(
        np.random.default_rng(17),
        np.zeros(2),
        np.ones(2),
        50,
        config,
    )

    assert positions.shape == (50, 2)
    assert np.all(np.isfinite(parameter_prior.log_prior(positions, config)))


def test_log_posterior_applies_configured_physical_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = physical_prior_config()
    u20 = 2 / np.pi * np.arctan(20 ** (1 / 3))
    samples = np.array(
        [
            [0.25, 0.5],
            [0.50, 0.5],
            [u20 + 1.0e-4, 0.5],
        ]
    )
    log_posterior.g_min = np.zeros(2)
    log_posterior.g_max = np.ones(2)
    log_posterior.g_parameterization_config = config
    log_posterior.g_emulation_config = SimpleNamespace(
        analysis_settings=SimpleNamespace()
    )
    log_posterior.g_emulation_results = {}
    log_posterior.g_emulator_cov_unexplained = {}
    log_posterior.g_experimental_results = {
        "y": np.zeros(1),
        "y_err_stat": np.ones(1),
    }
    log_posterior.g_C_stat = np.eye(1)
    log_posterior.g_systematic_covariance = np.zeros((1, 1))
    monkeypatch.setattr(
        log_posterior.emulation,
        "predict",
        lambda x, *_args, **_kwargs: {
            "central_value": np.zeros((len(x), 1)),
            "cov": np.zeros((len(x), 1, 1)),
        },
    )

    actual = log_posterior.log_posterior(samples)
    expected = parameter_prior.log_prior(samples, config)

    np.testing.assert_allclose(actual[:2], expected[:2])
    assert actual[2] == -np.inf
