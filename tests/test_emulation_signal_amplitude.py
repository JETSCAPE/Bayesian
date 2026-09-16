"""Signal amplitude is multiplicative, PC-specific, and independent of WhiteKernel."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from bayesian.emulation import base, sk_learn


def configuration(enabled=True):
    active = ["matern", "noise"]
    if enabled:
        active.insert(1, "signal_amplitude")
    return {
        "force_retrain": False,
        "n_pc": 2,
        "observable_list": ["toy"],
        "kernels": {
            "active": active,
            "matern": {"nu": 1.5, "length_scale_bounds_factor": [0.005, 1e6]},
            "noise": {"type": "white", "args": {"noise_level": 0.5, "noise_level_bounds": [1e-5, 100]}},
            "signal_amplitude": {"constant_value_factor": 1.0, "constant_value_bounds_factor": [1e-4, 1e4]},
        },
        "GPR": {"n_restarts": 0, "alpha": 1e-8},
    }


@pytest.mark.parametrize("smooth", [Matern(nu=1.5), RBF()])
@pytest.mark.parametrize("add_constant", [False, True])
def test_only_smooth_kernel_is_scaled(smooth, add_constant):
    settings = sk_learn.SKLearnEmulatorSettings.from_config(configuration())
    kernel = smooth + WhiteKernel(0.4)
    if add_constant:
        kernel += ConstantKernel(0.3)
    y = np.array([-4.0, 0.0, 4.0])
    actual = sk_learn._kernel_for_pc(kernel, y, settings)
    x = np.array([[0.0], [0.2], [0.8]])
    expected = np.var(y) * smooth(x) + 0.4 * np.eye(3) + (0.3 if add_constant else 0.0)
    np.testing.assert_allclose(actual(x), expected)
    # WhiteKernel is zero for a cross-covariance, even with identical inputs.
    expected_cross = np.var(y) * smooth(x, x) + (0.3 if add_constant else 0.0)
    np.testing.assert_allclose(actual(x, x), expected_cross)


def test_pc_variance_initialization_bounds_and_zero_variance():
    settings = sk_learn.SKLearnEmulatorSettings.from_config(configuration())
    kernel = Matern() + WhiteKernel(0.5)
    for y in (np.array([-3.0, 0.0, 3.0]), np.zeros(4)):
        actual = sk_learn._kernel_for_pc(kernel, y, settings)
        variance = max(float(np.var(y)), 1e-8)
        assert actual.k1.k1.constant_value == variance
        np.testing.assert_allclose(actual.k1.k1.constant_value_bounds, variance * np.array([1e-4, 1e4]))
        assert actual.k2 == kernel.k2
    assert kernel == Matern() + WhiteKernel(0.5)


def test_disabled_option_preserves_legacy_kernel():
    settings = sk_learn.SKLearnEmulatorSettings.from_config(configuration(False))
    kernel = Matern() + WhiteKernel(0.5)
    assert sk_learn._kernel_for_pc(kernel, np.array([0.0, 8.0]), settings) is kernel
    sk_learn._validate_signal_amplitude_cache({}, settings)


@pytest.mark.parametrize(
    "amplitude",
    [
        None,
        {},
        {"constant_value_factor": 1.0, "constant_value_bounds": [1e-4, 1e4]},
        {"constant_value_factor": "typo", "constant_value_bounds_factor": [1e-4, 1e4]},
        {"constant_value_factor": 0.0, "constant_value_bounds_factor": [1e-4, 1e4]},
        {"constant_value_factor": np.nan, "constant_value_bounds_factor": [1e-4, 1e4]},
        {"constant_value_factor": 1.0, "constant_value_bounds_factor": [1e-4, np.inf]},
        {"constant_value_factor": 1.0, "constant_value_bounds_factor": [-1.0, 2.0]},
        {"constant_value_factor": 1.0, "constant_value_bounds_factor": [2.0, 1.0]},
        {"constant_value_factor": 1.0, "constant_value_bounds_factor": [1.0, 1.0]},
        {"constant_value_factor": 3.0, "constant_value_bounds_factor": [1.0, 2.0]},
        {"constant_value_factor": 1.0, "constant_value_bounds_factor": [1.0]},
    ],
)
def test_invalid_amplitude_rejected(amplitude):
    config = configuration()
    config["kernels"]["signal_amplitude"] = amplitude
    with pytest.raises(ValueError, match="signal_amplitude"):
        sk_learn.SKLearnEmulatorSettings.from_config(config)


@pytest.fixture
def toy_data(tmp_path, monkeypatch):
    x = np.linspace(0.02, 0.98, 20)[:, None]
    y = np.column_stack((np.sin(x[:, 0] * 3), np.cos(x[:, 0] * 4), np.sin(x[:, 0] * 5)))
    monkeypatch.setattr(sk_learn.data_IO, "predictions_matrix_from_h5", lambda **_kwargs: y)
    monkeypatch.setattr(sk_learn.data_IO, "design_array_from_h5", lambda *_args, **_kwargs: x)
    analysis = SimpleNamespace(
        output_dir=tmp_path,
        io=SimpleNamespace(observables_filename="unused.h5"),
        parameterization="toy",
        raw_analysis_config={"parameterization": {"toy": {"min": [0.0], "max": [1.0]}}},
    )
    return x, y, analysis


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("enabled", [False, True])
def test_standard_fit_predict_and_cache_roundtrip(toy_data, enabled):
    x, _, analysis = toy_data
    settings = sk_learn.SKLearnEmulatorSettings.from_config(configuration(enabled))
    result = sk_learn.fit_emulator(settings, analysis)
    base.IO.write_emulator(result, settings, analysis)
    loaded = base.IO.read_emulator(settings, analysis)
    assert sk_learn.fit_emulator(settings, analysis) == {}
    points = np.array([[0.15], [0.6]])
    prediction = sk_learn.predict(points, loaded, settings)
    assert np.isfinite(prediction["central_value"]).all()
    assert np.linalg.eigvalsh(prediction["cov"]).min() >= -1e-12
    for i, actual in enumerate(result["emulators"]):
        scores = result["PCA"]["Y_pca_truncated"][:, i]
        smooth = Matern(length_scale=np.array([1.0]), length_scale_bounds=np.array([[0.005, 1e6]]), nu=1.5)
        if enabled:
            v = max(float(np.var(scores)), 1e-8)
            smooth = ConstantKernel(v, (v * 1e-4, v * 1e4)) * smooth
        expected = GaussianProcessRegressor(
            kernel=smooth + WhiteKernel(0.5, (1e-5, 100)),
            alpha=1e-8,
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=False,
        ).fit(x, scores)
        np.testing.assert_allclose(actual.kernel_.theta, expected.kernel_.theta, atol=1e-9)
        for actual_values, expected_values in zip(
            actual.predict(points, return_std=True), expected.predict(points, return_std=True), strict=True
        ):
            np.testing.assert_allclose(actual_values, expected_values, atol=1e-10)
        assert actual.normalize_y is False
        assert actual.alpha == 1e-8
    assert ("signal_amplitude" in result) == enabled


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("initially_enabled", [False, True])
def test_changed_settings_require_retraining(toy_data, initially_enabled):
    _, _, analysis = toy_data
    old = sk_learn.SKLearnEmulatorSettings.from_config(configuration(initially_enabled))
    result = sk_learn.fit_emulator(old, analysis)
    base.IO.write_emulator(result, old, analysis)
    new = sk_learn.SKLearnEmulatorSettings.from_config(configuration(not initially_enabled))
    with pytest.raises(ValueError, match="retrain"):
        sk_learn.fit_emulator(new, analysis)
    with pytest.raises(ValueError, match="retrain"):
        sk_learn.predict(np.array([[0.4]]), result, new)
    new.base_settings.force_retrain = True
    replacement = sk_learn.fit_emulator(new, analysis)
    assert ("signal_amplitude" in replacement) != initially_enabled


def test_changed_amplitude_bounds_reject_cache():
    config = configuration()
    settings = sk_learn.SKLearnEmulatorSettings.from_config(config)
    result = {"signal_amplitude": sk_learn._signal_amplitude_settings(settings)}
    modified = copy.deepcopy(config)
    modified["kernels"]["signal_amplitude"]["constant_value_bounds_factor"] = [1e-3, 1e3]
    with pytest.raises(ValueError, match="retrain"):
        sk_learn._validate_signal_amplitude_cache(result, sk_learn.SKLearnEmulatorSettings.from_config(modified))
