import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from silx.io.dictdump import dicttoh5
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from bayesian.emulation import base as emulation_base
from bayesian.emulation import sk_learn


def _emulator_config(
    *,
    normalize_y: bool | None = None,
    use_prediction_statistical_uncertainty: bool | None = None,
) -> dict:
    config = {
        "force_retrain": False,
        "emulator_package": "sk_learn",
        "n_pc": 1,
        "kernels": {
            "active": ["matern"],
            "matern": {
                "nu": 1.5,
                "length_scale_bounds_factor": [0.1, 10.0],
            },
        },
        "GPR": {
            "n_restarts": 0,
            "alpha": 1.0e-8,
        },
        "observable_list": ["test"],
    }
    if normalize_y is not None:
        config["GPR"]["normalize_y"] = normalize_y
    if use_prediction_statistical_uncertainty is not None:
        config["GPR"]["use_prediction_statistical_uncertainty"] = (
            use_prediction_statistical_uncertainty
        )
    return config


def _analysis_settings(tmp_path=None):
    return SimpleNamespace(
        output_dir=tmp_path,
        io=SimpleNamespace(observables_filename="observables.h5"),
        parameterization="unit",
        raw_analysis_config={
            "parameterization": {
                "unit": {
                    "names": ["x"],
                    "min": [0.0],
                    "max": [1.0],
                }
            }
        },
    )


def test_methodology_options_are_explicit_and_backward_compatible():
    config = _emulator_config()

    settings = sk_learn.SKLearnEmulatorSettings.from_config(config)
    assert settings.normalize_y is False
    assert settings.random_state is None
    assert settings.use_prediction_statistical_uncertainty is False

    config["GPR"]["normalize_y"] = True
    config["GPR"]["random_state"] = 20260728
    config["GPR"]["use_prediction_statistical_uncertainty"] = True
    settings = sk_learn.SKLearnEmulatorSettings.from_config(config)
    assert settings.normalize_y is True
    assert settings.random_state == 20260728
    assert settings.use_prediction_statistical_uncertainty is True


def test_normalized_gp_restores_target_scale_in_mean_and_covariance():
    settings = sk_learn.SKLearnEmulatorSettings.from_config(
        _emulator_config(normalize_y=True)
    )
    kernel = Matern(
        length_scale=0.4,
        length_scale_bounds="fixed",
        nu=1.5,
    ) + WhiteKernel(
        noise_level=0.03,
        noise_level_bounds="fixed",
    )
    x = np.linspace(0.0, 1.0, 12)[:, np.newaxis]
    y = np.sin(2 * np.pi * x[:, 0])
    x_predict = np.array([[0.15], [0.55], [0.9]])

    base = sk_learn._build_gaussian_process(kernel, settings).fit(x, y)
    scaled = sk_learn._build_gaussian_process(kernel, settings).fit(x, 25 * y)
    mean, covariance = base.predict(x_predict, return_cov=True)
    scaled_mean, scaled_covariance = scaled.predict(x_predict, return_cov=True)

    np.testing.assert_allclose(scaled_mean, 25 * mean)
    np.testing.assert_allclose(scaled_covariance, 25**2 * covariance)


def test_training_metadata_detects_settings_and_array_changes(tmp_path):
    y = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y_err_stat = np.full_like(y, 0.1)
    design = np.array([[0.1], [0.5], [0.9]])
    normalized = sk_learn.SKLearnEmulatorSettings.from_config(
        _emulator_config(normalize_y=True)
    )
    error_aware = sk_learn.SKLearnEmulatorSettings.from_config(
        _emulator_config(
            normalize_y=True,
            use_prediction_statistical_uncertainty=True,
        )
    )
    historical = sk_learn.SKLearnEmulatorSettings.from_config(
        _emulator_config(normalize_y=False)
    )
    analysis_settings = _analysis_settings()

    expected = sk_learn._training_metadata(
        normalized,
        analysis_settings,
        y,
        design,
    )
    assert expected != sk_learn._training_metadata(
        historical,
        analysis_settings,
        y,
        design,
    )
    assert expected != sk_learn._training_metadata(
        normalized,
        analysis_settings,
        y + 0.01,
        design,
    )
    assert expected != sk_learn._training_metadata(
        error_aware,
        analysis_settings,
        y,
        design,
        y_err_stat,
    )
    assert sk_learn._training_metadata(
        error_aware,
        analysis_settings,
        y,
        design,
        y_err_stat,
    ) != sk_learn._training_metadata(
        error_aware,
        analysis_settings,
        y,
        design,
        y_err_stat + 0.01,
    )
    sk_learn._validate_cached_emulator(
        pickle.loads(pickle.dumps({"training_metadata": expected})),
        expected,
        tmp_path / "emulator.pkl",
    )

    with pytest.raises(RuntimeError, match="different settings or input arrays"):
        sk_learn._validate_cached_emulator(
            {"training_metadata": expected},
            sk_learn._training_metadata(
                historical,
                analysis_settings,
                y,
                design,
            ),
            tmp_path / "emulator.pkl",
        )


def test_unverifiable_legacy_cache_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="no training metadata"):
        sk_learn._validate_cached_emulator(
            {"PCA": {}, "emulators": []},
            {"version": 1},
            tmp_path / "emulator.pkl",
        )


def test_parameterization_changes_invalidate_training_metadata():
    settings = sk_learn.SKLearnEmulatorSettings.from_config(_emulator_config())
    y = np.array([[1.0], [2.0], [3.0]])
    design = np.array([[0.1], [0.5], [0.9]])
    original = _analysis_settings()
    changed = _analysis_settings()
    changed.raw_analysis_config["parameterization"]["unit"]["max"] = [2.0]

    assert sk_learn._training_metadata(
        settings,
        original,
        y,
        design,
    ) != sk_learn._training_metadata(
        settings,
        changed,
        y,
        design,
    )


def test_additional_name_positional_argument_remains_compatible():
    config = _emulator_config()
    base_settings = emulation_base.BaseEmulatorSettings.from_emulator_settings(
        config
    )
    settings = sk_learn.SKLearnEmulatorSettings(
        base_settings,
        1,
        None,
        {"matern": config["kernels"]["matern"]},
        0,
        1.0e-8,
        config,
        "legacy_name",
    )

    assert settings.additional_name == "legacy_name"
    assert settings.normalize_y is False


def test_prediction_uncertainties_are_projected_to_normalized_pc_variance():
    y = np.array(
        [
            [1.0, 10.0],
            [2.0, 14.0],
            [4.0, 18.0],
            [8.0, 22.0],
        ]
    )
    y_err_stat = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
            [0.4, 0.7],
        ]
    )
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="full")
    y_pca = pca.fit_transform(scaler.fit_transform(y))
    jitter = 1.0e-8

    alpha = sk_learn._project_prediction_uncertainties_to_pc_space(
        Y_err_stat=y_err_stat,
        scaler=scaler,
        pca=pca,
        Y_pca_truncated=y_pca,
        normalize_y=True,
        jitter=jitter,
    )

    scaled_variance = (y_err_stat / scaler.scale_) ** 2
    expected = scaled_variance @ (pca.components_**2).T
    expected /= np.std(y_pca, axis=0) ** 2
    expected += jitter
    np.testing.assert_allclose(alpha, expected)
    assert sk_learn._project_prediction_uncertainties_to_pc_space(
        Y_err_stat=None,
        scaler=scaler,
        pca=pca,
        Y_pca_truncated=y_pca,
        normalize_y=True,
        jitter=jitter,
    ) is None


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_emulator_uses_prediction_statistical_uncertainty(tmp_path):
    observable = "5020__PbPb__hadron__pt_ch_cms____0-5"
    design = np.linspace(0.0, 1.0, 10)[:, np.newaxis]
    y = np.vstack(
        [
            0.4 + 0.2 * design[:, 0],
            0.7 - 0.1 * design[:, 0] ** 2,
        ]
    )
    y_err_stat = np.vstack(
        [
            np.linspace(0.01, 0.03, design.shape[0]),
            np.linspace(0.02, 0.04, design.shape[0]),
        ]
    )
    dicttoh5(
        {
            "Design": design,
            "Prediction": {
                observable: {
                    "y": y,
                    "y_err_stat": y_err_stat,
                }
            },
        },
        str(tmp_path / "observables.h5"),
    )
    analysis_settings = _analysis_settings(tmp_path)
    config = _emulator_config(
        normalize_y=True,
        use_prediction_statistical_uncertainty=True,
    )
    config["observable_list"] = [observable]
    settings = sk_learn.SKLearnEmulatorSettings.from_config(config)

    result = sk_learn.fit_emulator(settings, analysis_settings)

    alpha_per_pc = result["PCA"]["alpha_per_pc"]
    assert alpha_per_pc.shape == (design.shape[0], settings.n_pc)
    assert np.ptp(alpha_per_pc[:, 0]) > 0
    np.testing.assert_allclose(result["emulators"][0].alpha, alpha_per_pc[:, 0])
    assert result["training_metadata"]["settings"][
        "use_prediction_statistical_uncertainty"
    ]

    emulation_base.IO.write_emulator(
        emulator_output=result,
        emulator_settings=settings,
        analysis_settings=analysis_settings,
    )
    assert sk_learn.fit_emulator(settings, analysis_settings) == {}

    analysis_settings.raw_analysis_config["parameterization"]["unit"]["max"] = [
        2.0
    ]
    with pytest.raises(RuntimeError, match="different settings or input arrays"):
        sk_learn.fit_emulator(settings, analysis_settings)
