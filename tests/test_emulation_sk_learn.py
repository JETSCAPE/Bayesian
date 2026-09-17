import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bayesian.emulation import sk_learn


class _DummySettings:
    n_pc = 1


class _ZeroVarianceEmulator:
    def predict(self, parameters, return_cov=False):
        central = np.zeros(parameters.shape[0])
        if return_cov:
            return central, np.zeros((parameters.shape[0], parameters.shape[0]))
        return central


def test_pca_truncation_covariance_is_batch_invariant():
    y = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0],
            [3.0, 5.0, 8.0],
            [4.0, 7.0, 11.0],
        ]
    )
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit_transform(scaler.fit_transform(y))

    additional_covariance = np.array(
        [
            [0.10, 0.02, 0.00],
            [0.02, 0.20, 0.03],
            [0.00, 0.03, 0.30],
        ]
    )
    results = {
        "PCA": {
            "pca": pca,
            "scaler": scaler,
        },
        "emulators": [_ZeroVarianceEmulator()],
    }

    parameters = np.array([[0.1, 0.2], [0.3, 0.4]])
    batch_predictions = sk_learn.predict(parameters, results, _DummySettings(), additional_covariance)
    single_predictions = [
        sk_learn.predict(parameters[i : i + 1], results, _DummySettings(), additional_covariance)
        for i in range(parameters.shape[0])
    ]

    for i, single_prediction in enumerate(single_predictions):
        np.testing.assert_allclose(batch_predictions["central_value"][i], single_prediction["central_value"][0])
        np.testing.assert_allclose(batch_predictions["cov"][i], single_prediction["cov"][0])


def test_pca_truncation_covariance_matches_residual_sample_covariance():
    y = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0],
            [3.0, 5.0, 8.0],
            [4.0, 7.0, 11.0],
            [5.0, 11.0, 16.0],
        ]
    )
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)
    pca = PCA(svd_solver="full")
    y_pca = pca.fit_transform(y_scaled)
    retained_scores = np.zeros_like(y_pca)
    retained_scores[:, : _DummySettings.n_pc] = y_pca[
        :, : _DummySettings.n_pc
    ]
    residual = y_scaled - pca.inverse_transform(retained_scores)
    expected = np.cov(residual, rowvar=False, ddof=1)

    actual = sk_learn.compute_emulator_cov_unexplained(
        _DummySettings(),
        {"PCA": {"pca": pca}},
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-14)
