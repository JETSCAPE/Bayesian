"""Experimental-source reconstruction for editable, fresh-training examples."""
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian.systematic_correlation import SystematicCorrelationManager

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/hybrid_lres'
spec = importlib.util.spec_from_file_location('hybrid_training_inputs', EXAMPLE / 'experimental_inputs.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


@pytest.fixture
def metadata(tmp_path):
    manager = SystematicCorrelationManager(correlation_groups=defaultdict(list))
    manager.parse_configuration([('a', ['sys:shape', 'sys,taa:norm'], [], None)])
    manager.set_correlation_parameters({'default': '-1:1', 'shape': 'jetscape:0.2', 'norm': '-1:1'})
    ranges = [(0, 3, 'a__0-10')]
    manager.register_observable_ranges(ranges)
    manager.set_observable_pt({'a__0-10': np.array([0., .3, 1.])})
    names = manager.get_all_systematic_names()
    errors = np.zeros((3, len(names)))
    errors[:, names.index('sys:shape')] = [.1, .2, .3]
    errors[:, names.index('sys,taa:norm')] = .04
    stored = manager.to_dict()
    stored['_observable_pt'] = {key: value.tolist() for key, value in stored['_observable_pt'].items()}
    data = dict(unknown_shape_tags=['shape'], experiment=dict(
        y=[1., 2., 3.], y_err_stat=[.02]*3, y_err_syst=errors.tolist(),
        systematic_names=names, observable_ranges=ranges,
        per_observable_external_stat_cov={'a__0-10': dict(start=0, end=3, matrix=(np.eye(3)*.002).tolist())},
        correlation_manager=stored))
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('ell', [0., .05, .2, .37, .5, np.inf])
def test_exact_native_sources_at_arbitrary_length(metadata, ell):
    exp, covariance = inputs.load_experiment(metadata, ell)
    x = np.array([0., .3, 1.])
    correlation = np.eye(3) if ell == 0 else np.exp(-np.square((x[:, None]-x)/ell))
    expected = .002*np.eye(3) + .04**2 + np.outer([.1, .2, .3], [.1, .2, .3])*correlation
    np.testing.assert_allclose(covariance, expected, atol=1e-15)
    np.testing.assert_array_equal(exp['correlation_manager'].to_dict()['_observable_pt']['a__0-10'], x)


@pytest.mark.parametrize('ell', [-1., np.nan, -np.inf])
def test_invalid_lengths(metadata, ell):
    with pytest.raises(ValueError, match='nonnegative'):
        inputs.load_experiment(metadata, ell)


def test_source_identity_cannot_be_silently_changed(metadata):
    data = json.loads(metadata.read_text())
    data['unknown_shape_tags'] = []
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity mismatch'):
        inputs.load_experiment(metadata)


def test_real_covariance_round_trip_if_available():
    folder = EXAMPLE / 'data_archive'
    if not (folder / 'experiment_sources.json').exists():
        pytest.skip('Private training inputs not installed')
    with np.load(folder / 'reference_covariances.npz', allow_pickle=False) as ref:
        for ell, key in ((0., 'ell_0'), (.05, 'ell_0p05'), (.2, 'ell_0p2'), (.5, 'ell_0p5'), (np.inf, 'ell_inf')):
            _, covariance = inputs.load_experiment(folder / 'experiment_sources.json', ell)
            np.testing.assert_allclose(covariance, ref['covariance_' + key], atol=1e-14, rtol=1e-12)
