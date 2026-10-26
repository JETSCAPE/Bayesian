"""Portable entry-point checks on small synthetic tables, never paper caches."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bayesian import data_IO
from bayesian.systematic_correlation import SystematicCorrelationManager

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/hybrid_lres'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


run = module('portable_paper_analysis', EXAMPLE / 'run_analysis.py')


def test_selection_and_config_reject_typos():
    rows = [('x__hadron__pt_alice', 10, 20), ('x__hadron__pt_alice', 30, 40),
            ('x__inclusive_jet__pt_cms', 100, 200), ('x__rg_atlas', 0, .1), ('x__rg_atlas', .1, .2)]
    cfg = run.validate_config({'selection': {'drop_first_angular_bins': {'rg': 1}}})
    assert run.select_features(rows, cfg['selection']) == [tuple(row) for row in rows[1:3] + rows[4:]]
    with pytest.raises(ValueError, match='Unknown keys'):
        run.validate_config({'experiment': {'length': .2}})
    with pytest.raises(ValueError, match='Prior'):
        run.validate_config({'prior': {'kind': 'uniform_full_square'}})


def test_inference_math_prior_and_nested_grid():
    q, k = np.linspace(0, 20, 9), np.linspace(.3, .6, 7)
    ll = np.zeros((9, 7))
    physical, tv = run.ni.convergence(ll, q, k)
    assert tv < 1e-14
    unit, _ = run.ni.convergence(ll, q, k, 'uniform_unit_same_support')
    assert not np.array_equal(physical, unit)
    assert run.ni.summary(unit, q, k, 'uniform_unit_same_support')['full_unit_prior'] is False
    np.testing.assert_allclose(run.ni.physical(run.ni.unit(q)), q, atol=1e-13)


def test_fresh_train_infer_plot_and_config_reuse(tmp_path, monkeypatch):
    # Reuse the independent native trainer's synthetic full-noise fixture.
    native = module('portable_native_fixture', ROOT / 'tests/test_paper_training.py')
    sample = native.inputs.__wrapped__(tmp_path)
    raw = sample['raw']
    for suffix in ('', '_validation'):
        x = raw['Design' + suffix][:, 0]
        raw['Design' + suffix] = np.column_stack((x, x**2))
    inputs = tmp_path / 'portable'
    inputs.mkdir()
    data_IO.write_dict_to_h5(raw, inputs, 'accepted_observables.h5', verbose=False)
    (inputs / 'full_training.npz').write_bytes(sample['covariance_path'].read_bytes())
    template = sample['config']
    template['analyses']['analysis_test']['parameterization']['unit'] = dict(
        names=['x', 'z'], min=[0., 0.], max=[1., 1.])
    (inputs / 'native_template.yaml').write_text(yaml.safe_dump(template))
    rows = sample['full_features']
    n = len(rows)
    manager = SystematicCorrelationManager()
    ranges = []
    for start in range(0, n, 9):
        ranges.append((start, start + 9, rows[start][0]))
    manager.register_observable_ranges(ranges)
    metadata = dict(features=rows, unknown_shape_tags=[], experiment=dict(
        y=np.ones(n).tolist(), y_err_stat=np.full(n, 100.).tolist(),
        y_err_syst=np.empty((n, 0)).tolist(), systematic_names=[],
        observable_ranges=ranges, correlation_manager=manager.to_dict()))
    (inputs / 'experiment_sources.json').write_text(json.dumps(metadata))
    np.savez_compressed(inputs / 'grid_axes.npz', q=np.linspace(0, 20, 361), kappa=np.linspace(.3, .6, 241))
    config = tmp_path / 'run.yaml'
    config.write_text(yaml.safe_dump(dict(selection=dict(sectors=['jets']),
        training=dict(n_restarts=0), grid=dict(mode='compact', counts=[5, 5], batch_size=8))))
    output = tmp_path / 'fresh'
    run.main(['--config', str(config), '--inputs', str(inputs), '--output', str(output)])
    assert (output / 'COMPLETE.json').is_file()
    assert (output / 'figures/posterior_joint.svg').is_file()
    assert (output / 'figures/predictions_01.pdf').is_file()
    report = json.loads((output / 'training/training_report.json').read_text())
    assert len(report['features']) == 9
    assert report['training_ids'] == list(range(6))
    assert report['holdout_ids'] == [38]
    summary = json.loads((output / 'inference/summary.json').read_text())
    assert summary['converged'] and summary['exploratory']
    # An experimental-correlation change can reuse this fresh selected model.
    cfg = run.validate_config(yaml.safe_load(config.read_text()))
    cfg['experiment']['ell'] = .37
    loaded = run.load_inputs(inputs, cfg)
    reused = run.load_fresh_training(output / 'training', loaded, cfg)
    assert reused['report']['joint_fit_id'] == report['joint_fit_id']
    # A new output mask or changed native template must require a new fit.
    cfg['selection']['sectors'] = ['hadrons']
    cfg['selection']['hadron_min_pt'] = 10
    with pytest.raises(ValueError):
        run.load_fresh_training(output / 'training', run.load_inputs(inputs, cfg), cfg)
    cfg['selection']['sectors'] = ['jets']
    loaded = run.load_inputs(inputs, cfg)
    loaded['template']['analyses']['analysis_test']['parameters']['emulators']['jet']['n_pc'] = 2
    with pytest.raises(ValueError, match='Native template differs'):
        run.load_fresh_training(output / 'training', loaded, cfg)
    with pytest.raises(FileExistsError):
        run.main(['--config', str(config), '--inputs', str(inputs), '--output', str(output)])
