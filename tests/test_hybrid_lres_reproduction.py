"""Safe-cache contracts and optional full portable rendering integration."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/hybrid_lres'
sys.path.insert(0, str(EXAMPLE))
from cache_io import Bundle, SCHEMA, array_digest, safe_child, sha


@pytest.fixture(scope='module')
def bundle():
    if not (EXAMPLE / 'numerical_cache/manifest.json').exists():
        if os.environ.get('HYBRID_LRES_RENDER_TEST') == '1':
            pytest.fail('Explicit render validation requires the separate private numerical_cache artifact')
        pytest.skip('Separate private numerical_cache artifact is not installed; safe-cache/source tests still run')
    return Bundle(EXAMPLE / 'numerical_cache')


def test_all_18_manuscript_figures_are_explicit(bundle):
    figures = bundle.manifest['manuscript_figures']
    assert len(figures) == len(set(figures)) == 18
    assert set(figures) == set(bundle.json('manuscript_reference'))
    assert all(bundle.json('manuscript_reference')[f]['science'] for f in figures)


def test_full_170_bin_prediction_galleries_and_routes(bundle):
    assert len(bundle.manifest['models']) == 25
    for case, name in bundle.manifest['models'].items():
        meta = bundle.json(name)
        predicted = bundle.npz('predictive_' + case)
        assert predicted['features170'].shape == (170, 3)
        assert predicted['quantiles'].shape == (5, 170)
        np.testing.assert_array_equal(predicted['probabilities'], [.025, .16, .5, .84, .975])
        assert (np.diff(predicted['quantiles'], axis=0) >= 0).all()
        own = {(n, float(a), float(b)) for n, a, b in meta['spec']['features']}
        selected = np.array([(n, float(a), float(b)) in own for n, a, b in predicted['features170']])
        np.testing.assert_array_equal(predicted['emulator_route_per_feature'][selected], 'own_subset_model')
        np.testing.assert_array_equal(predicted['emulator_route_per_feature'][~selected], 'auxiliary_frozen_N170_excluded_target')
        assert selected.sum() == meta['spec']['n_features']


def test_full_support_and_no_grid_prediction_bloat(bundle):
    for case in bundle.manifest['models']:
        grid = bundle.npz('posterior_' + case)
        np.testing.assert_array_equal(grid['q'][[0, -1]], [0, 20])
        np.testing.assert_allclose(grid['kappa'][[0, -1]], [.3, .6], rtol=0, atol=1e-15)
        assert 'prediction_mean' not in grid and 'prediction_variance' not in grid
        for key in grid:
            if key == 'mass' or key.startswith('mass_'):
                assert grid[key].shape == (len(grid['q']), len(grid['kappa']))
                assert np.isfinite(grid[key]).all() and (grid[key] >= 0).all()
                np.testing.assert_allclose(grid[key].sum(), 1, atol=1e-9, rtol=0)
    for case in ('inclusive', 'all'):
        prior = bundle.npz('prior_' + case)
        assert prior['u'][0] == 0 and prior['u'][-1] == 1
        assert len(prior['u']) > len(prior['q'])
        np.testing.assert_allclose(prior['square_mass'].sum(), 1, atol=1e-9)


def test_full_covariances_not_only_marginal_bars(bundle):
    experiment = bundle.npz('experiment')
    for key in ('ell_0', 'ell_0p05', 'ell_0p2', 'ell_0p5', 'ell_inf'):
        cov = experiment['covariance_' + key]
        assert cov.shape == (170, 170)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)
        assert np.count_nonzero(cov - np.diag(cov.diagonal())) > 0
    assert bundle.npz('qa_cv')['observed_covariance'].shape == (39, 138, 138)
    assert bundle.npz('qa_cv')['latent_covariance'].shape == (39, 138, 138)
    archive = bundle.manifest['external_training_archive']
    assert archive['arrays']['covariance']['shape'] == [40, 170, 40, 170]
    assert archive['sha256'] == bundle.manifest['sources'][
        'atlas_rg_inference_20261015/recovery_20261016/precision_continuation/timeout_completion/export/full_simulation_covariance_rg.npz']['sha256']


def test_wake_and_conditional_comparison_completeness(bundle):
    for family in ('hadrons', 'jets', 'substructure', 'all'):
        for endpoint in (0, 1):
            grid = bundle.npz(f'wake_{family}_a{endpoint}')
            assert grid['mass'].shape == (361, 241)
            np.testing.assert_allclose(grid['mass'].sum(), 1, atol=1e-9)
    np.testing.assert_array_equal(bundle.npz('wake_hadrons_a0')['mass'], bundle.npz('wake_hadrons_a1')['mass'])
    fixed = bundle.npz('wake_fixed')
    assert fixed['features'].shape == (36, 3)
    assert fixed['response_covariance'].shape == (36, 36)
    assert fixed['covariance'].shape == (72, 72)
    conditional = bundle.npz('conditional')
    indices = np.concatenate([conditional[d + '_heldout_indices'] for d in
                             ('inclusive_to_substructure', 'substructure_to_inclusive')])
    np.testing.assert_array_equal(np.sort(indices), np.arange(138))


def test_no_unsafe_or_local_runtime_data(bundle):
    for name in bundle.manifest['datasets']:
        if bundle.manifest['datasets'][name]['file'].endswith('.npz'):
            arrays = bundle.npz(name)
            assert all(not a.dtype.hasobject for a in arrays.values())
        else:
            assert '/home/jean/' not in json.dumps(bundle.json(name))
    assert '/home/jean/' not in (bundle.root / 'manifest.json').read_text()
    for path in (EXAMPLE / 'original').glob('*.py'):
        text = path.read_text()
        assert '/home/jean/' not in text
        assert 'import pickle' not in text
        assert 'import queue_models' not in text


@pytest.mark.parametrize('relative', ['/tmp/a', '../outside', 'inside/../../x', 'a\\b'])
def test_confined_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_child(tmp_path, relative)


def test_tampered_bundle_rejected(tmp_path):
    path = tmp_path / 'cache.json'
    path.write_text('{}')
    manifest = dict(schema=SCHEMA, scope='frozen-result/figure replay; no refitting or event-level reproduction',
                    files={'cache.json': dict(sha256='0'*64)}, datasets={})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        Bundle(tmp_path)


def test_self_consistent_but_unreviewed_manifest_rejected(tmp_path):
    payload = tmp_path / 'arbitrary_science.json'
    payload.write_text('{"science_after": [1234], "science_matches_original": true}')
    manifest = dict(schema=SCHEMA, scope='frozen-result/figure replay; no refitting or event-level reproduction',
        files={payload.name: dict(sha256=sha(payload))},
        datasets={'arbitrary': dict(file=payload.name)})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    # Generic safe-cache validation alone permits internally consistent fixtures.
    assert Bundle(tmp_path).json('arbitrary')['science_matches_original'] is True
    # Production reproduction must additionally bind the reviewed tracked manifest.
    with pytest.raises(ValueError, match='shipped tracked reference'):
        Bundle(tmp_path, expected_manifest=EXAMPLE / 'manifest.reference.json')


def test_shipped_reference_is_identical(bundle):
    checked = Bundle(bundle.root, expected_manifest=EXAMPLE / 'manifest.reference.json')
    assert checked.reference_verified


def test_pickle_array_rejected(tmp_path):
    path = tmp_path / 'unsafe.npz'
    np.savez(path, bad=np.array([{}], dtype=object))
    manifest = dict(schema=SCHEMA, scope='frozen-result/figure replay; no refitting or event-level reproduction',
        files={'unsafe.npz': dict(sha256=sha(path))},
        datasets={'unsafe': dict(file='unsafe.npz', arrays={})})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Object arrays'):
        Bundle(tmp_path).npz('unsafe')


def test_source_and_historical_snapshot_hashes():
    original = EXAMPLE / 'original'
    for name, record in json.loads((original / 'PROVENANCE.json').read_text()).items():
        assert sha(original / name) == record['sha256']
    archive = original / 'executed_analysis'
    records = json.loads((archive / 'SHA256.json').read_text())
    assert 'paper_local_checks_20261019/remaining_inference.py' in records
    for relative, record in records.items():
        assert sha(archive / record['snapshot']) == record['sha256']
        assert record['snapshot'].endswith('.py.txt')
    fonts = EXAMPLE / 'fonts'
    for name, digest in json.loads((fonts / 'PROVENANCE.json').read_text())['files'].items():
        assert sha(fonts / name) == digest


def test_native_prediction_states(bundle):
    # Imports canonical repo API only; no historical source or model pickle loads.
    from bayesian.emulation.review_state import load_prediction_state
    assert set(bundle.manifest['native_prediction_state_cases']) == {'all', 'inclusive', 'jets'}
    for case in bundle.manifest['native_prediction_state_cases']:
        path = bundle.root / bundle.manifest['datasets']['state_' + case]['file']
        model, features, designs, metadata = load_prediction_state(path)
        reference = bundle.npz('state_' + case + '_reference')
        actual = model.predict(reference['points_unit'])
        assert metadata['provenance']['fit_id'] == bundle.json('model_' + case)['identity']['fit_id']
        np.testing.assert_array_equal(actual['central_value'], reference['central_value'])
        for name in actual:
            np.testing.assert_allclose(actual[name], reference[name], rtol=5e-12, atol=5e-13)


@pytest.mark.skipif(os.environ.get('HYBRID_LRES_RENDER_TEST') != '1', reason='Explicit finite 18-figure integration run')
def test_render_from_unrelated_cwd_and_refuse_overwrite(tmp_path):
    if not (EXAMPLE / 'numerical_cache/manifest.json').exists():
        pytest.fail('Explicit render validation requires the separate private numerical_cache artifact')
    output = tmp_path / 'rendered'
    command = [sys.executable, str(EXAMPLE / 'reproduce.py'), '--input', str(EXAMPLE / 'numerical_cache'),
               '--output', str(output), '--manuscript-only']
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=300)
    receipt = json.loads((output / 'REPRODUCED.json').read_text())
    assert len(receipt['figures']) == 18
    for stem, row in receipt['figures'].items():
        assert row['science_matches_original']
        for suffix in ('.pdf', '.png', '.svg'):
            assert (output / (stem + suffix)).is_file()
    import pymupdf
    for stem in receipt['manuscript_figures']:
        with pymupdf.open(output / (stem + '.pdf')) as doc:
            assert len(doc) == 1
            assert abs(doc[0].rect.width - 5.95276*72) < .002
            spans = [span for block in doc[0].get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for span in line['spans']]
            roman = [s for s in spans if s['font'] == 'LMRoman10-Regular']
            assert roman and not any('DejaVu' in s['font'] for s in spans)
            assert all(min(abs(s['size'] - 10.90909), abs(s['size'] - 7.63636)) < .005 for s in roman)
    with pymupdf.open(output / 'indexed_figures.pdf') as doc:
        assert len(doc) == len(doc.get_toc()) == 18
    rerun = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert rerun.returncode != 0 and 'Refusing to overwrite' in rerun.stderr
