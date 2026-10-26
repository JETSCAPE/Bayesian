"""Author export of frozen numerical caches. Sources are read-only; never fit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np

from cache_io import SCHEMA, array_digest, sha

RAW = ('atlas_rg_inference_20261015/recovery_20261016/precision_continuation/'
       'timeout_completion/export/full_simulation_covariance_rg.npz')
QUEUE = 'paper_local_checks_20261019/queue'


class Export:
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.files, self.datasets, self.sources = {}, {}, {}

    def pin(self, relative):
        path = self.source / relative
        self.sources[relative] = dict(sha256=sha(path), bytes=path.stat().st_size)
        return path

    def read_json(self, relative):
        return json.loads(self.pin(relative).read_text())

    def read_npz(self, relative, names=None):
        with np.load(self.pin(relative), allow_pickle=False) as archive:
            keys = archive.files if names is None else names
            return {key: archive[key].copy() for key in keys}

    def clean(self, value):
        if isinstance(value, dict):
            return {self.clean(k): self.clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            return value.replace(str(self.source) + '/', 'author-source:')
        return value

    def put_json(self, name, value):
        path = self.output / (name + '.json')
        path.write_text(json.dumps(self.clean(value), indent=2, allow_nan=False) + '\n')
        self.files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        self.datasets[name] = dict(file=path.name)

    def put_npz(self, name, arrays):
        if any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError('Object arrays are forbidden')
        path = self.output / (name + '.npz')
        np.savez_compressed(path, **arrays)
        self.files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        self.datasets[name] = dict(file=path.name,
                                  arrays={k: array_digest(v) for k, v in arrays.items()})

    def checked(self, relative, receipt):
        path = self.pin(relative)
        outputs = receipt.get('output_sha256', receipt.get('outputs', {}))
        found = [v.get('sha256') if isinstance(v, dict) else v for k, v in outputs.items()
                 if k == str(path) or k == path.name]
        if found != [self.sources[relative]['sha256']]:
            raise ValueError('Missing/mismatched frozen completion hash: ' + relative)


def export(source, output, training_archive, prediction_states=None):
    out = Export(source, output)
    manifest = out.read_json(QUEUE + '/MANIFEST.json')
    model_meta = {}
    for case, spec in manifest['models'].items():
        folder = QUEUE + '/results/' + case + '/inference/'
        complete = out.read_json(folder + 'COMPLETE.json')
        identity = complete['fit_source_identity']
        declared = out.read_json(folder + 'INPUTS.json')
        summary = out.read_json(folder + 'summary.json')
        if identity != declared['model_identity'] or identity != summary['fit_source_identity']:
            raise ValueError('Source/fit identity mismatch: ' + case)
        for name in ('grid.npz', 'predictive.npz', 'summary.json', 'INPUTS.json'):
            out.checked(folder + name, complete)
        with np.load(source / (folder + 'grid.npz'), allow_pickle=False) as z:
            keys = [k for k in z.files if k in ('q', 'kappa', 'mass', 'features', 'input_identity', 'fit_id')
                    or k.startswith('mass_')]
            grid = {k: z[k].copy() for k in keys}
        if str(grid['fit_id']) != identity['fit_id']:
            raise ValueError('Wrong posterior fit ID')
        out.put_npz('posterior_' + case, grid)
        prediction = out.read_npz(folder + 'predictive.npz')
        out.put_npz('predictive_' + case, prediction)
        qa = out.read_json(QUEUE + '/results/' + case + '/QA/COMPLETE.json')
        if qa.get('fit_source_identity', qa.get('model_identity')) != identity:
            raise ValueError('Wrong native QA identity')
        # Full identity once, not duplicated inside every ell summary.
        summaries = {key: {k: v for k, v in row.items() if k != 'fit_source_identity'}
                     for key, row in summary['cases'].items()}
        meta = dict(spec={k: spec[k] for k in ('features', 'n_features', 'lengths', 'prior_scan') if k in spec},
                    identity=identity, input_identity=complete['input_identity'], cases=summaries,
                    predictive_precision=summary['predictive_precision'],
                    remaining_cautions=summary.get('remaining_cautions', []), native_QA=qa)
        if spec.get('prior_scan'):
            for name in ('prior.npz', 'prior_summary.json'):
                out.checked(folder + name, complete)
            out.put_npz('prior_' + case, out.read_npz(folder + 'prior.npz',
                ['q', 'kappa', 'u', 'square_mass', 'physical_mass', 'square_same_support_mass']))
            out.put_json('prior_summary_' + case, out.read_json(folder + 'prior_summary.json'))
        out.put_json('model_' + case, meta)
        model_meta[case] = 'model_' + case

    covariance = out.read_npz(QUEUE + '/covariances.npz')
    receipt = out.read_json(QUEUE + '/COVARIANCES.json')
    if receipt['output_sha256'] != out.sources[QUEUE + '/covariances.npz']['sha256']:
        raise ValueError('Experimental covariance binding mismatch')
    out.put_npz('experiment', covariance)
    out.put_json('experimental_covariance_identity', receipt)
    common = out.read_npz(QUEUE + '/consistency/common_all138/conditional_results.npz',
        ['features', 'observed', 'experimental_covariance'] + [d + '_' + f
        for d in ('inclusive_to_substructure', 'substructure_to_inclusive')
        for f in ('heldout_indices', 'measurement_quantiles', 'predictive_covariance')])
    out.put_npz('conditional', common)
    out.put_json('conditional_identity', out.read_json(QUEUE + '/consistency/COMPLETE.json'))

    for family in ('hadrons', 'jets', 'substructure', 'all'):
        for endpoint in (0, 1):
            key = f'{family}_a{endpoint}'
            folder = f'wake_subset_training_20261018/{key}/inference/'
            done = out.read_json(folder + 'COMPLETE.json')
            grid = out.read_npz(folder + 'grid.npz', ['q', 'kappa', 'mass'])
            if done['grid_sha256'] != out.sources[folder + 'grid.npz']['sha256'] or not done['training_only_selected_bins']:
                raise ValueError('Wake subset provenance mismatch')
            out.put_npz('wake_' + key, grid)
            out.put_json('wake_identity_' + key, dict(completion=done,
                summary=out.read_json(folder + 'SUMMARY.json')))
    profile = 'wake_bounded_transport_20261017/validated_local/'
    out.put_npz('wake_profile', out.read_npz(profile + 'comparison.npz', ['profile_moments']))
    out.put_json('wake_profile_report', out.read_json(profile + 'REPORT.json'))
    out.put_json('wake_method', dict(text=out.pin('wake_bounded_transport_20261017/METHOD.txt').read_text()))
    fixed = 'wake_individual_substructure_20261019/fixed_design_31/'
    out.put_npz('wake_fixed', out.read_npz(fixed + 'comparison.npz'))
    out.put_json('wake_fixed_point', out.read_json(fixed + 'NUMERICAL.json'))

    qa_root = QUEUE + '/results/all/QA/'
    cv = out.read_npz(qa_root + 'fivefold_predictions.npz',
        ['design_ids', 'design_unit', 'truth', 'observed_mean', 'observed_covariance', 'latent_covariance'])
    hold = out.read_npz(qa_root + 'independent_validation.npz',
        ['design_ids', 'design_unit', 'truth', 'observed_mean', 'observed_covariance', 'latent_covariance'])
    out.put_npz('qa_cv', cv)
    out.put_npz('qa_holdout', hold)
    out.put_json('qa_audit', out.read_json(qa_root + 'optimizer_and_PC_audit.json'))
    out.put_json('qa_validation_summary', out.read_json(qa_root + 'validation_summary.json'))
    closure = 'BayesianLres-draft-qa/appendix_qa/'
    suffix = '_181x121' if (source / (closure + 'CLOSURE_181x121.json')).exists() else ''
    out.put_npz('qa_closure', out.read_npz(closure + 'closure_grid' + suffix + '.npz'))
    out.put_json('qa_closure_report', out.read_json(closure + 'CLOSURE' + suffix + '.json'))

    raw = out.read_npz(RAW)
    own = [(str(n), float(a), float(b)) for n, a, b in manifest['models']['all']['features']]
    raw_features = list(zip(raw['observable_names'].tolist(), raw['xmin'], raw['xmax']))
    ri = [raw_features.index(f) for f in own]
    np.testing.assert_allclose(cv['truth'], raw['central_values'][cv['design_ids']][:, ri], rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(hold['truth'], raw['central_values'][hold['design_ids']][:, ri], rtol=2e-6, atol=1e-12)
    variance = np.stack([raw['covariance'][int(d), ri, int(d)][:, ri].diagonal() for d in cv['design_ids']])
    out.put_npz('qa_mc', dict(mc_variance=variance))
    training_archive = training_archive.resolve()
    training_archive.parent.mkdir(parents=True, exist_ok=True)
    if training_archive.exists():
        if sha(training_archive) != out.sources[RAW]['sha256']:
            raise FileExistsError('Existing training archive differs; preserve it')
        print('REUSED_HASH_IDENTICAL_TRAINING_ARCHIVE', flush=True)
    else:
        shutil.copyfile(source / RAW, training_archive)
    training = dict(filename=training_archive.name, sha256=sha(training_archive), bytes=training_archive.stat().st_size,
                    arrays={k: array_digest(a) for k, a in raw.items()},
                    scope='Full moment-derived central_values and cross-design/cross-bin covariance; NOT the accepted pinned old-table means. Separate audit archive, not required to draw figures.')
    del raw
    accepted_relative = ('atlas_rg_inference_20261015/recovery_20261016/precision_continuation/'
        'timeout_completion/local_inference/results/main/analysis_raa_unit_Lres-E-loss/observables.h5')
    accepted_source = out.pin(accepted_relative)
    accepted_target = training_archive.parent / 'accepted_observables.h5'
    if accepted_target.exists():
        if sha(accepted_target) != out.sources[accepted_relative]['sha256']:
            raise FileExistsError('Existing accepted-table archive differs; preserve it')
    else:
        shutil.copyfile(accepted_source, accepted_target)
    accepted = dict(filename=accepted_target.name, sha256=sha(accepted_target), bytes=accepted_target.stat().st_size,
        scope='Bytewise accepted frozen observables.h5: Design/Design_indices/Prediction, validation tables, and experimental inputs. Accepted old 134-bin prediction means are not replaced by moment-derived central_values in full_training.npz.')

    tex = out.pin('BayesianLres/arxiv-v1.tex').read_text()
    figures = re.findall(r'\\includegraphics\[[^]]+\]\{(preliminary_[^}]+|qa_[^}]+)\.pdf\}', tex)
    reference = out.read_json('BayesianLres-draft-qa/jhep_design/RENDERED.json')
    if len(figures) != 18 or set(figures) != set(reference['figures']):
        raise ValueError('Manuscript figure membership differs from reviewed 18')
    references = {}
    for stem in figures:
        row = reference['figures'][stem]
        out.pin('BayesianLres/figure/' + stem + '.pdf')
        references[stem] = dict(science=row['science_after'], display_only=row.get('display_only'),
                                height_inches=row['height_inches'], width_inches=row['width_inches'])
    out.put_json('manuscript_reference', references)
    state_cases = []
    if prediction_states is not None:
        state_report = json.loads((prediction_states / 'EXPORT.json').read_text())
        for case in ('all', 'inclusive', 'jets'):
            path = prediction_states / (case + '.npz')
            if sha(path) != state_report[case]['prediction_state_sha256']:
                raise ValueError('Native checkpoint SHA mismatch: ' + case)
            for suffix in ('', '_reference'):
                path = prediction_states / (case + suffix + '.npz')
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in z.files}
                out.put_npz('state_' + case + suffix, arrays)
                out.sources['native_prediction_states/' + path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
            state_cases.append(case)
        out.put_json('native_prediction_state_export', state_report)
    record = dict(schema=SCHEMA,
        scope='frozen-result/figure replay; no refitting or event-level reproduction',
        no_pickle=True, no_fits=True, no_inference=True, personal_paths_at_runtime=False,
        files=out.files, datasets=out.datasets, sources=out.sources,
        models=model_meta, manuscript_figures=figures, external_training_archive=training,
        external_accepted_tables=accepted,
        native_prediction_state_cases=state_cases,
        source_checkout_anchors={
            'native_repository_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
            'manuscript_repository': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source / 'BayesianLres', text=True).strip(),
            'author_analysis_sources': 'Exact SHA256 snapshots authoritative; not necessarily tracked in either commit'},
        coordinate_convention='q = physical pi*T*Lres on [0,20]; u is unit-square resolution input, not q; kappa in [0.3,0.6]',
        omissions=['No grid-point prediction_mean/prediction_variance or loglike arrays: full posterior masses and saved predictive quantiles retained.',
                   'No event records, optimizer reruns, emulator loads or inference. Native model checkpoints are a separate validation task.'])
    (out.output / 'manifest.json').write_text(json.dumps(out.clean(record), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(bundle=str(out.output), bytes=sum(r['bytes'] for r in out.files.values()),
                          training_archive_bytes=training['bytes'], manuscript_figures=len(figures))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Author workspace only; never used at reproduction runtime')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-archive', type=Path, required=True, help='Separate full covariance archive, not figure-cache bloat')
    parser.add_argument('--prediction-states', type=Path, help='Optional completed native numerical prediction-state export')
    args = parser.parse_args()
    export(args.source_root.resolve(), args.output, args.training_archive, args.prediction_states)
