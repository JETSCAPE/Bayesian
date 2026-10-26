"""Regenerate frozen-result figures from safe, portable numeric caches only."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace as NS
from unittest.mock import patch

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'original'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf

import hybrid_plotting as hp
import composition
import core
import geometry
import layout
import plots
import qa
import slides
import typography
from cache_io import Bundle, array_digest, sha

WIDTH, BODY_PT = 5.95276, 10.90909


def validate_result(row):
    plots.validate_grid(row.grid)
    for key in row.spec['lengths']:
        plots.validate_grid(row.grid, 'mass_' + key)
    expected = plots.features(row.spec['features'])
    if expected != plots.features(row.grid['features']) or expected != plots.features(row.identity['feature_membership']):
        raise ValueError('Training/likelihood feature identities differ: ' + row.case)
    if str(row.grid['fit_id']) != row.identity['fit_id']:
        raise ValueError('Fit identity differs: ' + row.case)
    ff = plots.features(row.predictive['features170'])
    quantiles = row.predictive['quantiles']
    if len(ff) != 170 or quantiles.shape != (5, 170) or not np.isfinite(quantiles).all() or np.any(np.diff(quantiles, axis=0) < 0):
        raise ValueError('Incomplete/malformed predictive gallery: ' + row.case)
    if not np.array_equal(row.predictive['probabilities'], [.025, .16, .5, .84, .975]):
        raise ValueError('Unknown predictive probabilities')
    for f, route in zip(ff, row.predictive['emulator_route_per_feature'], strict=True):
        wanted = 'own_subset_model' if f in set(expected) else 'auxiliary_frozen_N170_excluded_target'
        if route != wanted:
            raise ValueError('Own-fit/auxiliary label mismatch: ' + row.case)
    if row.prior is not None:
        for key in ('physical_mass', 'square_same_support_mass'):
            plots.validate_grid(dict(q=row.prior['q'], kappa=row.prior['kappa'], mass=row.prior[key]))
        np.testing.assert_array_equal(row.prior['physical_mass'], row.grid['mass'])
        square = row.prior['square_mass']
        if square.shape != (len(row.prior['u']), len(row.prior['kappa'])) or not np.isclose(square.sum(), 1):
            raise ValueError('Incomplete full-square prior mass')


def load_results(bundle):
    rows = {}
    for case, name in bundle.manifest['models'].items():
        meta = bundle.json(name)
        row = NS(case=case, spec=meta['spec'], identity=meta['identity'],
            summary=dict(cases=meta['cases'], predictive_precision=meta['predictive_precision']),
            grid=bundle.npz('posterior_' + case), predictive=bundle.npz('predictive_' + case),
            qa=meta['native_QA'], prior=None, prior_summary=None)
        row.nominal = row.summary['cases']['ell_0p2']
        if row.spec.get('prior_scan'):
            row.prior = bundle.npz('prior_' + case)
            row.prior_summary = bundle.json('prior_summary_' + case)
        validate_result(row)
        rows[case] = row
    return rows


def verify_states(bundle):
    """Use the repository API, never duplicate native GP mathematics here."""
    if not bundle.reference_verified:
        raise ValueError('Native state verification requires the shipped tracked reference manifest')
    from bayesian.emulation.review_state import load_prediction_state
    report = {}
    cases = bundle.manifest.get('native_prediction_state_cases', [])
    if set(cases) != {'all', 'inclusive', 'jets'}:
        raise ValueError('All/inclusive/jets frozen native states are required for this check')
    for case in cases:
        dataset = bundle.manifest['datasets']['state_' + case]
        bundle.npz('state_' + case)
        model, features, design_ids, metadata = load_prediction_state(bundle.root / dataset['file'])
        identity = bundle.json('model_' + case)['identity']
        if metadata['provenance']['fit_id'] != identity['fit_id']:
            raise ValueError('State and posterior fit identities differ: ' + case)
        if plots.features(features) != plots.features(identity['feature_membership']):
            raise ValueError('State and posterior feature identities differ: ' + case)
        reference = bundle.npz('state_' + case + '_reference')
        actual = model.predict(reference['points_unit'])
        differences = {}
        for key, value in actual.items():
            np.testing.assert_allclose(value, reference[key], rtol=5e-12, atol=5e-13)
            differences[key] = float(np.max(np.abs(value - reference[key])))
        report[case] = dict(fit_id=identity['fit_id'], points=len(reference['points_unit']),
            mean_bitwise_equal=np.array_equal(actual['central_value'], reference['central_value']),
            max_absolute_difference=differences, rtol=5e-12, atol=5e-13)
        np.testing.assert_array_equal(actual['central_value'], reference['central_value'])
    return report


def closure_figure(bundle):
    data, report = bundle.npz('qa_closure'), bundle.json('qa_closure_report')
    if not report['convergence_pass']:
        raise ValueError('Frozen closure quadrature failed its convergence gate')
    q, k, mass = (data[key] for key in ('q', 'kappa', 'mass'))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.6), layout='constrained')
    for ax, x, marginal, truth, xlabel in (
        (axes[0], q, mass.sum(axis=1)/geometry.weights(q), report['true_physical'][0], hp.LRES_LABEL),
        (axes[1], k, mass.sum(axis=0)/geometry.weights(k), report['true_physical'][1], hp.KAPPA_LABEL)):
        ax.plot(x, marginal, color=hp.DEFAULT_COLOR, lw=2, label='Recovered posterior')
        ax.fill_between(x, 0, marginal, color=hp.DEFAULT_COLOR, alpha=.15)
        ax.axvline(truth, color=hp.comparison_style('vermillion')['color'], ls='--', lw=1.5, label='Generating parameter')
        ax.set(xlim=(x[0], x[-1]), ylim=(0, None), xlabel=xlabel, ylabel='Posterior density')
        hp.frame_axes(ax, minor=False)
    axes[1].legend(loc='upper left', frameon=False, fontsize=8)
    hp.save_figure(fig, Path('qa_parameter_recovery'))
    plt.close(fig)


def configure(bundle, rows, output, records):
    reference = bundle.json('manuscript_reference')
    experiment = bundle.npz('experiment')
    ff = plots.features(experiment['features'])
    selected = [ff.index(f) for f in plots.features(rows['all'].spec['features'])]
    cv, hold = bundle.npz('qa_cv'), bundle.npz('qa_holdout')
    own = plots.features(rows['all'].spec['features'])
    groups = [np.array([i for i, r in enumerate(own) if tag in r[0]]) for _, _, tag in qa.GROUPS]
    if sorted(np.concatenate(groups).tolist()) != list(range(138)) or cv['truth'].shape != (39, 138):
        raise ValueError('Native QA group coverage differs')
    qa.inputs = lambda: ({}, own, bundle.npz('qa_mc'), None, cv, hold, groups,
                         experiment['covariance_ell_0p2'][np.ix_(selected, selected)])
    qa.AUDIT = bundle.json('qa_audit')
    qa.ROOT = output
    qa.save = lambda *args: None
    qa.sha = sha
    qa.render_closure = lambda: closure_figure(bundle)
    typography.WIDTH, typography.TEXT_PT = WIDTH, BODY_PT
    typography.plots, typography.qa, typography.core, typography.slides = plots, qa, core, slides
    typography.finish = composition.finish
    core.plots = plots
    core.v2_helpers = lambda: nullcontext(layout)
    layout.base = NS(geometry=geometry)
    plots.geometry = lambda: geometry
    def contour(ax, row, role, mass=None):
        grid = row.grid if hasattr(row, 'grid') else row
        geometry.contour(ax, grid['mass'] if mass is None else mass, grid['q'], grid['kappa'],
                         {k: v for k, v in hp.comparison_style(role).items() if k in ('color', 'linestyle')})
    def endpoint(key):
        grid = bundle.npz('wake_' + key)
        plots.validate_grid(grid)
        return grid, bundle.json('wake_identity_' + key)['summary']
    def parameter_axis(ax, ylabel=True):
        ax.set(xscale='log', xlim=(.05, 20), ylim=(.3, .6), xlabel=hp.LRES_LABEL)
        if ylabel:
            ax.set_ylabel(hp.KAPPA_LABEL)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([.05, .2, 1, 5, 20]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        hp.frame_axes(ax, minor=False)
    old = NS(result=rows.__getitem__, endpoint=endpoint, contour=contour,
             parameter_axis=parameter_axis, geo=geometry)
    typography.old = old
    slides.old, slides.geo, slides.INPUT = old, geometry, bundle
    slides.WAKE, slides.Q_RANGE, slides.K_RANGE = {}, (.05, 20), (.3, .6)
    slides.INK, slides.GREY = '#202326', '#62666A'
    slides.save = lambda *args, **kwargs: None
    composition.base = typography
    composition.WIDTH, composition.PT = WIDTH, BODY_PT
    composition.mpl = matplotlib
    composition.DATA, composition.GALLERY = {}, {}
    typography.v2_helpers = core.v2_helpers

    def save(fig, stem, before, extra=None):
        composition.style(fig)
        after = typography.signature(fig)
        if before != after:
            raise ValueError('Drawing values changed during styling: ' + stem)
        expected = reference[stem]['science']
        after = json.loads(json.dumps(after))
        # Compare every line, band, collection, patch, scale and view range.
        if after != expected:
            differences = [i for i in range(max(len(after), len(expected)))
                           if i >= len(after) or i >= len(expected) or after[i] != expected[i]]
            raise ValueError(f'Original scientific-value comparison failed: {stem}, axes {differences}')
        paths = hp.save_figure(fig, output / stem)
        records[stem] = dict(kind='manuscript', science_matches_original=True, science=after,
            width_inches=WIDTH, height_inches=float(fig.get_figheight()), body_bp=BODY_PT,
            files={p.name: sha(p) for p in paths}, display_only=extra)
        plt.close(fig)
        print('VERIFIED', stem, flush=True)
    composition.save = save
    return experiment


def galleries_and_ell(rows, experiment):
    row = rows['all']
    y, error = experiment['y'], np.sqrt(np.diag(experiment['covariance_ell_0p2']))
    composition.DATA.update(row=row, y=y, error=error)
    captured = {}
    book = NS(checks={}, save=lambda fig, stem, *args: captured.update({stem: fig}))
    plots.galleries(book, {'all': row}, y, error)
    for j, fig in enumerate(captured.values(), 1):
        composition.finish(fig, f'preliminary_observables_{j:02}', typography.signature(fig))
    class FirstOnly(Exception):
        pass
    def first(fig, stem, *args):
        captured.clear()
        captured[stem] = fig
        raise FirstOnly
    try:
        plots.ell_figures(NS(save=first), rows, geometry)
    except FirstOnly:
        pass
    fig = captured['ell_parameter_scan']
    composition.finish(fig, 'preliminary_correlation_scan', typography.signature(fig))


def cross_predictions(bundle):
    data = bundle.npz('conditional')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.25))
    color = hp.DEFAULT_COLOR
    for j, (direction, tag) in enumerate((('inclusive_to_substructure', '__dr12_'),
                                        ('substructure_to_inclusive', '__pt_cms__'))):
        held = data[direction + '_heldout_indices']
        ff = data['features'][held]
        names = list(dict.fromkeys(ff[:, 0]))
        name = next(n for n in names if tag in n and (j == 0 or '__R0.4__' in n))
        quantiles = data[direction + '_measurement_quantiles']
        ax = axes[j]
        for i, (n, low, high) in enumerate(plots.features(ff)):
            if n == name:
                ax.fill_between([low, high], [quantiles[0, i]]*2, [quantiles[4, i]]*2, color=color, alpha=.10)
        row = NS(case='all', spec=dict(features=ff), predictive=dict(features170=ff, quantiles=quantiles))
        plots.draw_observable(ax, name, row, data['observed'][held],
                              np.sqrt(np.diag(data['experimental_covariance']))[held])
        ax.set(title='', ylabel='Measured ratio')
        if j == 0:
            ax.set_xscale('symlog', linthresh=.003)
            ax.set_xlim(0, 1)
    before = typography.signature(fig)
    composition.panels(fig, 3.25, top=.78, bottom=.19)
    axes[0].set_title(r'ATLAS $dR_{12}$', pad=6)
    axes[1].set_title(r'CMS jets, $R=0.4$', pad=6)
    from matplotlib.patches import Patch
    composition.key(fig, [Patch(facecolor=color, alpha=.10), Patch(facecolor=color, alpha=.22),
        Line2D([], [], color='black', marker='o', lw=0)], ['95% interval', '68% interval', 'Data'], ncol=3)
    for ax in axes:
        ax.set_ylabel('')
        ax.xaxis.set_label_coords(.5, -.20)
    fig.supylabel('Measured ratio', x=.015, y=.48)
    axes[0].set_xticks([0, .01, .1, 1])
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    composition.save(fig, 'preliminary_cross_predictions', before)


class ExtraFigures:
    def __init__(self, output, records):
        self.output, self.records = output, records
        self.checks = {}

    def save(self, fig, stem, title, cases, note):
        # Native extra panels were slide-sized; reflow only the display to body size.
        before = typography.signature(fig)
        labels, handles = [], []
        for legend in fig.legends:
            for handle, text in zip(legend.legend_handles, legend.texts):
                if text.get_text() not in labels:
                    handles.append(handle)
                    labels.append(text.get_text())
        typography.clear_figure_text(fig)
        n = len(fig.axes)
        height = 4.65 if n <= 2 else max(3.8, fig.get_figheight() * WIDTH / fig.get_figwidth())
        fig.set_size_inches(WIDTH, height)
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        # Preserve existing labels and handles in an external key, not data artists.
        for ax in fig.axes:
            hh, ll = ax.get_legend_handles_labels()
            for h, label in zip(hh, ll):
                if label and not label.startswith('_') and label not in labels:
                    handles.append(h)
                    labels.append(label)
        if not handles and cases:
            handles = [plots.handle(plots.label(c), plots.style(c)) for c in cases]
            labels = [h.get_label() for h in handles]
        top = .70 if len(handles) > 4 else .76 if len(handles) > 2 else .84
        if n == 2:
            for ax, box in zip(fig.axes, ([.14, .28, .34, top-.28], [.65, .28, .32, top-.28])):
                ax.set_position(box)
        elif n == 1:
            fig.axes[0].set_position([.14, .28, .82, top-.28])
        else:
            fig.subplots_adjust(left=.13, right=.97, bottom=.20, top=top, hspace=.7, wspace=.5)
        if handles:
            composition.key(fig, handles, labels, ncol=2 if len(handles) > 2 else len(handles))
        has_low_cut = any(case.startswith('hpt10_') for case in cases)
        has_default = any(not case.startswith('hpt10_') for case in cases)
        cut = '30 and 10 GeV (compared)' if has_low_cut and has_default else '10 GeV' if has_low_cut else '30 GeV'
        annotation = ['Hadron threshold: ' + cut,
                      textwrap.fill('Models: ' + ', '.join(cases or [stem]), width=70)]
        if stem.startswith('observables_'):
            annotation.append('Own fit: solid/filled; auxiliary: dashed/hollow')
        annotation = '\n'.join(annotation)
        fig.text(.04, .012, annotation, ha='left', va='bottom', fontsize=BODY_PT,
                 color='#44484B', linespacing=1.15)
        composition.style(fig)
        for ax in fig.axes:
            if n == 2:
                ax.tick_params(labelbottom=True)
                if not ax.get_xlabel() and ax.get_xscale() in ('log', 'symlog') and ax.get_xlim()[1] == 20:
                    ax.set_xlabel(hp.LRES_LABEL)
        if typography.signature(fig) != before:
            raise ValueError('Extra figure scientific values changed during reflow: ' + stem)
        # Long observable titles are wrapped, without deleting physical bin labels.
        for ax in fig.axes:
            if len(ax.get_title()) > 28:
                ax.set_title(ax.get_title().replace(', ', ',\n', 1))
        paths = hp.save_figure(fig, self.output / stem)
        self.records[stem] = dict(kind='extra', title=title, cases=list(cases), note=note,
            annotation=annotation, science=typography.signature(fig), files={p.name: sha(p) for p in paths})
        plt.close(fig)
        print('EXTRA', stem, flush=True)


def extras(bundle, rows, experiment, output, records):
    book = ExtraFigures(output, records)
    for stem, cases, title in (
        ('sector_comparison', ('hadrons', 'jets', 'inclusive', 'substructure', 'all'), 'Sector comparisons'),
        ('angular_comparison', ('theta_g', 'dR12', 'rg', 'substructure'), 'Angular comparisons'),
        ('pair_combinations', ('inclusive', 'inclusive_plus_theta_g_plus_dR12', 'inclusive_plus_theta_g_plus_rg',
                               'inclusive_plus_dR12_plus_rg', 'all'), 'Angular-pair additions')):
        plots.posterior_overlay(book, rows, cases, stem, title, geometry)
    for family in ('inclusive', 'all'):
        cases = [family] + [family + '_without_' + e for e in ('alice', 'atlas', 'cms')]
        plots.posterior_overlay(book, rows, cases, 'leaveouts_' + family, 'Experiment leave-outs', geometry)
        plots.posterior_overlay(book, rows, (family, 'hpt10_' + family), 'hadron_cut_' + family,
            'Hadron cut sensitivity', geometry,
            roles={family: hp.comparison_style('default'), 'hpt10_' + family: hp.comparison_style('vermillion')})
    plots.posterior_overlay(book, rows, ('all', 'all_without_first_rg', 'all_without_first_dR12',
                            'all_without_first_rg_and_dR12'), 'first_bin_cuts', 'Angular first-bin cuts', geometry)
    plots.ell_figures(book, rows, geometry)
    plots.prior_figures(book, rows, layout)
    # Complete own-source galleries, including every auxiliary bin for all 25 fits.
    plots.galleries(book, rows, experiment['y'], np.sqrt(np.diag(experiment['covariance_ell_0p2'])))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.8))
    for endpoint, role in ((0, 'default'), (1, 'vermillion')):
        grid = bundle.npz('wake_all_a' + str(endpoint))
        typography.old.contour(axes[0], grid, role)
        axes[1].plot(grid['q'], grid['mass'].sum(1)/geometry.weights(grid['q']),
                     label='Original wake' if endpoint == 0 else 'Ad hoc reshaped wake', **slides.style(role))
    typography.old.parameter_axis(axes[0])
    axes[1].set(xscale='log', xlim=(.05, 20), xlabel=hp.LRES_LABEL, ylabel='Posterior density')
    book.save(fig, 'wake_resolution', 'Matched wake-bank resolution', [],
              '30-design matched sensitivity bank, not the nominal 39-design fit.')
    return book.checks


def reproduce(input_bundle, output, include_extras=True, training_archive=None, verify_state=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite or silently rerun an existing output directory')
    bundle = Bundle(input_bundle, expected_manifest=HERE / 'manifest.reference.json')
    if training_archive is not None:
        bundle.verify_external(training_archive)
    states = verify_states(bundle) if verify_state else {}
    provenance = json.loads((HERE / 'original/PROVENANCE.json').read_text())
    for name, record in provenance.items():
        if sha(HERE / 'original' / name) != record['sha256']:
            raise ValueError('Reviewed plotting helper changed: ' + name)
    for path in (HERE / 'fonts').glob('*.otf'):
        font_manager.fontManager.addfont(path)
    if not (HERE / 'fonts/lmroman10-regular.otf').exists():
        raise FileNotFoundError('Vendored licensed Latin Modern fonts are required')
    rows = load_results(bundle)
    output.mkdir(parents=True)
    records = {}
    experiment = configure(bundle, rows, output, records)
    with hp.paper_style(), matplotlib.rc_context({'font.family': 'Latin Modern Roman',
        'font.serif': ['Latin Modern Roman'], 'font.size': BODY_PT, 'axes.unicode_minus': False,
        'mathtext.fontset': 'cm'}):
        typography.core_plot()
        typography.family_plots()
        typography.experiment_plot()
        galleries_and_ell(rows, experiment)
        cross_predictions(bundle)
        typography.wake_plots()
        typography.qa_plots()
        checks = extras(bundle, rows, experiment, output, records) if include_extras else {}
    missing = set(bundle.manifest['manuscript_figures']) - records.keys()
    if missing:
        raise ValueError('Missing manuscript figures: ' + str(sorted(missing)))
    order = bundle.manifest['manuscript_figures'] + sorted(set(records) - set(bundle.manifest['manuscript_figures']))
    with pymupdf.open() as merged:
        toc = []
        for stem in order:
            toc.append([1, stem, len(merged) + 1])
            with pymupdf.open(output / (stem + '.pdf')) as source:
                merged.insert_pdf(source)
        merged.set_toc(toc)
        merged.set_metadata(dict(title='Frozen numerical-cache reproduction; no refitting'))
        merged.save(output / 'indexed_figures.pdf')
    result = dict(scope=bundle.manifest['scope'], bundle_manifest_sha256=sha(bundle.root / 'manifest.json'),
                  figures=records, checks=checks, manuscript_figures=bundle.manifest['manuscript_figures'],
                  full_training_archive_verified=training_archive is not None,
                  native_prediction_state_verification=states,
                  indexed_pdf_sha256=sha(output / 'indexed_figures.pdf'))
    (output / 'REPRODUCED.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Portable numerical-cache directory')
    parser.add_argument('--output', type=Path, help='New output directory; existing paths are refused')
    parser.add_argument('--manuscript-only', action='store_true')
    parser.add_argument('--training-archive', type=Path, help='Optionally verify the separate full covariance archive')
    parser.add_argument('--verify-state', action='store_true', help='Check frozen states through native repo FastMeanAware')
    parser.add_argument('--verify-state-only', action='store_true', help='Only check native states; do not draw figures')
    args = parser.parse_args()
    if args.verify_state_only:
        print(json.dumps(verify_states(Bundle(args.input, expected_manifest=HERE / 'manifest.reference.json')), indent=2))
    else:
        if args.output is None:
            parser.error('--output is required unless --verify-state-only is selected')
        result = reproduce(args.input, args.output, not args.manuscript_only, args.training_archive, args.verify_state)
        print(json.dumps(dict(figures=len(result['figures']), scope=result['scope'])))
