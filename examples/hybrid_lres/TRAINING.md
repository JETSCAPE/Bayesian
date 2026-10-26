# Refit The Emulator And Rerun Inference

This is the training/inference entry point, **not** the frozen-figure replay.
It starts from accepted simulation tables and measured simulation covariance;
no reference emulator or posterior is read to produce the results.

## Setup

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv-review
source .venv-review/bin/activate
python -m pip install -r examples/hybrid_lres/requirements.lock.txt
```

Obtain the single private `hybrid_lres_review_bundle.tar.xz` from the authors.
It contains all non-code inputs for fresh training/inference and frozen-figure
replay: accepted tables, experimental source errors/metadata, physical grid axes,
integrity hashes, the full measured simulation covariance, and the numerical
figure/QA/wake reference caches. The covariance retains all cross-design,
cross-bin, cross-observable and common-pp correlations, with no precision loss.

Extract **from the repository root**. This creates `inputs/` for fresh training
and `examples/hybrid_lres/numerical_cache/` for optional frozen-figure replay.
Code and fonts come from the repository, not the archive. No collision-event
files, cluster account, ROOT, FastJet or TeX installation is needed.

```bash
tar -xJf /path/to/hybrid_lres_review_bundle.tar.xz
```

## Baseline Run

```bash
python examples/hybrid_lres/run_analysis.py \
  --config examples/hybrid_lres/configs/nominal_all.yaml \
  --inputs inputs --output runs/nominal --stage all
```

This selects 138 bins (hadrons above 30 GeV plus jets, theta_g, dR12 and r_g),
fits each selected group's scaler/PCA, jointly trains the mean-aware REML GP
with the full measured simulation noise, checks the independent holdout, and
evaluates the posterior on the native physical grid. It then plots its own
posterior and fitted-observable predictions as PDF/PNG/SVG.

Inspect `training/training_report.json`, `holdout_summary.json`,
`inference/summary.json`, and `figures/`. A successful run has a top-level
`COMPLETE.json`; failed optimization or grid convergence is not suppressed.
Outputs are always new directories. The baseline has 39 training designs,
holdout 38, 19 retained PCs and nine seeded optimizer starts. Training took
about three minutes on the authors' laptop; full inference and plotting took
about fifteen minutes. Runtime depends on the machine.

The fresh default run was checked against the reference: all nine optimizer
records and fitted kernel parameters agree exactly; posterior total variation
is below 1e-9. See `TRAINING_VALIDATION.json` for the measured comparison.

Expected baseline 16/50/84 percentiles are approximately:

| Parameter | 16% | 50% | 84% |
|---|---:|---:|---:|
| pi*T*Lres | 1.497 | 1.926 | 2.454 |
| kappa_sc | 0.41504 | 0.42290 | 0.43100 |

Compare numerical intervals and predictions, not PDF bytes. Optimizer behavior
can vary across library/platform versions; report differences rather than
changing error bars or declaring a failed fit successful.

## Change Something

Copy a run YAML and edit it, then choose a **new** output directory:

- `selection.sectors`: any combination of `hadrons`, `jets`, `theta_g`, `dR12`, `rg`.
- `selection.hadron_min_pt`: 10 or 30 GeV, for example.
- `selection.exclude_experiments` and `drop_first_angular_bins`: data cuts.
- `training.n_pc`: map native group names to retained-PC counts.
- `native_template`: editable native kernels, bounds, seeds and other group settings.
- `experiment.ell`: the Gaussian experimental shape-correlation length.
- `prior.kind`: `uniform_physical` or `uniform_unit_same_support`.

The unit-prior option is restricted to the same physical support; it is **not**
the full unit-square prior with its beyond-20 tail. Compact grids are explicitly
exploratory and still must pass the convergence check.

Changing selected bins or training choices requires fresh training. Changing
only experimental correlations or priors may reuse a previously fresh fit:

```bash
python examples/hybrid_lres/run_analysis.py \
  --config examples/hybrid_lres/configs/custom_ell.yaml \
  --inputs inputs --output runs/ell_changed --stage infer \
  --trained runs/nominal/training
```

`jets_only.yaml` is another ready-to-edit selection. Config paths are relative
to the run YAML; keep `native_template.yaml` beside copied run configs or set
its path explicitly. The runner checks that reused training matches both the
selected bins and effective native training settings.

## What To Report Back

Send the commit, effective config, completion/failure record, training report,
holdout summary and inference summary, together with any visibly problematic
plots. Start by reproducing the baseline, then change one choice at a time.

This entry point repeats one main fit and its paired holdout, not the whole
fivefold/25-selection campaign or wake-bank extraction. Frozen publication,
fivefold and wake figures remain separately available through `reproduce.py`.
Existing finite-MC, jet-validation-tail and observable-convention qualifications
are not removed by successful numerical reproduction.
