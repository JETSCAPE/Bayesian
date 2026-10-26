# Hybrid Resolution-Length Review Bundle

Two separate entry points serve different purposes:

- **Fresh emulator training and inference:** see [TRAINING.md](TRAINING.md).
  Starts from accepted prediction tables and their full measured simulation
  covariance. Editable configurations select observables, training settings,
  experimental correlations, and priors. No saved emulator or posterior is
  an input to the fresh fit.
- **Frozen-result/figure replay:** `reproduce.py`, documented below. This is
  faster, but is not a test of training or inference.

Neither workflow generates collision events.

The single private `hybrid_lres_review_bundle.tar.xz` contains all non-code
inputs for both workflows. Extract it from the repository root with
`tar -xJf /path/to/hybrid_lres_review_bundle.tar.xz`. It creates `inputs/`
and `examples/hybrid_lres/numerical_cache/`. The older two-file training
distribution and frozen-cache-only archive described below remain valid,
but are not needed in addition to the combined bundle.

This example redraws all 18 figures currently referenced by the manuscript from
JSON/NPZ numerical results. It does not copy manuscript PDFs, deserialize
Matplotlib figures, load model pickles, optimize kernels, run inference, or generate
events. Source PDFs are hashed during author export, never used as drawing inputs.
The original scientific values, posterior support, bands, bin edges, contour
probabilities, and view ranges are retained. Exact PDF bytes are not a goal.

## Replay Frozen Figures

Use Python 3.12 and install this example's minimal `requirements.txt`, then set
`PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"` to expose this checkout's
canonical `bayesian.emulation.review_state` API. A full project installation,
PyMC, and event/inference dependencies are not needed. The minimal list includes
attrs, PyYAML, silx, and h5py because native package initialization imports them,
even though the state check never opens model pickles or HDF5. No TeX installation
is needed. Extract the
companion cache archive into this checkout, so `numerical_cache/manifest.json`
is beside this README. The data are deliberately **not in Git**.

Set `$REPO` to your checkout root and use a **new** output location. This command
works from any working directory:

```bash
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python "$REPO/examples/hybrid_lres/reproduce.py" --input "$REPO/examples/hybrid_lres/numerical_cache" --output /tmp/hybrid-lres-reproduced --verify-state
```

The paths above are examples in documentation, not runtime constants. The renderer
uses only its own code/fonts, the requested input bundle, and the requested output
directory. It forces one BLAS thread. It refuses existing output directories,
including partial renders, rather than silently overwriting or resuming them.
`--manuscript-only` draws just the 18 paper figures. `--verify-state-only` checks
the numerical GP checkpoints without drawing. `--training-archive FILE` additionally
verifies the separate full raw training archive, which is not needed for plotting.

Each figure is written as PDF, 300-dpi PNG, and outlined-text SVG. An indexed,
bookmarked `indexed_figures.pdf` and `REPRODUCED.json` identify every page and its
numerical checks. Figures use `hybrid_plotting`, Latin Modern Roman, width
**5.95276 in**, body **10.90909 PDF bp**, Computer Modern math, and named color roles.
The four unmodified Latin Modern OTF files are from the official
[CTAN Latin Modern distribution](https://ctan.org/tex-archive/fonts/lm); the
GUST Font License, font README, and original font manifest are included in `fonts/`.

## Three Review Levels

1. **Frozen numerical results to figures.** Full physical posterior masses for
   all 25 selected-only fits, all five ell cases where computed, full-square prior
   tails, all 170 prediction-bin identities and all five quantiles, own-fit versus
   auxiliary routes, full experimental covariance matrices, native QA arrays,
   closure masses, common-model conditional predictions, wake endpoint masses,
   wake profile moments, and fixed-design response/covariance inputs are retained.
   Bulky per-grid prediction means/variances and likelihood arrays are omitted
   because neither figure regeneration nor mass comparisons require them.
2. **Frozen native prediction checks.** Numerical all/inclusive/jets checkpoints
   and four original probe-coordinate reference predictions are included. The
   canonical `FastMeanAware` implementation is imported through `review_state`;
   the example contains no independent GP implementation. Means are checked
   bitwise; covariance comparisons allow only `rtol=5e-12, atol=5e-13` roundoff.
   These states are prediction checkpoints, not replacements for training data,
   optimizer histories, kernel fits, or validation studies.
3. **Source and full-input scrutiny.** `original/executed_analysis/` contains exact
   SHA-bound historical analysis/control sources, stored as `.py.txt` so they are
   inert and cannot accidentally be imported or collected as tests. Their old
   author paths, loaders, and fit commands are historical evidence, **not portable
   replay entry points**. `SHA256.json` maps each original filename to its snapshot.
   Canonical methods now live in `bayesian.emulation.mean_aware`,
   `bayesian.emulation.omitted_coordinates`, and `bayesian.experimental_selection`.
   The separate full covariance archive preserves the full cross-design/cross-bin
   matrix, not just the diagonal errors used by display panels.

The physical resolution variable `q = pi*T*Lres` on `[0,20]` is distinct from
the unit-square coordinate `u = 2/pi*atan(cbrt(q))`. Full unit-square priors include
their beyond-20 tail; contours are not renormalized to the visible range. Native
jet-validation tails, conditional observable definitions, plug-in assumptions,
and the distinction between nominal 39-design and matched-wake 30-design banks
remain qualifications. Conditional predictions are auxiliary controls, not
own-subset fits or formal compatibility p-values.

## Contents And Provenance

The manuscript's 18 filenames are enumerated in `manifest.reference.json` and in
the companion cache manifest. The renderer compares the SHA signatures of every
scientific line, band, contour, patch, axis scale, and view range against the
reviewed JHEP render record. All three observable galleries are regenerated,
including the low-pT auxiliary points, not silently omitted.

Additional pages cover sector and angular comparisons, angular-pair additions,
experiment leave-outs, hadron and first-angular-bin cuts, ell intervals and
contours, both prior comparisons, a matched wake-resolution comparison, and
complete three-page own-source galleries for every one of the 25 fitted models.
Every extra page checks that reflow/styling did not alter scientific artist
geometry. All gallery bins are explicitly audited for complete coverage.

`original/PROVENANCE.json` binds the original plotting files, selected functions,
source line ranges, exact symbol digests, generated helper digests, and the few
explicit data-access substitutions. `hybrid_plotting.py` is copied bytewise. The
helper is a bounded snapshot, not a new plotting or inference framework.

Source checkout anchors at export time:

- Native review repository base: `9eb58eec4606e0999d42f6822f7759c3f0c0668b`.
- Canonical methods validation checkout: `991e78dc9d8d658f04d38bd803c33eb4c120147b`.
- Manuscript checkout: `b45a9ff32a8c353dbbb1b3c2488afc906e4d04ad`.
- Author-local analysis/render sources include uncommitted/unversioned work;
  their exact SHA256 snapshots and input receipts, not either commit alone, are
  authoritative. No claim is made that these commits reproduce the author caches.

`manifest.reference.json` is the tracked export reference. The reproduction and
native-state CLI actions require an exact SHA256 match to that shipped manifest
before trusting any cache, scientific signature, or state. A self-consistent
replacement cache with its own signatures is not accepted. It lists original
source hashes, array shapes/digests, model/feature identities, and cache payload
hashes without personal filesystem paths. The data tarball must match it; SHA256
checks are integrity checks, not signatures establishing the author's identity.

## Companion Data

The default cache includes the three prediction states and references. The large
raw training archive is deliberately separate and ignored by Git:

```text
data_archive/full_training.npz
```

It is a byte-for-byte copy of the original
`atlas_rg_inference_20261015/recovery_20261016/precision_continuation/timeout_completion/export/full_simulation_covariance_rg.npz`.
The companion manifest records its original SHA256 and all array identities. The
raw covariance shape is `(40,170,40,170)`; moment-derived central values, design IDs, observable
names, edges, radius-bin map, and source diagnostics accompany it. The archive is
351,241,717 bytes. It supports deeper review; no marginal-error plot substitutes
for it. Obtain it from the author's companion release artifact, or copy that
  original local file and verify its hash with the manifest. No public download URL
has been assigned here; do not infer one from the historical directory name.

**The first 134 `central_values` entries in this covariance archive are
moment-derived means, not the old pinned table means accepted by the analysis.**
For genuine refitting, `data_archive/accepted_observables.h5` is a required
small training input: a byte-for-byte copy of the original frozen
`local_inference/results/main/analysis_raa_unit_Lres-E-loss/observables.h5`.
It preserves accepted `Design`, `Design_indices`, `Prediction`, validation
tables, and experimental inputs. Its original SHA and size are recorded as
`external_accepted_tables` in the manifest. The frozen-figure replay does not
open it. The fresh-training workflow does; see [TRAINING.md](TRAINING.md).

Distribution is through **private manuscript-repository release assets only**,
not a public code release. The manuscript repository remains code-free: attach
the data archive as a private release asset; keep the maintained code in the
native-methods repository/PR. No upload is performed here.

The quick companion tarball is `data_archive/hybrid_lres_reproduction.tar.gz`.
It contains `numerical_cache/` (including safe native states and references) and
the small optional accepted-table archive. The maintained code, fonts, inert source
evidence, and tracked reference manifest come from the native-methods code PR;
the data asset does not put code in the manuscript repository. It excludes the
dense raw training covariance and rendered output. Extract at the repository root
with `tar -xzf ARTIFACT`. Distribution URLs and final artifact hashes are in the
release record once published; no upload is performed by this example.

## Author Export Only

The author workspace is accepted only as a command argument. Sources stay
read-only and existing output paths are refused:

```bash
python examples/hybrid_lres/export_local.py --source-root "$AUTHOR_WORKSPACE" --output /tmp/new-hybrid-cache --training-archive /tmp/new-full-training.npz --prediction-states "$PREDICTION_STATES"
```

`vendor_original.py --source-root AUTHOR_WORKSPACE` explicitly refreshes the
bounded reviewed drawing snapshots and inert historical sources. This is a
maintenance operation, not a collaborator reproduction step. It never imports
the historical analysis drivers.

## Tests

```bash
python -m pytest tests/test_hybrid_lres_reproduction.py -q
HYBRID_LRES_RENDER_TEST=1 python -m pytest tests/test_hybrid_lres_reproduction.py -q
```

The second command additionally renders all 18 manuscript figures from an
unrelated working directory, verifies all original scientific signatures, checks
all three output formats plus the PDF index, and checks overwrite refusal.
Rendering also validates clipped text and legends intersecting axes. Actual
rendered pages still require visual inspection; those automated checks are not
an exhaustive overlap detector or scientific certification.

`requirements.lock.txt` records the complete resolved minimal environment used
for validation; it does not install the full project's inference stack.
