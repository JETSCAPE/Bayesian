Hybrid resolution-length paper: method review
============================================

This branch makes the paper-used local methods reviewable. It does not alter
the accepted fits, tables, inference settings, or manuscript figures. No old
model is silently promoted to a new implementation. The frozen-result replay
example is separate from the Overleaf repository.

Entry points
------------

* ``emulation.simulation_noise`` aligns measured simulation covariance by design
  ID and observable/bin identity, and propagates it through standardization and
  PCA. Noise need not be equal per bin or diagonal across bins, designs or PCs.
* ``emulation.joint_groups`` preserves measured cross-group blocks. Independent
  latent GP priors do not imply independent observations or independent fits.
* ``emulation.mean_aware.RestrictedMeanGP`` is the paper's explicit opt-in
  constant-mean-marginalized REML objective, retaining native signal kernels,
  bounds, restart vectors and numerical conditioning.
* ``MeanAware`` and ``FastMeanAware`` implement dense and reduced algebra for the
  same latent prediction. No target MC error is added to latent predictions.
  Comparing against a simulated target requires its explicit train/target and
  target/target MC blocks; a shared pp reference is not independent noise.
* ``emulation.omitted_coordinates.omitted_maps`` completes the omitted physical
  coordinate complement even when the number of training designs is smaller
  than the number of bins. This does not add fitted PCs or invent empirical
  physical variance in directions absent from the SVD.
* ``parameter_prior`` implements the declared physical prior, including its
  coordinate Jacobian when evaluated in emulator coordinates.
* ``experimental_selection.select_experiment`` preserves the principal
  experimental covariance, source sharing and original pre-cut coordinates
  when selecting bins. It rejects incompatible legacy covariance conventions.

The ordinary native joint-group interface remains opt-in and does not
automatically select REML. The paper uses the named mean-aware classes above;
do not reproduce it by running default zero-mean training instead. Likewise,
the included optional shift/shape diagnostic is not the nominal experimental
covariance. The nominal unresolved-source kernel is ``jetscape:0.2``, with
Gaussian exponent 2, not the pre-existing ``ptgauss`` exponent 1.9.

Every fitted observable subset has its own selected-bin PCA and emulator.
The simulation covariance is marginalized to that selection, not conditioned
on unselected noisy measurements. Excluded-data plots use a separately labelled
auxiliary full-bank mapping. They are not predictions from the subset fit.

Portable prediction checkpoints
-------------------------------

``emulation.review_state`` saves numerical conditioning state and a restricted
kernel description in NPZ/JSON, without pickle. ``load_prediction_state`` uses
the same repository ``FastMeanAware.predict`` implementation. It supports
latent prediction and explicit paired noisy-target prediction, not optimization
or refitting. Training covariance must be supplied separately for a refit.
Machine-roundoff changes from array memory layout are tested and reported;
they are not claimed to be bitwise-identical predictions.

The author-only ``scripts/export_hybrid_lres_prediction_state.py`` loads trusted
historical caches, exports them without optimization, and compares reconstructed
predictions with the executed implementation. It takes an explicit workspace
path. Collaborator replay does not require that workspace or trust its pickles.

Review boundaries
-----------------

The empirical PCA basis and plug-in kernel parameters are fixed during
prediction. The measured finite-MC covariance, first-order ratio propagation,
fixed generator cross sections, empirical omitted variation and unknown
experimental correlations are assumptions, not a complete theory-error model.
Passing implementation tests is not proof of calibrated scientific coverage.
The paper's jet cross-validation tails persist; the bounded jet-error stresses
support the combined resolution extraction more strongly than exact jet-only
posterior positions. Separate optimizer-completion controls remain separate
from their original baseline fits.

The wake comparison is an ad hoc conserving, reduced-statistics sensitivity
test. Its event-analysis code belongs in the analysis repository, not here.
It is not an implementation of a full-shower hydrodynamic wake or a calibrated
theoretical nuisance prior. The nominal and wake design/event samples differ.

Source provenance
-----------------

The native covariance/prior/interface files are byte-preserving transplants
from the exact paper-used checkout based on JETSCAPE/Bayesian ``9c4e7bf``.
The mean-aware classes are import-only migrations of
``emulator_mean_check_20261013/mean_model.py`` and
``emulator_mean_refit_20261013/refit_model.py``. Their class ASTs are checked
against those sources. The complete omitted mapper is migrated unchanged from
``emulator_discarded_checks_20261014/omitted_completion.py``.
Old sources and completed caches are preserved. The source map accompanying
the review bundle binds their SHA256 values and repository locations.
