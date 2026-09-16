# Per-PC GP signal amplitude

## Motivation

`sk_learn.fit_emulator` applies `StandardScaler` to observable features, then
`PCA(whiten=False)` and one `GaussianProcessRegressor` per retained PC. Standardizing
the features does not standardize the PC scores. With `normalize_y=False` (the
unchanged sklearn default), those targets can have very different variances.

The Matérn/RBF kernel has unit diagonal signal variance. The optional `constant`
kernel is additive, not a multiplicative signal amplitude. A fixed unit amplitude
can unnecessarily constrain the smooth fit and distort its trade-off with length
scales and fitted white noise. This issue is particularly relevant without PCA
whitening; whitening reduces the scale mismatch but does not fix the optimal
smooth-signal/noise split a priori.

Activating `signal_amplitude` gives each PC `i` the kernel

```text
k_i(theta, theta') = A_i * k_smooth(theta, theta')
                  + k_constant(theta, theta')  # only when already configured
                  + k_white(theta, theta')     # only when already configured
```

`A_i` is an independently optimized `ConstantKernel.constant_value`, a variance
amplitude, not a standard deviation. Only the Matérn/RBF term is multiplied.
WhiteKernel is neither replaced nor rescaled; the existing additive `constant`
retains its meaning. Each GP optimizes its own amplitude, length scales and noise
through the existing log-marginal-likelihood training.

## Configuration

In an emulator's `kernels` block, add `signal_amplitude` to `active` and provide:

```yaml
kernels:
  active: [matern, signal_amplitude, noise]
  signal_amplitude:
    constant_value_factor: 1.0
    constant_value_bounds_factor: [1.0e-4, 1.0e4]
  # Keep the existing matern and noise blocks unchanged.
```

RBF is supported in place of Matérn. For each retained PC, define
`v_i = max(var(Y_pca[:, i], ddof=0), 1e-8)`. The initial amplitude is
`v_i * constant_value_factor`; its optimizer bounds are
`v_i * constant_value_bounds_factor`. The floor avoids a zero-valued kernel for
a degenerate PC; it does not alter the target or add observation noise.
Require finite `0 < lower <= initial_factor <= upper`, with `lower < upper`.

These factors are example settings, not universally optimal physics defaults.
Check fitted kernels and held-out prediction/uncertainty calibration for the
specific design and observable groups. PC selection remains an analysis choice.

## Compatibility and scope

- Omitting `signal_amplitude` from `active` preserves the previous kernel.
- `normalize_y`, `GPR.alpha`, WhiteKernel bounds, PCA/PC counts, PCA truncation
  covariance, covariance reconstruction and the likelihood are unchanged.
- This does not supply simulation statistical errors as per-design `alpha`,
  remove WhiteKernel from predictive covariance, or subtract noise from discarded
  PCs. Those are distinct methodology changes.
- New amplitude-enabled fits record the amplitude settings in the pickle.
  Fitting with an existing pickle or predicting with supplied/loaded results
  rejects an amplitude-setting mismatch, including enabling/disabling the option
  or changing its factors. Retrain using `fit_emulators: true` and
  `force_retrain: true`, or use a fresh output directory. A force flag alone does
  not retrain when the training stage is disabled.
- Legacy artifacts without amplitude metadata remain usable for legacy configs.
  They must be retrained when the option is enabled. This is an amplitude-specific
  compatibility guard, not a general cache-provenance or group-filename fix.

## Tests

`tests/test_emulation_signal_amplitude.py` covers both smooth kernels, additive
constant/WhiteKernel independence, PC-specific initialization and bounds,
zero-variance handling, invalid configuration, legacy behavior, standard
fit/predict/pickle round trips, cache mismatch rejection and forced retraining.
The standard fitter is compared numerically with an explicit sklearn construction
of `ConstantKernel * Matern + WhiteKernel`, with unchanged `alpha` and normalization.

Amplitude flexibility is not a claim that all emulator uncertainties are calibrated;
held-out QA and inference sensitivity remain necessary for an analysis.
