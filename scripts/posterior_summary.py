#!/usr/bin/env python
"""Posterior summary for a jetscape_wr chain.

    usage: myenv312/bin/python scripts/posterior_summary.py <mcmc.h5> [burn_fraction]

Prints per-parameter R-hat, integrated autocorrelation time (from the file if stored),
histogram mode, median and 16/84% in NATURAL space. c1,c2,c3 are sampled in natural log
(log_scale_indices [2,3,5]); modes are taken in the sampling space then mapped back.
NOTE: R-hat < 1.05 is necessary, not sufficient -- also check N/tau >= 50 (see compare_chains.py).
"""
import sys, h5py, numpy as np

PARAMS = ["alpha_s", "Q0", "c1", "c2", "tau0", "c3"]
LOG_IDX = {2, 3, 5}
PRIOR_MIN = [0.1, 1, 0.006737946999085467, 0.006737946999085467, 0, 0.049787068367863944]

f = sys.argv[1]
with h5py.File(f, "r") as h:
    chain = h["chain"][:]                                   # (steps, walkers, params)
    tau = h["autocorrelation_time"][:] if "autocorrelation_time" in h else np.full(6, np.nan)
nsteps, nwalk, npar = chain.shape
is_log_space = chain[:, :, 2].min() < PRIOR_MIN[2] - 1e-6


def rhat(x):
    m, n = x.shape[1], x.shape[0]
    wm = x.mean(axis=0); B = n * wm.var(ddof=1); W = x.var(axis=0, ddof=1).mean()
    return np.sqrt(((n - 1) / n * W + B / n) / W)


burn = int(float(sys.argv[2]) * nsteps) if len(sys.argv) > 2 else 0
print(f"file: {f}")
print(f"chain: {nsteps} steps x {nwalk} walkers x {npar} params ; "
      f"sampling_space={'LOG' if is_log_space else 'NATURAL'} ; burn={burn}")
print(f"{'param':7s} {'Rhat':>6s} {'tau':>7s} | {'mode':>8s} {'median':>8s}  {'16%':>8s} {'84%':>8s}")
for i, name in enumerate(PARAMS):
    col = chain[burn:, :, i]
    rh = rhat(col)
    nat = np.exp(col) if (is_log_space and i in LOG_IDX) else col
    flat = nat.ravel()
    med = np.median(flat); q16, q84 = np.percentile(flat, [16, 84])
    plot_vals = np.log(flat) if i in LOG_IDX else flat
    hist, edges = np.histogram(plot_vals, bins=60)
    c = 0.5 * (edges[:-1] + edges[1:])[np.argmax(hist)]
    mode = np.exp(c) if i in LOG_IDX else c
    print(f"{name:7s} {rh:6.3f} {tau[i]:7.1f} | {mode:8.4f} {med:8.4f}  {q16:8.4f} {q84:8.4f}")
