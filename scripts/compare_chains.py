#!/usr/bin/env python
"""Overlay posterior marginals for two chains AND report the convergence metrics that matter.

    usage: myenv312/bin/python scripts/compare_chains.py <lbl1> <h5_1> <lbl2> <h5_2> <out.pdf>

Per parameter (paper order alpha_s, Q0, tau0, ln c1, ln c2, ln c3):
  mode, median, R-hat, integrated autocorrelation time tau, ESS = N_steps*N_walkers/tau, N/tau.
LESSONS (2026-09): R-hat < 1.05 is necessary-not-sufficient. emcee's stricter criterion is
N_steps >= 50*tau (it raises AutocorrError below that). For DE moves, ACCEPTANCE IS MISLEADING:
DE ran at 0.036 acceptance vs 0.188 for StretchMove yet HALVED tau (2x ESS) -- judge by tau/ESS.
For STAT's stored chain (50 walkers, 5000 steps, shape (walkers,steps,params)): transpose to
(steps,walkers,params) and keep ALL steps -- it is already post-burn-in; dropping half inflates R-hat.
"""
import sys, h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee

LOG_IDX = {2, 3, 5}
ORDER = [0, 1, 4, 2, 3, 5]
NAMES = {0: "alpha_s", 1: "Q0", 4: "tau0", 2: "c1", 3: "c2", 5: "c3"}
LABEL = {0: r"$\alpha_S^{\rm fix}$", 1: r"$Q_0$", 4: r"$\tau_0$",
         2: r"$\ln c_1$", 3: r"$\ln c_2$", 5: r"$\ln c_3$"}


def load(f):
    with h5py.File(f, "r") as h:
        return h["chain"][:]           # (steps, walkers, params)


def rhat(x):
    n, m = x.shape[0], x.shape[1]
    wm = x.mean(axis=0); B = n * wm.var(ddof=1); W = x.var(axis=0, ddof=1).mean()
    return np.sqrt(((n - 1) / n * W + B / n) / W)


lbl1, f1, lbl2, f2, out = sys.argv[1:6]
ch = {lbl1: load(f1), lbl2: load(f2)}
COLOR = {lbl1: "#1f77b4", lbl2: "#d62728"}

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, i in zip(axes.ravel(), ORDER):
    for lbl, c in ch.items():
        ax.hist(c[:, :, i].ravel(), bins=80, density=True, histtype="step", lw=1.9,
                color=COLOR[lbl], label=lbl)
    ax.set_xlabel(LABEL[i]); ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=10)
fig.suptitle(f"Posterior marginals: {lbl1}  vs  {lbl2}", fontsize=13)
fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}\n")

tau = {lbl: emcee.autocorr.integrated_time(c, tol=0) for lbl, c in ch.items()}
for lbl, c in ch.items():
    n, w, _ = c.shape
    try:
        emcee.autocorr.integrated_time(c); verdict = "PASSES emcee >=50*tau"
    except emcee.autocorr.AutocorrError:
        verdict = "FAILS emcee >=50*tau (under-sampled)"
    print(f"{lbl}: {n} steps x {w} walkers -> {verdict}")
print()
hdr = f"{'param':8s}" + "".join(f" | {l:>9s} mode {'med':>8s} {'Rhat':>6s} {'tau':>6s} {'N/tau':>6s} {'ESS':>7s}" for l in ch)
print(hdr)
for i in ORDER:
    line = f"{NAMES[i]:8s}"
    for lbl, c in ch.items():
        n, w, _ = c.shape
        col = c[:, :, i]; nat = np.exp(col) if i in LOG_IDX else col
        flat = nat.ravel()
        pv = np.log(flat) if i in LOG_IDX else flat
        hist, edges = np.histogram(pv, bins=60)
        cc = 0.5 * (edges[:-1] + edges[1:])[np.argmax(hist)]
        mode = np.exp(cc) if i in LOG_IDX else cc
        t = tau[lbl][i]
        line += (f" | {mode:14.4f} {np.median(flat):8.4f} {rhat(col):6.3f} {t:6.0f} "
                 f"{n / t:6.1f} {n * w / t:7.0f}")
    print(line)
