"""Reconstruct experimental errors from source metadata, before selecting bins."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian.log_posterior import _build_statistical_covariance, _build_systematic_covariance
from bayesian.systematic_correlation import SystematicCorrelationManager


def load_experiment(path: Path, ell: float = 0.2):
    """Return full experiment and covariance at a user-specified Gaussian length.

    Only explicitly tagged unknown shape correlations vary. Published common
    sources and the original full-measurement coordinates remain unchanged.
    Selecting bins later means taking a principal marginal, not rescaling x.
    """
    ell = float(ell)
    if np.isnan(ell) or ell < 0:
        raise ValueError("Experimental correlation length must be nonnegative")
    data = json.loads(Path(path).read_text())
    exp = data["experiment"]
    for key in ("y", "y_err_stat", "y_err_syst"):
        exp[key] = np.asarray(exp[key], dtype=float)
    exp["observable_ranges"] = [tuple(row) for row in exp["observable_ranges"]]
    for block in exp.get("per_observable_external_stat_cov", {}).values():
        block["matrix"] = np.asarray(block["matrix"], dtype=float)
    manager = SystematicCorrelationManager.from_dict(exp["correlation_manager"])
    params = dict(manager.to_dict()["_pending_correlation_params"])
    changed = [tag for tag in manager.correlation_groups
               if str(params.get(tag, "")).partition(":")[0].lower()
               in ("jetscape", "ptgauss")]
    if changed != data["unknown_shape_tags"]:
        raise ValueError("Experimental shape-source identity mismatch")
    params.update({tag: f"jetscape:{ell!r}" for tag in changed})
    manager.set_correlation_parameters(params)
    manager.register_observable_ranges(exp["observable_ranges"])
    exp["correlation_manager"] = manager
    covariance = _build_statistical_covariance(exp) + _build_systematic_covariance(exp)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite experimental covariance")
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14, rtol=1e-12)
    np.linalg.cholesky(covariance)
    return exp, covariance
