"""Pure fresh-fit inference mathematics, migrated from the original checks.

No data/model loading or author-workspace imports. Physical q = pi*T*Lres
must not be confused with its bounded emulator coordinate u_L.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import ndtr


def unit(q):
    return (2 / np.pi) * np.arctan(np.cbrt(q))


def physical(u):
    return np.tan(.5 * np.pi * np.asarray(u)) ** 3


def physical_points(q, kappa):
    u, k = np.meshgrid(unit(q), (np.asarray(kappa) - .3) / .3, indexing="ij")
    return np.column_stack((u.ravel(), k.ravel()))


def weights(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Quadrature axes must be finite, strictly increasing, length >= 3")
    return np.r_[np.diff(x)[0], x[2:] - x[:-2], np.diff(x)[-1]] / 2


def mass(loglike, q, kappa, prior="uniform_physical"):
    x = unit(q) if prior == "uniform_unit_same_support" else q
    if prior not in ("uniform_physical", "uniform_unit_same_support"):
        raise ValueError("Unsupported prior; full unit support requires a separate tail integral")
    ll = np.asarray(loglike, dtype=float)
    if ll.shape != (len(q), len(kappa)) or not np.isfinite(ll).all():
        raise ValueError("Invalid likelihood grid")
    value = np.exp(ll - ll.max()) * weights(x)[:, None] * weights(kappa)[None, :]
    return value / value.sum()


def convergence(loglike, q, kappa, prior="uniform_physical"):
    """Native nested coarse/fine density TV, in the prior's own coordinates."""
    q, kappa = np.asarray(q), np.asarray(kappa)
    if len(q) % 2 != 1 or len(kappa) % 2 != 1 or min(len(q), len(kappa)) < 5:
        raise ValueError("Nested axes require odd counts >= 5 and both endpoints")
    result = mass(loglike, q, kappa, prior)
    coarse = mass(loglike[::2, ::2], q[::2], kappa[::2], prior)
    x = unit(q) if prior == "uniform_unit_same_support" else q
    volume = weights(x)[:, None] * weights(kappa)[None, :]
    density = coarse / (weights(x[::2])[:, None] * weights(kappa[::2])[None, :])
    mesh = np.stack(np.meshgrid(x, kappa, indexing="ij"), axis=-1)
    interpolated = RegularGridInterpolator((x[::2], kappa[::2]), density)(mesh)
    interpolated /= np.sum(interpolated * volume)
    return result, float(.5 * np.sum(np.abs(result / volume - interpolated) * volume))


def quantiles(value, x, axis):
    marginal = value.sum(axis=1 - axis)
    cdf = (np.cumsum(marginal) - .5 * marginal) / marginal.sum()
    return np.interp([.16, .5, .84], cdf, x)


def summary(value, q, kappa, prior):
    x = unit(q) if prior == "uniform_unit_same_support" else q
    qq = quantiles(value, x, 0)
    if prior == "uniform_unit_same_support":
        qq = physical(qq)
    return dict(pi_T_Lres_16_50_84=qq.tolist(),
                kappa_16_50_84=quantiles(value, kappa, 1).tolist(),
                percentile_convention="Native centered CDF", prior=prior,
                physical_support=dict(q=[float(q[0]), float(q[-1])],
                                      kappa=[float(kappa[0]), float(kappa[-1])]),
                unit_support=[float(unit(q[0])), float(unit(q[-1]))],
                full_unit_prior=False)


def validate_prediction(prediction, rows, features):
    mean = np.asarray(prediction["central_value"], dtype=float)
    cov = np.asarray(prediction["cov"], dtype=float)
    if mean.shape != (rows, features) or cov.shape != (rows, features, features):
        raise ValueError("Emulator prediction shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError("Nonfinite emulator prediction")
    np.testing.assert_allclose(cov, cov.swapaxes(-1, -2), atol=1e-11, rtol=1e-10)
    if np.any(np.linalg.eigvalsh(cov)[:, 0] < -1e-10 * np.maximum(abs(cov).max(axis=(1, 2)), 1)):
        raise ValueError("Latent covariance is not PSD; no repair allowed")
    variance = np.diagonal(cov, axis1=1, axis2=2).copy()
    if np.any(variance < 0):
        raise ValueError("Negative latent variance; no clipping allowed")
    return mean, variance


def mixture_quantiles(mean, variance, probability, levels=(.025, .16, .5, .84, .975)):
    """Exact marginal Gaussian-mixture CDF roots; no discarded grid mass."""
    mean, variance, probability = map(np.asarray, (mean, variance, probability))
    if mean.ndim != 2 or variance.shape != mean.shape or probability.shape != (len(mean),):
        raise ValueError("Misaligned predictive mixture")
    if (not all(np.isfinite(a).all() for a in (mean, variance, probability))
            or np.any(variance < 0) or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1)):
        raise ValueError("Invalid predictive mixture")
    result = np.empty((len(levels), mean.shape[1]))
    for j in range(mean.shape[1]):
        mu, sigma = mean[:, j], np.sqrt(variance[:, j])
        positive = sigma > 0
        if not positive.all():
            # Zero-variance components are point masses. Bisection handles jumps.
            def cdf(x):
                values = (mu <= x).astype(float)
                values[positive] = ndtr((x - mu[positive]) / sigma[positive])
                return probability @ values
        else:
            def cdf(x):
                return probability @ ndtr((x - mu) / sigma)
        lo, hi = float(np.min(mu - 12 * sigma)), float(np.max(mu + 12 * sigma))
        for i, level in enumerate(levels):
            if lo == hi:
                result[i, j] = lo
            elif positive.all():
                result[i, j] = brentq(lambda x: cdf(x) - level, lo, hi, xtol=1e-11)
            else:
                left, right = lo, hi
                for _ in range(80):
                    middle = (left + right) / 2
                    if cdf(middle) >= level:
                        right = middle
                    else:
                        left = middle
                result[i, j] = right
    return result


def conditional(mean, covariance, observed, fit, held):
    """Original Gaussian Schur conditional; auxiliary common-model use only."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    fit, held = np.asarray(fit, int), np.asarray(held, int)
    if not len(fit) or not len(held) or len(set(np.r_[fit, held])) != len(fit) + len(held):
        raise ValueError("Conditioning/heldout indices must be nonempty and disjoint")
    aa = covariance[:, fit][:, :, fit]
    ba = covariance[:, held][:, :, fit]
    bb = covariance[:, held][:, :, held]
    gain = np.linalg.solve(aa, ba.swapaxes(1, 2)).swapaxes(1, 2)
    central = mean[:, held] + np.einsum("mij,mj->mi", gain, observed[fit] - mean[:, fit])
    schur = bb - gain @ ba.swapaxes(1, 2)
    return central, (schur + schur.swapaxes(1, 2)) / 2
