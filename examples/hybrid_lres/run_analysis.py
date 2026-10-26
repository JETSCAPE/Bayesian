#!/usr/bin/env python3
"""Portable source tables -> own-subset PCA/REML -> physical-grid inference.

Unlike reproduce.py, this driver never consumes frozen review models/posteriors.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Set before importing NumPy/SciPy. threadpoolctl also limits already loaded BLAS.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numerical_inference as ni


def features(rows):
    result = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("Features require (observable, lower, upper) triples")
        name, lower, upper = str(row[0]), float(row[1]), float(row[2])
        if not name or not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Invalid feature identity")
        result.append((name, lower, upper))
    if not result or len(set(result)) != len(result):
        raise ValueError("Empty or duplicate feature layout")
    return result


def save_json(path, value):
    def finite_metadata(item):
        if isinstance(item, dict):
            return {key: finite_metadata(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite_metadata(value) for value in item]
        if isinstance(item, float) and np.isinf(item):
            return "infinity" if item > 0 else "-infinity"
        return item
    def encode(item):
        if isinstance(item, (np.ndarray, np.generic)):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    with Path(path).open("x") as stream:
        json.dump(finite_metadata(value), stream, indent=2, allow_nan=False, default=encode)
        stream.write("\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_features(experiment):
    # The loader retains native ranges, while the driver attaches exact bin IDs.
    if "features" not in experiment:
        raise ValueError("Attach experiment['features'] from experiment_sources.json before infer()")
    return features(experiment["features"])


def infer(bank, experiment, covariance, q, kappa, output, *,
          prior="uniform_physical", batch_size=128, tv_tolerance=.02,
          exploratory=False):
    """Infer from a freshly trained bank; never train or load a reference here.

    Attach the full exported triples as ``experiment['features']``. Only the
    bank's own features are likelihood/predictive targets. Native likelihood
    includes the parameter-dependent determinant, without the constant n log 2pi.
    ``output`` must not exist. Failed refinement is saved, then raises.
    """
    from bayesian.log_posterior import _loglikelihood

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite inference output: " + str(output))
    report = bank.get("report", {})
    if report.get("status") != "TRAINED" or report.get("optimize") is not True or report.get("best_success") is not True:
        raise ValueError("Inference requires fresh, optimized, converged REML training")
    if type(batch_size) is not int or batch_size < 1 or not 0 < tv_tolerance <= .02:
        raise ValueError("Positive batch_size and TV tolerance <= .02 required")
    full, own = experiment_features(experiment), features(bank["features"])
    try:
        ix = np.array([full.index(row) for row in own], dtype=int)
    except ValueError as error:
        raise ValueError("Trained features absent from exported experiment") from error
    q, kappa = np.asarray(q, float), np.asarray(kappa, float)
    ni.weights(q)
    ni.weights(kappa)
    if len(q) % 2 != 1 or len(kappa) % 2 != 1 or min(len(q), len(kappa)) < 5:
        raise ValueError("Exact nested odd grid counts >= 5 required")
    if not exploratory and (len(q), len(kappa)) != (361, 241):
        raise ValueError("Nonbaseline grid counts must be explicitly exploratory")
    if q[0] < 0 or q[-1] > 20 or kappa[0] < .3 or kappa[-1] > .6:
        raise ValueError("Inference support outside q=[0,20], kappa=[.3,.6]")
    observed = np.asarray(experiment["y"], float)
    covariance = np.asarray(covariance, float)
    if observed.shape != (len(full),) or covariance.shape != (len(full), len(full)):
        raise ValueError("Experimental arrays do not match full feature layout")
    if not np.isfinite(observed).all() or not np.isfinite(covariance).all():
        raise ValueError("Nonfinite experimental inputs")
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14, rtol=1e-12)
    observed, covariance = observed[ix], covariance[np.ix_(ix, ix)]
    np.linalg.cholesky(covariance)
    # Validate prior before creating an output or evaluating a model.
    ni.mass(np.zeros((len(q), len(kappa))), q, kappa, prior)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "STARTED.json", dict(joint_fit_id=report["joint_fit_id"],
        features=own, prior=prior, exploratory=exploratory, shape=[len(q), len(kappa)],
        native_likelihood="bayesian.log_posterior._loglikelihood, including log determinant"))
    points = ni.physical_points(q, kappa)
    means = np.empty((len(points), len(own)))
    variances = np.empty_like(means)
    loglike = np.empty(len(points))
    with threadpool_limits(limits=1):
        for start in range(0, len(points), batch_size):
            stop = min(start + batch_size, len(points))
            prediction = bank["model"].predict(points[start:stop])
            means[start:stop], variances[start:stop] = ni.validate_prediction(prediction, stop - start, len(own))
            for j, cell in enumerate(range(start, stop)):
                loglike[cell] = _loglikelihood(
                    np.asarray(means[cell] - observed, dtype=np.float64),
                    np.asarray(prediction["cov"][j] + covariance, dtype=np.float64))
            if start % (batch_size * 16) == 0 or stop == len(points):
                print(f"FRESH_INFERENCE {stop}/{len(points)}", flush=True)
        loglike = loglike.reshape(len(q), len(kappa))
        mass, tv = ni.convergence(loglike, q, kappa, prior)
        np.savez_compressed(output / "posterior_grid.npz", q=q, kappa=kappa, u_L=ni.unit(q),
                            mass=mass, loglike=loglike, features=np.asarray(own, str),
                            experimental_covariance=covariance, observed=observed)
        result = ni.summary(mass, q, kappa, prior)
        result.update(joint_fit_id=report["joint_fit_id"], selected_features=len(own),
                      coarse_fine_TV=tv, tv_tolerance=tv_tolerance,
                      converged=bool(tv <= tv_tolerance), exploratory=exploratory,
                      paper_baseline_grid=not exploratory,
                      predictions_scope="Own-subset fitted targets only; no auxiliary frozen mapping")
        save_json(output / "summary.json", result)
        if tv > tv_tolerance:
            raise RuntimeError(f"Posterior refinement failed: TV={tv:.6g} > {tv_tolerance}; no bypass or retry")
        probabilities = (.025, .16, .5, .84, .975)
        quantiles = ni.mixture_quantiles(means, variances, mass.ravel(), probabilities)
        posterior_mean = mass.ravel() @ means
        posterior_variance = mass.ravel() @ (variances + means**2) - posterior_mean**2
        np.savez_compressed(output / "predictions.npz", features=np.asarray(own, str),
            probabilities=probabilities, quantiles=quantiles, mean=posterior_mean,
            variance=posterior_variance, observed=observed, experimental_covariance=covariance,
            route=np.full(len(own), "fresh_own_subset_model"))
    save_json(output / "COMPLETE.json", dict(joint_fit_id=report["joint_fit_id"],
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    return result


SECTORS = ("hadrons", "jets", "theta_g", "dR12", "rg")
DEFAULT_CONFIG = dict(schema=1, native_template=None,
    selection=dict(sectors=list(SECTORS), hadron_min_pt=30., exclude_experiments=[],
                   drop_first_angular_bins=dict(theta_g=0, dR12=0, rg=0)),
    training=dict(n_pc={}, n_restarts=None, random_state=None),
    experiment=dict(ell=.2, kernel_power=2),
    prior=dict(kind="uniform_physical", q=[0., 20.], kappa=[.3, .6]),
    grid=dict(mode="paper", counts=None, batch_size=128, tv_tolerance=.02),
    baseline_summary=None)
INPUT_NAMES = ("accepted_observables.h5", "full_training.npz", "native_template.yaml",
               "experiment_sources.json", "grid_axes.npz")


def _merge(default, supplied, where="config"):
    if not isinstance(supplied, dict) or set(supplied) - set(default):
        raise ValueError(f"Unknown keys or nonmapping at {where}")
    result = deepcopy(default)
    for key, value in supplied.items():
        if isinstance(default[key], dict):
            # n_pc is intentionally a free native group-name mapping.
            if key == "n_pc":
                if not isinstance(value, dict):
                    raise ValueError("training.n_pc must map native group names to counts")
                result[key] = value
            else:
                result[key] = _merge(default[key], value, where + "." + key)
        else:
            result[key] = value
    return result


def validate_config(supplied):
    cfg = _merge(DEFAULT_CONFIG, supplied)
    if type(cfg["schema"]) is not int or cfg["schema"] != 1:
        raise ValueError("Expected config schema: 1")
    if cfg["native_template"] is not None and not isinstance(cfg["native_template"], str):
        raise ValueError("native_template must be an optional editable YAML path")
    selection = cfg["selection"]
    sectors = selection["sectors"]
    if (not isinstance(sectors, list) or not sectors or len(set(sectors)) != len(sectors)
            or not set(sectors) <= set(SECTORS)):
        raise ValueError("sectors must be a nonempty unique list of known source sectors")
    minimum = selection["hadron_min_pt"]
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not np.isfinite(minimum) or minimum < 10:
        raise ValueError("hadron_min_pt must be finite and >= 10 GeV")
    exclusions = selection["exclude_experiments"]
    if (not isinstance(exclusions, list) or any(not isinstance(x, str) or not x for x in exclusions)
            or len(set(x.lower() for x in exclusions)) != len(exclusions)):
        raise ValueError("exclude_experiments must be a unique list of names")
    for count in selection["drop_first_angular_bins"].values():
        if type(count) is not int or count < 0:
            raise ValueError("Angular first-bin cuts must be nonnegative integer counts")
    for key, value in cfg["training"].items():
        choices = value.values() if isinstance(value, dict) else [value]
        for item in choices:
            if item is not None and (type(item) is not int or item < (1 if key == "n_pc" else 0)):
                raise ValueError("Training choices require integer PC counts > 0, restarts/seeds >= 0")
    ell = cfg["experiment"]["ell"]
    if isinstance(ell, bool) or not isinstance(ell, (int, float)) or np.isnan(ell) or ell < 0:
        raise ValueError("ell must be numeric, nonnegative (infinity is supported)")
    if cfg["experiment"]["kernel_power"] != 2:
        raise ValueError("Only the native Gaussian source kernel (power 2) is supported")
    if cfg["prior"]["kind"] not in ("uniform_physical", "uniform_unit_same_support"):
        raise ValueError("Prior must use physical support; full unit prior requires an explicit tail")
    for name, limits in (("q", (0, 20)), ("kappa", (.3, .6))):
        bounds = cfg["prior"][name]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bounds)
                or not np.isfinite(bounds).all() or not limits[0] <= bounds[0] < bounds[1] <= limits[1]):
            raise ValueError("Prior support must lie inside q=[0,20], kappa=[.3,.6]")
    grid = cfg["grid"]
    if grid["mode"] not in ("paper", "compact"):
        raise ValueError("grid.mode must be paper or compact")
    if grid["mode"] == "paper":
        if grid["counts"] is not None or cfg["prior"]["q"] != [0, 20] or cfg["prior"]["kappa"] != [.3, .6]:
            raise ValueError("Paper baseline requires exact exported axes and original physical support")
    else:
        counts = grid["counts"]
        if not isinstance(counts, list) or len(counts) != 2 or any(type(n) is not int or n < 5 or n % 2 != 1 for n in counts):
            raise ValueError("Compact exploratory grid counts must be two odd integers >= 5")
    if type(grid["batch_size"]) is not int or grid["batch_size"] < 1:
        raise ValueError("grid.batch_size must be a positive integer")
    tolerance = grid["tv_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 < tolerance <= .02:
        raise ValueError("TV tolerance must be positive and <= .02; no convergence bypass")
    if cfg["baseline_summary"] is not None and not isinstance(cfg["baseline_summary"], str):
        raise ValueError("baseline_summary must be an optional JSON path")
    return cfg


def sector(name):
    lowered = name.lower()
    matches = [kind for kind, tokens in (
        ("hadrons", ("__hadron__",)), ("jets", ("__inclusive_jet__pt_", "__jet__pt__")),
        ("theta_g", ("__tg_",)), ("dR12", ("__dr12_",)), ("rg", ("__rg_",)))
        if any(token in lowered for token in tokens)]
    if len(matches) != 1:
        raise ValueError("Source feature has an ambiguous or unknown sector: " + name)
    return matches[0]


def select_features(rows, policy):
    rows = features(rows)
    kinds = {name: sector(name) for name, _, _ in rows}
    excluded = [name.lower() for name in policy["exclude_experiments"]]
    for name in excluded:
        if not any(name in row[0].lower() for row in rows):
            raise ValueError("Excluded experiment absent from source: " + name)
    dropped = set()
    for name in dict.fromkeys(row[0] for row in rows):
        count = policy["drop_first_angular_bins"].get(kinds[name], 0)
        if count:
            bins = sorted((row for row in rows if row[0] == name), key=lambda row: row[1:])
            if count >= len(bins):
                raise ValueError("Angular cut removes an entire observable: " + name)
            dropped.update(bins[:count])
    selected = [row for row in rows if kinds[row[0]] in policy["sectors"]
        and not any(name in row[0].lower() for name in excluded)
        and (kinds[row[0]] != "hadrons" or row[1] >= policy["hadron_min_pt"])
        and row not in dropped]
    return features(selected)


def grid_axes(path, cfg):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"q", "kappa"}:
            raise ValueError("Grid input must contain axes only, never a posterior")
        axes = [archive[key].copy() for key in ("q", "kappa")]
    if tuple(len(a) for a in axes) != (361, 241):
        raise ValueError("Require exported native 361x241 physical template")
    for axis, bounds in zip(axes, ([0., 20.], [.3, .6]), strict=True):
        ni.weights(axis)
        if not np.array_equal(axis[[0, -1]], bounds):
            raise ValueError("Native physical template support differs")
    if cfg["grid"]["mode"] == "compact":
        selected = []
        for name, axis, count in zip(("q", "kappa"), axes, cfg["grid"]["counts"], strict=True):
            bounds = cfg["prior"][name]
            if any(value not in axis for value in bounds):
                raise ValueError("Exploratory support endpoints must be exact exported grid nodes")
            axis = axis[(axis >= bounds[0]) & (axis <= bounds[1])]
            if count > len(axis):
                raise ValueError("Compact counts exceed exported axis nodes")
            selected.append(axis[np.rint(np.linspace(0, len(axis) - 1, count)).astype(int)])
        axes = selected
    return axes


def training_overrides(cfg):
    return {key: value for key, value in cfg["training"].items() if value is not None and value != {}}


def load_inputs(root, cfg, config_dir=None):
    from experimental_inputs import load_experiment

    root = Path(root).resolve(strict=True)
    paths = {name: root / name for name in INPUT_NAMES}
    if cfg["native_template"] is not None:
        chosen = Path(cfg["native_template"])
        paths["native_template.yaml"] = (Path(config_dir or root) / chosen).resolve()
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Missing source inputs; expected " + ", ".join(INPUT_NAMES))
    hashes = {name: sha(path) for name, path in paths.items()}
    receipt = root / "TRAINING_INPUTS.json"
    if receipt.exists():
        manifest = json.loads(receipt.read_text())
        if manifest.get("schema") != "hybrid-lres-training-inputs-v1":
            raise ValueError("Unsupported input manifest")
        for name, digest in hashes.items():
            if name == "native_template.yaml":
                continue  # Editable scientific configuration, not immutable data.
            if manifest["files"].get(name, {}).get("sha256") != digest:
                raise ValueError("Training input hash mismatch: " + name)
    source = json.loads(paths["experiment_sources.json"].read_text())
    full = features(source["features"])
    selected = select_features(full, cfg["selection"])
    exp, cov = load_experiment(paths["experiment_sources.json"], cfg["experiment"]["ell"])
    exp["features"] = full
    q, kappa = grid_axes(paths["grid_axes.npz"], cfg)
    template = yaml.safe_load(paths["native_template.yaml"].read_text())
    return dict(paths=paths, hashes=hashes, selected=selected, exp=exp, cov=cov,
                q=q, kappa=kappa, template=template)


def load_fresh_training(path, inputs, cfg):
    """Load only a local canonical train() artifact with matching source identity."""
    from bayesian.emulation.review_state import load_prediction_state

    path = Path(path).resolve(strict=True)
    if not (path / "training_report.json").exists() and (path / "training").is_dir():
        path = path / "training"
    report = json.loads((path / "training_report.json").read_text())
    if (report.get("status") != "TRAINED" or report.get("optimize") is not True
            or report.get("best_success") is not True):
        raise ValueError("Not a fresh optimized converged paper_training output")
    if (report.get("source_observables_sha256") != inputs["hashes"]["accepted_observables.h5"]
            or report.get("covariance_sha256") != inputs["hashes"]["full_training.npz"]):
        raise ValueError("Fresh training source hashes differ; new fit required")
    if (set(features(report["features"])) != set(inputs["selected"])
            or report.get("overrides", {}) != training_overrides(cfg)):
        raise ValueError("Masks or training choices changed; new fit required")
    model, rows, ids, metadata = load_prediction_state(path / "prediction_state.npz")
    if metadata["provenance"] != report or features(rows) != features(report["features"]):
        raise ValueError("Fresh state and training report identity differ")
    np.testing.assert_array_equal(ids, report["training_ids"])
    # Compare the supplied native template with the effective output config, after
    # replacing only run-local paths. No model state is used to resolve choices.
    from bayesian.emulation.paper_training import _selected_config
    names = dict.fromkeys(row[0] for row in inputs["selected"])
    effective = yaml.safe_load((path / "config.yaml").read_text())
    analyses = [name for name, item in effective["analyses"].items()
                if "emulators" in item.get("parameters", {})]
    # The canonical API may retain unused analyses in the template; its report
    # records the actual result path, which determines the selected analysis.
    actual = Path(report["native_output"]).name
    matches = [(a, p) for a in analyses for p in effective["analyses"][a]["parameterizations"]
               if actual == a + "_" + p]
    kwargs = dict(analysis_name=matches[0][0], parameterization=matches[0][1]) if len(matches) == 1 else {}
    expected, _, _, _ = _selected_config(inputs["template"], path, names,
        inputs["paths"]["full_training.npz"].resolve(), training_overrides(cfg),
        kwargs.get("analysis_name"), kwargs.get("parameterization"))
    def portable_config(value):
        value = deepcopy(value)
        for key in ("output_dir", "observable_table_dir", "observable_config_dir"):
            value.pop(key, None)
        for entry in value["analyses"].values():
            for group in entry.get("parameters", {}).get("emulators", {}).values():
                group.get("simulation_noise", {}).pop("covariance_file", None)
        return value
    if portable_config(expected) != portable_config(effective):
        raise ValueError("Native template differs from fresh fit configuration; new fit required")
    return dict(model=model, features=features(rows), report=report, training_path=path)


def holdout_summary(training, output):
    """Keep native paired-noise QA separate from posterior fitted-bin plots."""
    path = Path(training) / "holdout_qa.npz"
    if not path.exists():
        result = dict(available=False, reason="Native trainer supplied no holdout rows")
    else:
        shutil.copyfile(path, Path(output) / "holdout_qa.npz")
        with np.load(path, allow_pickle=False) as archive:
            qa = {key: archive[key] for key in archive.files}
        paired = "observed_covariance" in qa
        mean = qa["observed_mean"] if paired else qa["latent_mean"]
        covariance = qa["observed_covariance"] if paired else qa["latent_covariance"]
        residual = qa["truth"] - mean
        variance = np.diagonal(covariance, axis1=1, axis2=2)
        if np.any(variance <= 0):
            raise ValueError("Holdout variance nonpositive; no repair allowed")
        pull = residual / np.sqrt(variance)
        distances = [float(r @ np.linalg.solve(c, r)) for r, c in zip(residual, covariance, strict=True)]
        result = dict(available=True, paired_mc_available=paired,
            target="Noisy heldout simulation with measured test/train cross noise" if paired else "Latent only: not noise-calibrated coverage",
            design_ids=qa["design_ids"].tolist(), bins=int(mean.shape[1]),
            marginal_1sigma_fraction=float(np.mean(abs(pull) <= 1)),
            marginal_2sigma_fraction=float(np.mean(abs(pull) <= 2)),
            full_covariance_squared_distances=distances,
            caveat="Shared-MC validation does not establish independent-generator calibration")
    save_json(Path(output) / "holdout_summary.json", result)
    return result


def render_figures(inference, output, *, ell=.2, holdout=None, selection=None):
    """New scientific plots, using the vendored original geometry and style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = HERE / "original"
    if str(original) not in sys.path:
        sys.path.insert(0, str(original))
    from original import geometry
    from original import hybrid_plotting as hp

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(Path(inference) / "posterior_grid.npz", allow_pickle=False) as archive:
        q, kappa, mass = (archive[key] for key in ("q", "kappa", "mass"))
    with np.load(Path(inference) / "predictions.npz", allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    rows = features(prediction["features"])
    hadron_bins = [row[1] for row in rows if sector(row[0]) == "hadrons"]
    cut = (rf"hadron $p_{{\mathrm{{T}}}}\geq {min(hadron_bins):g}$ GeV"
           if hadron_bins else "no hadrons fitted")
    mask = "source-sector selection"
    if selection:
        mask_parts = ["drop " + ",".join(selection["exclude_experiments"])] if selection["exclude_experiments"] else []
        mask_parts += [f"{name}: first {count} removed" for name, count in selection["drop_first_angular_bins"].items() if count]
        mask = "; ".join(mask_parts) if mask_parts else "no additional masks"
    header = f"Fresh own-subset fit: {len(rows)} bins; {cut}\n{mask}"
    style = hp.length_style(ell) if ell == 0 or np.isinf(ell) or .02 <= ell <= 1 else hp.comparison_style("default")
    with hp.paper_style():
        fig, ax = plt.subplots(figsize=(hp.FIGURE_WIDTH, 4.8), layout="constrained")
        fig.suptitle(header, fontsize=12)
        density, levels = geometry.density_and_levels(mass, q, kappa)
        levels = sorted({float(level) for level in levels if density.min() < level < density.max()})
        if levels:
            ax.contour(q, kappa, density.T, levels=levels, colors=[style["color"]], linewidths=2.6)
        else:
            ax.pcolormesh(q, kappa, density.T, shading="nearest", cmap="Greys")
        ax.set(xscale="log", xlabel=hp.LRES_LABEL, ylabel=hp.KAPPA_LABEL,
               xlim=(max(.5, q[0]), q[-1]), ylim=(kappa[0], kappa[-1]))
        ax.text(.02, .025, f"Full prior integrated: q = {q[0]:g} to {q[-1]:g}",
                transform=ax.transAxes, fontsize=11, va="bottom")
        hp.frame_axes(ax, minor=False)
        hp.save_figure(fig, output / "posterior_joint")
        plt.close(fig)
        for page, name in enumerate(dict.fromkeys(row[0] for row in rows), 1):
            ix = np.array([j for j, row in enumerate(rows) if row[0] == name])
            center = np.array([(rows[j][1] + rows[j][2]) / 2 for j in ix])
            span = np.array([[rows[j][1], rows[j][2]] for j in ix]).ravel()
            bands = prediction["quantiles"][:, ix]
            fig, ax = plt.subplots(figsize=(hp.FIGURE_WIDTH, 4.8), layout="constrained")
            fig.suptitle(header, fontsize=12)
            ax.fill_between(span, bands[0].repeat(2), bands[4].repeat(2), color=style["color"], alpha=.12)
            ax.fill_between(span, bands[1].repeat(2), bands[3].repeat(2), color=style["color"], alpha=.25)
            ax.plot(span, bands[2].repeat(2), color=style["color"], linestyle=style["linestyle"], label="Fresh latent prediction")
            ax.errorbar(center, prediction["observed"][ix],
                yerr=np.sqrt(np.diag(prediction["experimental_covariance"])[ix]),
                fmt="o", color="black", markersize=4, label="Data (marginal errors)")
            kind = sector(name)
            experiment = next((key.upper() for key in ("alice", "atlas", "cms") if key in name.lower()), "Simulation")
            observable = {"hadrons": "hadrons", "jets": "jets", "theta_g": r"$\theta_g$",
                          "dR12": r"$dR_{12}$", "rg": r"$r_g$"}[kind]
            parts = name.split("__")
            radius = parts[4].split("_")[0] if len(parts) > 4 and parts[4].startswith("R") else ""
            title = f"{experiment} {observable}, {parts[-1]}%" + (f", {radius}" if radius else "")
            window = next((token[2:] for token in parts[4].split("_") if token.startswith("pt")), "") if len(parts) > 4 else ""
            if window:
                title += f", jet pT {window} GeV"
            xlabel = {"hadrons": r"$p_{\mathrm{T}}$ [GeV]", "jets": r"$p_{\mathrm{T}}$ [GeV]",
                      "theta_g": r"$\theta_{\mathrm{g}}$", "dR12": r"$dR_{12}$",
                      "rg": r"$r_{\mathrm{g}}$"}[kind]
            ax.set(title=title, xlabel=xlabel,
                   ylabel="PbPb / pp" if kind == "theta_g" else hp.RAA_LABEL)
            if kind == "dR12":
                ax.set_xscale("symlog", linthresh=.003)
                ax.set_xlim(min(rows[j][1] for j in ix), max(rows[j][2] for j in ix))
            ax.legend(fontsize=11, loc="best")
            hp.frame_axes(ax)
            hp.save_figure(fig, output / f"predictions_{page:02d}")
            plt.close(fig)
        if holdout is not None and Path(holdout).exists():
            with np.load(holdout, allow_pickle=False) as qa:
                truth, latent = qa["truth"], qa["latent_mean"]
                fig, ax = plt.subplots(figsize=(hp.FIGURE_WIDTH, 4.8), layout="constrained")
                ax.scatter(truth.ravel(), latent.ravel(), s=18, color=hp.DEFAULT_COLOR)
                low, high = min(truth.min(), latent.min()), max(truth.max(), latent.max())
                ax.plot([low, high], [low, high], color="black", linestyle=":")
                fig.suptitle(header, fontsize=12)
                ax.set(xlabel="Noisy heldout simulation", ylabel="Fresh latent emulator",
                       title="Latent scatter, not observed-MC calibration")
                hp.frame_axes(ax)
                hp.save_figure(fig, output / "holdout_scatter")
                plt.close(fig)
    save_json(output / "FIGURES.json", dict(formats=["pdf", "png", "svg"],
        geometry="Vendored original density_and_levels; physical coordinates",
        style="hybrid_plotting approved defaults", ell="infinity" if np.isinf(ell) else ell,
        selection=selection, own_fitted_bins=len(rows), displayed_q="True log axis [0.5,20]; full support integrated",
        checked="Automatic clipping checks; inspect rendered figures for overlaps"))


def compare_baseline(path, fresh_summary, output):
    """Optional metadata comparison, called only after fresh inference completes."""
    baseline = json.loads(Path(path).read_text())
    comparison = dict(reference_sha256=sha(path),
        reference_role="Read after fresh train/infer only; never a warm start or result source",
        fresh_joint_fit_id=fresh_summary["joint_fit_id"], intervals={})
    for key in ("pi_T_Lres_16_50_84", "kappa_16_50_84"):
        if key in baseline:
            old, new = np.asarray(baseline[key], float), np.asarray(fresh_summary[key], float)
            if old.shape != (3,) or not np.isfinite(old).all():
                raise ValueError("Invalid baseline interval metadata")
            comparison["intervals"][key] = dict(reference=old.tolist(), fresh=new.tolist(),
                                                 fresh_minus_reference=(new - old).tolist())
    save_json(Path(output) / "baseline_comparison.json", comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW directory; no overwrites or resume")
    parser.add_argument("--stage", choices=("validate", "prepare", "train", "infer", "all"), default="all")
    parser.add_argument("--trained", type=Path, help="Fresh canonical train output (or parent containing training/); infer only")
    parser.add_argument("--no-plots", action="store_true", help="Keep numerical outputs and native holdout QA")
    args = parser.parse_args(argv)
    if (args.stage == "infer") != (args.trained is not None):
        parser.error("--stage infer requires --trained; other stages must not load training state")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite: " + str(output))
    cfg = validate_config(yaml.safe_load(args.config.read_text()))
    with threadpool_limits(limits=1):
        inputs = load_inputs(args.inputs, cfg, args.config.resolve().parent)
        bank = load_fresh_training(args.trained, inputs, cfg) if args.stage == "infer" else None
        output.mkdir(parents=True, exist_ok=False)
        with (output / "effective_run.yaml").open("x") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        save_json(output / "INPUTS.json", dict(hashes=inputs["hashes"],
            selected_features=inputs["selected"], effective_config=cfg,
            native_template_editable=True, stage=args.stage, blas_threads=1,
            scope="Fresh PCA/REML and own-subset physical-grid inference; never replay"))
        started = time.monotonic()
        journal = output / "journal.jsonl"
        def event(kind, **values):
            with journal.open("a") as stream:
                stream.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic() - started, **values)) + "\n")
        event("started", stage=args.stage)
        try:
            if args.stage in ("prepare", "train", "all"):
                from bayesian.emulation import paper_training
                function = paper_training.prepare if args.stage == "prepare" else paper_training.train
                event("native_prepare" if args.stage == "prepare" else "native_train")
                training = output / ("prepared" if args.stage == "prepare" else "training")
                bank = function(inputs["paths"]["accepted_observables.h5"], inputs["template"],
                    inputs["paths"]["full_training.npz"], inputs["selected"], training,
                    overrides=training_overrides(cfg))
                event("prepared" if args.stage == "prepare" else "trained", report=bank["report"]["status"])
                if args.stage != "prepare":
                    bank["training_path"] = training
            if args.stage in ("train", "all", "infer"):
                holdout_summary(bank["training_path"], output)
            if args.stage in ("infer", "all"):
                event("inference_started")
                result = infer(bank, inputs["exp"], inputs["cov"], inputs["q"], inputs["kappa"],
                    output / "inference", prior=cfg["prior"]["kind"],
                    batch_size=cfg["grid"]["batch_size"], tv_tolerance=cfg["grid"]["tv_tolerance"],
                    exploratory=cfg["grid"]["mode"] == "compact")
                event("inference_passed", coarse_fine_TV=result["coarse_fine_TV"])
                if not args.no_plots:
                    render_figures(output / "inference", output / "figures", ell=cfg["experiment"]["ell"],
                                   holdout=output / "holdout_qa.npz", selection=cfg["selection"])
                if cfg["baseline_summary"]:
                    baseline = Path(cfg["baseline_summary"])
                    if not baseline.is_absolute():
                        baseline = args.config.resolve().parent / baseline
                    compare_baseline(baseline, result, output)
            # Recheck immutable input contents; no task changes these source files.
            for name, path in inputs["paths"].items():
                if sha(path) != inputs["hashes"][name]:
                    raise ValueError("Input changed during run: " + name)
            event("complete")
            save_json(output / "COMPLETE.json", dict(stage=args.stage, input_hashes=inputs["hashes"],
                seconds=time.monotonic() - started,
                artifacts={str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}))
        except Exception as error:
            event("failed", error_type=type(error).__name__, message=str(error))
            save_json(output / "FAILED.json", dict(type=type(error).__name__, message=str(error),
                      policy="No retry, noise modification, or convergence bypass"))
            raise
    print("FRESH_ANALYSIS_COMPLETE " + str(output), flush=True)
    return output


if __name__ == "__main__":
    main()
