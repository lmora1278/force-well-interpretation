"""Fit frozen FORCE candidates without evaluating reserved observations."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.logistic_regression import BatchLogisticClassifier
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import FEATURES, eligible_rows
from dsm_project.experiments.force_linear import sample_training_wells
from dsm_project.experiments.force_robustness import fit_fixed_regressors

MODEL_NAMES = {
    "regression": {"mean_baseline", "median_baseline", "linear", "tree", "pca2_linear"},
    "classification": {"prior_baseline", "logistic"},
}


def file_hash(path):
    """Identify file bytes rather than a mutable filename."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_candidates(frame, reserved_wells):
    """Exclude whole reserved wells before eligibility, sampling or learning.

    Input is the validated adapter table. Only the membership column of reserved
    rows is consulted here; their features and targets never reach fitting.
    Small synthetic well counts are supported for boundary tests.
    """
    reserved = set(reserved_wells)
    wells = set(frame.WELL.unique())
    if not reserved or len(reserved) != len(reserved_wells):
        raise ValueError("Reserved wells must be nonempty and unique")
    if not reserved.issubset(wells) or not wells - reserved:
        raise ValueError("Require present reserved wells and development wells")
    development = frame.loc[~frame.WELL.isin(reserved)].copy()
    models, tasks = {}, {}
    for task in MODEL_NAMES:
        eligible = eligible_rows(development, task)
        sample = sample_training_wells(eligible, rows_per_well=250, seed=577)
        X = sample.loc[:, FEATURES]
        if task == "regression":
            candidates = fit_fixed_regressors(sample)
        else:
            if sample.sandstone.nunique() != 2:
                raise ValueError("Classification training requires both classes")
            candidates = {
                "prior_baseline": DummyClassifier(strategy="prior"),
                "logistic": make_pipeline(StandardScaler(), BatchLogisticClassifier()),
            }
            for model in candidates.values():
                model.fit(X, sample.sandstone.astype(int))
        details = {}
        for name, model in candidates.items():
            steps = model.steps if hasattr(model, "steps") else [(name, model)]
            settings, fitted = {}, {}
            for step_name, step in steps:
                settings[step_name] = {
                    "class": f"{type(step).__module__}.{type(step).__name__}",
                    "parameters": step.get_params(deep=False),
                }
                fitted[step_name] = {}
                for attr in (
                    "mean_",
                    "scale_",
                    "components_",
                    "explained_variance_",
                    "coef_",
                    "intercept_",
                    "class_prior_",
                    "constant_",
                ):
                    if hasattr(step, attr):
                        fitted[step_name][attr] = np.asarray(
                            getattr(step, attr)
                        ).tolist()
            estimator = steps[-1][1]
            convergence = None
            if hasattr(estimator, "converged_"):
                if not estimator.converged_:
                    raise ValueError(
                        f"{task}/{name} did not converge; no artifact saved"
                    )
                convergence = {
                    "converged": True,
                    "iterations": int(estimator.n_iter_),
                    "gradient_norm": float(estimator.gradient_norm_),
                }
            details[name] = {
                "steps": settings,
                "fitted": fitted,
                "convergence": convergence,
            }
        models[task] = candidates
        tasks[task] = {
            "eligible_rows": len(eligible),
            "sample_rows": len(sample),
            "eligible_wells": sorted(eligible.WELL.unique()),
            "sample_wells": sorted(sample.WELL.unique()),
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(sorted(sample.observation_id)).encode()
            ).hexdigest(),
            "support": {
                name: [float(X[name].min()), float(X[name].max())] for name in FEATURES
            },
            "models": details,
        }
    return models, {
        "schema_version": 1,
        "feature_names": list(FEATURES),
        "feature_units": ["gAPI", "g/cm3", "fraction"],
        "development_wells": sorted(wells - reserved),
        "reserved_wells": sorted(reserved),
        "development_rows": len(development),
        "sampling_seed": 577,
        "rows_per_well": 250,
        "selected_regressor": "tree",
        "classification_threshold": 0.5,
        "primary_regression_metric": "mean_absolute_error",
        "final_test_scored": False,
        "tasks": tasks,
    }


def support_mask(frame, support):
    """Return per-row support under training-only marginal inclusive bounds.

    This is not a physical plausibility or joint-density guarantee. Strict input
    schema errors are distinct from finite, valid but unsupported measurements.
    """
    if frame.columns.duplicated().any() or set(frame.columns) != set(FEATURES):
        raise ValueError("Require exactly the named GR, RHOB and NPHI columns")
    if frame.empty:
        raise ValueError("Require at least one observation")
    if any(
        not pd.api.types.is_numeric_dtype(frame[name])
        or pd.api.types.is_bool_dtype(frame[name])
        for name in FEATURES
    ):
        raise ValueError("Measurements must be numeric, not strings or booleans")
    X = frame.loc[:, FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Measurements must be finite and complete")
    bounds = np.asarray([support[name] for name in FEATURES], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("Invalid support bounds")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Reversed support bounds")
    return ((bounds[:, 0] <= X) & (bounds[:, 1] >= X)).all(axis=1)


def save_candidates(models, metadata, destination):
    """Write a new bundle without overwriting an existing artifact directory.

    A failed write leaves an incomplete directory without a completion manifest.
    Hashes detect accidental changes, not malicious replacement of both files.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "models.joblib"
    joblib.dump(models, payload)
    document = dict(metadata)
    document["created_at"] = datetime.now(UTC).isoformat()
    document["models_sha256"] = file_hash(payload)
    (destination / "metadata.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "COMPLETE.json").write_text(
        json.dumps({"metadata_sha256": file_hash(destination / "metadata.json")}),
        encoding="utf-8",
    )
    return destination


def load_candidates(destination):
    """Verify local trusted bundle integrity before deserializing executable data.

    Never call this on an untrusted downloaded joblib artifact.
    """
    destination = Path(destination)
    complete = json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8"))
    if file_hash(destination / "metadata.json") != complete["metadata_sha256"]:
        raise ValueError("Artifact metadata checksum mismatch")
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["feature_names"] != list(FEATURES):
        raise ValueError("Unsupported artifact schema or features")
    if file_hash(destination / "models.joblib") != metadata["models_sha256"]:
        raise ValueError("Artifact model checksum mismatch")
    models = joblib.load(destination / "models.joblib")
    if set(models) != set(MODEL_NAMES) or any(
        set(models[task]) != names for task, names in MODEL_NAMES.items()
    ):
        raise ValueError("Unexpected candidate set")
    return models, metadata


def build_artifact(destination):
    """Fit from the audited archive only at a clean, identifiable source revision."""
    config = load_config("configs/force.toml")
    root = config.root
    if Path(destination).exists():
        raise FileExistsError(destination)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain"):
        raise ValueError("Commit source changes before building a final artifact")
    source_revision = git("rev-parse", "HEAD")
    tracked = git(
        "ls-files", "src", "configs", "pyproject.toml", "uv.lock"
    ).splitlines()
    hashes = {name: file_hash(root / name) for name in tracked}
    for name in ("docs/EVALUATION_PLAN.md", "docs/PRODUCT_CONTRACT.md"):
        hashes[name] = file_hash(root / name)
    manifest = json.loads(
        (root / config.split["reserved_manifest"]).read_text(encoding="utf-8-sig")
    )
    dataset = load_project_dataset(config)
    if len(manifest["reserved_test_wells"]) != 20:
        raise ValueError("Expected the frozen 20 reserved wells")
    if dataset.observations.WELL.nunique() != 98:
        raise ValueError("Expected the audited 98-well archive")
    if dataset.tasks["regression"].metric != "mean_absolute_error":
        raise ValueError("Regression declaration must match frozen MAE selection")
    models, metadata = fit_candidates(
        dataset.observations, manifest["reserved_test_wells"]
    )
    if git("rev-parse", "HEAD") != source_revision or git("status", "--porcelain"):
        raise ValueError("Source changed during fitting")
    if any(file_hash(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("Source bytes changed during fitting")
    metadata["provenance"] = {
        "source_revision": source_revision,
        "source_clean": True,
        "files_sha256": hashes,
        "archive_sha256": manifest["source_sha256"],
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }
    return save_candidates(models, metadata, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New local artifact directory; never overwritten",
    )
    args = parser.parse_args()
    print(f"Saved candidates without final-test scoring: {build_artifact(args.output)}")


if __name__ == "__main__":
    main()
