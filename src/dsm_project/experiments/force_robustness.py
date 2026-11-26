"""Fixed-model sensitivity across overlapping whole-well development splits."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.linear_regression import LectureLinearRegressor
from dsm_project.algorithms.student.logistic_regression import BatchLogisticClassifier
from dsm_project.algorithms.student.regression_tree import SquaredErrorTreeRegressor
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import (
    FEATURES,
    LITHOLOGY,
    development_partitions,
    eligible_rows,
)
from dsm_project.experiments.force_linear import sample_training_wells
from dsm_project.experiments.force_logistic import classification_metrics

SPLIT_SEEDS = (577, 578, 579, 580, 581)
SAMPLE_SEED = 577


def fit_fixed_regressors(training):
    """Same sample for every candidate; no tuning inside this experiment."""
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "median_baseline": DummyRegressor(strategy="median"),
        "linear": make_pipeline(StandardScaler(), LectureLinearRegressor()),
        "tree": SquaredErrorTreeRegressor(max_depth=4, min_samples_leaf=50),
        "pca2_linear": make_pipeline(
            StandardScaler(),
            PCA(n_components=2, svd_solver="full"),
            LectureLinearRegressor(),
        ),
    }
    for model in models.values():
        model.fit(training.loc[:, FEATURES], training.DTC)
    return models


def regression_metrics(y, prediction):
    """Signed bias is prediction minus observation, in target units."""
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(root_mean_squared_error(y, prediction)),
        "bias": float(np.mean(np.asarray(prediction) - np.asarray(y))),
    }


def regression_slices(validation, prediction, column):
    """Include missing labels as an explicit slice; never silently drop them."""
    rows = []
    scored = validation.assign(prediction=prediction)
    for value, group in scored.groupby(column, dropna=False, sort=True):
        label = "missing" if pd.isna(value) else str(value)
        rows.append(
            {
                "slice": label,
                "rows": len(group),
                "wells": group.WELL.nunique(),
                **regression_metrics(group.DTC, group.prediction),
            }
        )
    return rows


def evaluate_development(
    frame, reserved_wells, *, seeds=SPLIT_SEEDS, validation_wells=20, rows_per_well=250
):
    """Fit and evaluate fixed candidates; final-test rows never reach any model.

    These repeated holdouts overlap and follow earlier model development.
    Their ranges are descriptive sensitivity, not independent test estimates
    or confidence intervals. Seeds are fixed before inspecting these results.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require nonempty unique split seeds")
    metrics, per_well, rock, classification, class_rock = [], [], [], [], []
    manifests, convergence, coverage = [], [], []
    for seed in seeds:
        train, valid = development_partitions(
            frame,
            reserved_wells,
            validation_wells=validation_wells,
            seed=seed,
        )
        if set(train.WELL) & set(valid.WELL):
            raise RuntimeError("Training/validation overlap")
        if (set(train.WELL) | set(valid.WELL)) & set(reserved_wells):
            raise RuntimeError("Reserved well reached development")
        manifest = {
            "seed": seed,
            "training_wells": sorted(train.WELL.unique()),
            "validation_wells": sorted(valid.WELL.unique()),
        }
        for task in ("regression", "classification"):
            training = sample_training_wells(
                eligible_rows(train, task),
                rows_per_well=rows_per_well,
                seed=SAMPLE_SEED,
            )
            validation = eligible_rows(valid, task)
            if validation.empty:
                raise ValueError("No eligible validation observations")
            coverage.append(
                {
                    "seed": seed,
                    "task": task,
                    "training_partition_rows": len(train),
                    "training_eligible_rows": len(eligible_rows(train, task)),
                    "training_sample_rows": len(training),
                    "validation_partition_rows": len(valid),
                    "validation_eligible_rows": len(validation),
                    "training_eligible_wells": training.WELL.nunique(),
                    "validation_eligible_wells": validation.WELL.nunique(),
                }
            )
            manifest[f"{task}_sample_sha256"] = hashlib.sha256(
                "\n".join(sorted(training.observation_id)).encode()
            ).hexdigest()
            if task == "regression":
                models = fit_fixed_regressors(training)
                for name, model in models.items():
                    prediction = model.predict(validation.loc[:, FEATURES])
                    metrics.append(
                        {
                            "seed": seed,
                            "model": name,
                            **regression_metrics(validation.DTC, prediction),
                        }
                    )
                    per_well.extend(
                        {"seed": seed, "model": name, **row}
                        for row in regression_slices(validation, prediction, "WELL")
                    )
                    rock.extend(
                        {"seed": seed, "model": name, **row}
                        for row in regression_slices(validation, prediction, LITHOLOGY)
                    )
                    if name in ("linear", "pca2_linear"):
                        estimator = model.steps[-1][1]
                        convergence.append(
                            {
                                "seed": seed,
                                "model": name,
                                "converged": bool(estimator.converged_),
                                "iterations": estimator.n_iter_,
                            }
                        )
            else:
                models = {
                    "prior_baseline": DummyClassifier(strategy="prior"),
                    "logistic": make_pipeline(
                        StandardScaler(), BatchLogisticClassifier()
                    ),
                }
                if training.sandstone.nunique() != 2:
                    raise ValueError("Training classification requires both classes")
                for name, model in models.items():
                    model.fit(training.loc[:, FEATURES], training.sandstone.astype(int))
                    probability = model.predict_proba(validation.loc[:, FEATURES])[:, 1]
                    values = classification_metrics(
                        validation.sandstone.astype(int), probability
                    )
                    classification.append(
                        {
                            "seed": seed,
                            "model": name,
                            "rows": len(validation),
                            "prevalence": float(validation.sandstone.mean()),
                            **{
                                k: v
                                for k, v in values.items()
                                if k != "confusion_matrix"
                            },
                            "tn": values["confusion_matrix"][0][0],
                            "fp": values["confusion_matrix"][0][1],
                            "fn": values["confusion_matrix"][1][0],
                            "tp": values["confusion_matrix"][1][1],
                        }
                    )
                    scored = validation.assign(flag=probability >= 0.5)
                    for label, group in scored.groupby(LITHOLOGY, dropna=False):
                        class_rock.append(
                            {
                                "seed": seed,
                                "model": name,
                                "lithology_code": str(label),
                                "rows": len(group),
                                "wells": group.WELL.nunique(),
                                "positive_flag_rate": float(group.flag.mean()),
                            }
                        )
                    if name == "logistic":
                        estimator = model.steps[-1][1]
                        convergence.append(
                            {
                                "seed": seed,
                                "model": name,
                                "converged": bool(estimator.converged_),
                                "iterations": estimator.n_iter_,
                            }
                        )
        manifests.append(manifest)
    return {
        "regression": pd.DataFrame(metrics),
        "per_well": pd.DataFrame(per_well),
        "rock_type": pd.DataFrame(rock),
        "classification": pd.DataFrame(classification),
        "classification_rock_type": pd.DataFrame(class_rock),
        "coverage": pd.DataFrame(coverage),
        "convergence": pd.DataFrame(convergence),
        "split_manifests": manifests,
    }


def run_experiment():
    config = load_config("configs/force.toml")
    manifest = json.loads(
        (config.root / config.split["reserved_manifest"]).read_text(
            encoding="utf-8-sig"
        )
    )
    dataset = load_project_dataset(config)
    result = evaluate_development(dataset.observations, manifest["reserved_test_wells"])
    destination = config.root / "artifacts/force_robustness"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(destination / f"{name}.csv", index=False)
    report = {
        "source_sha256": manifest["source_sha256"],
        "split_seeds": list(SPLIT_SEEDS),
        "sampling_seed": SAMPLE_SEED,
        "rows_per_training_well": 250,
        "tree": {"max_depth": 4, "min_samples_leaf": 50},
        "classification_threshold": 0.5,
        "interpretation": "overlapping development splits; descriptive only",
        "final_test_scored": False,
        "all_converged": bool(result["convergence"].converged.all()),
        "split_manifests": result["split_manifests"],
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    result["report"] = report
    return result


if __name__ == "__main__":
    outcome = run_experiment()
    print(outcome["regression"].to_string(index=False))
    print(outcome["classification"].to_string(index=False))
    print("All iterative fits converged:", outcome["report"]["all_converged"])
