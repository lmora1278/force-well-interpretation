"""Whole-well sandstone screening with a fixed initial decision policy."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.logistic_regression import BatchLogisticClassifier
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import FEATURES, development_partitions, eligible_rows
from dsm_project.experiments.force_linear import sample_training_wells


def classification_metrics(targets, probabilities, threshold=0.5):
    """Summarize binary scores; undefined single-class measures are explicit nulls.

    Confusion matrix rows are observed [0,1], columns predicted [0,1]. Precision
    is zero when no positives are predicted; recall is null without positives.
    """
    y, p = np.asarray(targets), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.size == 0 or p.shape != y.shape:
        raise ValueError("Require aligned nonempty one-dimensional targets and scores")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Require binary targets and finite probabilities")
    if ((p < 0) | (p > 1)).any() or not 0 < threshold < 1:
        raise ValueError("Probabilities or decision threshold out of range")
    prediction = (p >= threshold).astype(int)
    both = np.unique(y).size == 2
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction))
        if both
        else None,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction)) if np.any(y == 1) else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def fit_logistic_models(training):
    """Fit identical sample/scaling boundaries for custom and reference models."""
    X, y = training.loc[:, FEATURES], training.sandstone.astype(int)
    models = {
        "prior_baseline": DummyClassifier(strategy="prior"),
        "batch_logistic": make_pipeline(StandardScaler(), BatchLogisticClassifier()),
        "logistic_reference": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-10, max_iter=10000),
        ),
    }
    for model in models.values():
        model.fit(X, y)
    return models


def run_experiment():
    """Report validation evidence, retaining the fixed 0.5 primary threshold."""
    config = load_config("configs/force.toml")
    manifest = json.loads(
        (config.root / config.split["reserved_manifest"]).read_text(
            encoding="utf-8-sig"
        )
    )
    dataset = load_project_dataset(config)
    train, validation = development_partitions(
        dataset.observations,
        manifest["reserved_test_wells"],
        validation_wells=config.split["validation_wells"],
        seed=config.split["random_state"],
    )
    training = sample_training_wells(eligible_rows(train, "classification"))
    validation = eligible_rows(validation, "classification")
    models = fit_logistic_models(training)
    metrics, scores, per_well = {}, {}, []
    for name, model in models.items():
        probabilities = model.predict_proba(validation.loc[:, FEATURES])[:, 1]
        scores[name] = probabilities
        metrics[name] = classification_metrics(
            validation.sandstone.astype(int), probabilities
        )
        for well, group in validation.assign(probability=probabilities).groupby("WELL"):
            values = classification_metrics(
                group.sandstone.astype(int), group.probability
            )
            per_well.append(
                {
                    "model": name,
                    "well": well,
                    "rows": len(group),
                    "positives": int(group.sandstone.sum()),
                    **{k: v for k, v in values.items() if k != "confusion_matrix"},
                }
            )
    thresholds = []
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        values = classification_metrics(
            validation.sandstone.astype(int), scores["batch_logistic"], threshold
        )
        thresholds.append(
            {
                "threshold": threshold,
                **{k: values[k] for k in ("balanced_accuracy", "precision", "recall")},
            }
        )
    fitted = models["batch_logistic"].steps[-1][1]
    scaler = models["batch_logistic"].steps[0][1]
    report = {
        "evaluation": "development validation only; final test not scored",
        "source_sha256": manifest["source_sha256"],
        "positive_label": "30000 Sandstone; mixed sandstone/shale remains negative",
        "seed": 577,
        "rows_per_training_well": 250,
        "threshold": 0.5,
        "threshold_sweep": "exploratory only; no threshold selected from validation",
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_wells": sorted(training.WELL.unique()),
        "validation_wells": sorted(validation.WELL.unique()),
        "training_prevalence": float(training.sandstone.mean()),
        "validation_prevalence": float(validation.sandstone.mean()),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(training.observation_id)).encode()
        ).hexdigest(),
        "metrics": metrics,
        "converged": fitted.converged_,
        "iterations": fitted.n_iter_,
        "gradient_norm": fitted.gradient_norm_,
        "training_log_loss": float(fitted.loss_history_[-1]),
        "parameters": fitted.get_params(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients_standardized": fitted.coef_[0].tolist(),
        "intercept_standardized": float(fitted.intercept_[0]),
        "max_probability_gap_to_reference": float(
            np.max(np.abs(scores["batch_logistic"] - scores["logistic_reference"]))
        ),
    }
    destination = config.root / "artifacts/force_logistic"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(per_well).to_csv(destination / "per_well.csv", index=False)
    pd.DataFrame(thresholds).to_csv(destination / "thresholds.csv", index=False)
    pd.DataFrame(
        {
            "iteration": np.arange(len(fitted.loss_history_)),
            "training_log_loss": fitted.loss_history_,
        }
    ).to_csv(destination / "loss_history.csv", index=False)
    return {
        "report": report,
        "training": training,
        "validation": validation,
        "models": models,
        "per_well": pd.DataFrame(per_well),
        "thresholds": pd.DataFrame(thresholds),
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment()["report"], indent=2))
