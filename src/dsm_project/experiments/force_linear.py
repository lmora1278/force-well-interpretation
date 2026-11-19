"""Bounded whole-well development experiment; never scores reserved test wells."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.linear_regression import LectureLinearRegressor
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import FEATURES, development_partitions, eligible_rows


def sample_training_wells(frame, rows_per_well=250, seed=577):
    """Sample at most a fixed row count per well without replacement.

    Sort first so input row order cannot change selection. Equal caps prevent
    long wells from dominating training. This changes training weighting and
    does not turn adjacent depth observations into independent samples.
    """
    if isinstance(rows_per_well, bool) or not isinstance(rows_per_well, int):
        raise ValueError("rows_per_well must be a positive integer")
    if rows_per_well < 1 or frame.empty:
        raise ValueError("Require nonempty data and positive rows_per_well")
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in frame.sort_values(["WELL", "DEPTH_MD"]).groupby("WELL", sort=True):
        positions = rng.choice(
            len(group), min(rows_per_well, len(group)), replace=False
        )
        groups.append(group.iloc[np.sort(positions)])
    return pd.concat(groups, ignore_index=True)


def fit_linear_models(training):
    """Fit scaling and all candidates on exactly the same training sample."""
    X, y = training.loc[:, FEATURES], training.DTC
    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "lecture_linear": make_pipeline(StandardScaler(), LectureLinearRegressor()),
        "least_squares_reference": make_pipeline(StandardScaler(), LinearRegression()),
    }
    for model in models.values():
        model.fit(X, y)
    return models


def run_experiment():
    """Write reproducible validation diagnostics to ignored artifacts/force_linear."""
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
    training = sample_training_wells(eligible_rows(train, "regression"))
    validation = eligible_rows(validation, "regression")
    models = fit_linear_models(training)
    metrics, per_well, predictions = {}, [], {}
    for name, model in models.items():
        prediction = model.predict(validation.loc[:, FEATURES])
        predictions[name] = prediction
        metrics[name] = {
            "mae": mean_absolute_error(validation.DTC, prediction),
            "rmse": root_mean_squared_error(validation.DTC, prediction),
        }
        scored = validation.assign(prediction=prediction)
        for well, group in scored.groupby("WELL", sort=True):
            per_well.append(
                {
                    "model": name,
                    "well": well,
                    "rows": len(group),
                    "mae": mean_absolute_error(group.DTC, group.prediction),
                    "rmse": root_mean_squared_error(group.DTC, group.prediction),
                }
            )
    fitted = models["lecture_linear"].steps[-1][1]
    scaler = models["lecture_linear"].steps[0][1]
    report = {
        "evaluation": "development validation only; final test not scored",
        "source_sha256": manifest["source_sha256"],
        "seed": 577,
        "rows_per_training_well": 250,
        "training_rows": len(training),
        "validation_rows": len(validation),
        "training_wells": sorted(training.WELL.unique()),
        "validation_wells": sorted(validation.WELL.unique()),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(training.observation_id)).encode()
        ).hexdigest(),
        "metrics": metrics,
        "converged": fitted.converged_,
        "iterations": fitted.n_iter_,
        "gradient_norm": fitted.gradient_norm_,
        "training_mse": float(fitted.loss_history_[-1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients_original_units": (fitted.coef_ / scaler.scale_).tolist(),
        "intercept_original_units": float(
            fitted.intercept_ - np.dot(fitted.coef_, scaler.mean_ / scaler.scale_)
        ),
        "max_prediction_gap_to_reference": float(
            np.max(
                np.abs(
                    predictions["lecture_linear"]
                    - predictions["least_squares_reference"]
                )
            )
        ),
    }
    destination = config.root / "artifacts" / "force_linear"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(per_well).to_csv(destination / "per_well.csv", index=False)
    pd.DataFrame(
        {
            "iteration": np.arange(len(fitted.loss_history_)),
            "training_mse": fitted.loss_history_,
        }
    ).to_csv(destination / "loss_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_experiment()
