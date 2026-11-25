"""Training-only standardized PCA, extending the template representation example."""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.linear_regression import LectureLinearRegressor
from dsm_project.config import load_config
from dsm_project.data.force import FEATURES
from dsm_project.experiments.force_linear import run_experiment as run_linear


def fit_representation_models(training):
    """Fit scaler, PCA and custom linear model using training rows only.

    PCA sees predictors only; DTC enters only the downstream regression fit.
    Full SVD and no whitening match ordinary standardized PCA. Retain all
    three dimensionalities as diagnostics, not a validation-selected policy.
    """
    models = {}
    for count in (1, 2, 3):
        models[count] = make_pipeline(
            StandardScaler(),
            PCA(n_components=count, svd_solver="full"),
            LectureLinearRegressor(),
        ).fit(training.loc[:, FEATURES], training.DTC)
    return models


def run_experiment():
    """Reuse regression data; exclude reserved wells from fitting and scoring."""
    base = run_linear()
    training, validation = base["training"], base["validation"]
    models = fit_representation_models(training)
    Xv = validation.loc[:, FEATURES]
    rows, per_well = [], []
    for count, model in models.items():
        scaler, pca, linear = [step[1] for step in model.steps]
        if not linear.converged_:
            raise RuntimeError("PCA regression did not converge")
        standardized = scaler.transform(Xv)
        reconstructed = pca.inverse_transform(pca.transform(standardized))
        prediction = model.predict(Xv)
        rows.append(
            {
                "components": count,
                "training_variance_retained": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "validation_standardized_reconstruction_mse": float(
                    np.mean((standardized - reconstructed) ** 2)
                ),
                "validation_mae": mean_absolute_error(validation.DTC, prediction),
                "validation_rmse": root_mean_squared_error(validation.DTC, prediction),
            }
        )
        for well, group in validation.assign(prediction=prediction).groupby("WELL"):
            per_well.append(
                {
                    "components": count,
                    "well": well,
                    "rows": len(group),
                    "mae": mean_absolute_error(group.DTC, group.prediction),
                }
            )
    full = models[3].named_steps["pca"]
    report = {
        k: base["report"][k]
        for k in (
            "source_sha256",
            "sample_ids_sha256",
            "training_rows",
            "validation_rows",
            "training_wells",
            "validation_wells",
        )
    }
    report.update(
        {
            "evaluation": "development; no dimension selected; final test excluded",
            "scope": "regression complete cases; PCA fits predictors without labels",
            "features": list(FEATURES),
            "explained_variance_ratio": full.explained_variance_ratio_.tolist(),
            "components": full.components_.tolist(),
            "scaler_mean": models[3].named_steps["standardscaler"].mean_.tolist(),
            "scaler_scale": models[3].named_steps["standardscaler"].scale_.tolist(),
            "comparisons": rows,
            "uncompressed_linear": base["report"]["metrics"]["lecture_linear"],
            "full_rotation_max_prediction_gap": float(
                np.max(
                    np.abs(
                        models[3].predict(Xv)
                        - base["models"]["lecture_linear"].predict(Xv)
                    )
                )
            ),
        }
    )
    destination = load_config("configs/force.toml").root / "artifacts/force_pca"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(destination / "dimensions.csv", index=False)
    pd.DataFrame(per_well).to_csv(destination / "per_well.csv", index=False)
    return {
        "report": report,
        "training": training,
        "validation": validation,
        "models": models,
        "per_well": pd.DataFrame(per_well),
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment()["report"], indent=2))
