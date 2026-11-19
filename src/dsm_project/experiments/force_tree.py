"""Small prespecified tree grid on development wells; reserved wells excluded."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from dsm_project.algorithms.student.regression_tree import SquaredErrorTreeRegressor
from dsm_project.config import load_config
from dsm_project.data.force import FEATURES
from dsm_project.experiments.force_linear import run_experiment as run_linear


def run_experiment():
    """Select by pooled validation MAE; retain all candidates and comparisons.

    Same training sample as linear regression. Ties favor shallower trees, then
    larger leaves. Selection scores are tuning evidence, not unbiased estimates
    of the selected model's future performance. No feature scaling is needed.
    """
    base = run_linear()
    training, validation = base["training"], base["validation"]
    X, y = training.loc[:, FEATURES], training.DTC
    Xv, yv = validation.loc[:, FEATURES], validation.DTC
    models, candidates = {}, []
    for depth in (2, 3, 4):
        for leaf_size in (50, 100):
            name = f"tree_depth{depth}_leaf{leaf_size}"
            model = SquaredErrorTreeRegressor(depth, leaf_size).fit(X, y)
            prediction = model.predict(Xv)
            models[name] = model
            candidates.append(
                {
                    "model": name,
                    "max_depth": depth,
                    "min_samples_leaf": leaf_size,
                    "leaves": model.n_leaves_,
                    "actual_depth": model.depth_,
                    "training_mae": mean_absolute_error(y, model.predict(X)),
                    "validation_mae": mean_absolute_error(yv, prediction),
                    "validation_rmse": root_mean_squared_error(yv, prediction),
                }
            )
    ranked = sorted(
        candidates,
        key=lambda c: (c["validation_mae"], c["max_depth"], -c["min_samples_leaf"]),
    )
    selected = ranked[0]
    models["selected_tree"] = models[selected["model"]]
    models["tree_reference"] = DecisionTreeRegressor(
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        criterion="squared_error",
        random_state=577,
    ).fit(X, y)
    metrics, per_well = dict(base["report"]["metrics"]), []
    for name in ("selected_tree", "tree_reference"):
        prediction = models[name].predict(Xv)
        metrics[name] = {
            "mae": mean_absolute_error(yv, prediction),
            "rmse": root_mean_squared_error(yv, prediction),
        }
        for well, group in validation.assign(prediction=prediction).groupby("WELL"):
            per_well.append(
                {
                    "model": name,
                    "well": well,
                    "rows": len(group),
                    "mae": mean_absolute_error(group.DTC, group.prediction),
                    "rmse": root_mean_squared_error(group.DTC, group.prediction),
                }
            )
    report = {
        key: base["report"][key]
        for key in (
            "source_sha256",
            "sample_ids_sha256",
            "training_rows",
            "validation_rows",
            "training_wells",
            "validation_wells",
            "seed",
            "rows_per_training_well",
        )
    }
    report.update(
        {
            "evaluation": "development tuning only; final test not scored",
            "selection_rule": (
                "minimum pooled validation MAE; ties: lower depth then larger leaves"
            ),
            "selected_candidate": selected,
            "candidates": candidates,
            "metrics": metrics,
            "reference_mean_absolute_prediction_gap": float(
                np.mean(
                    np.abs(
                        models["selected_tree"].predict(Xv)
                        - models["tree_reference"].predict(Xv)
                    )
                )
            ),
            "reference_max_prediction_gap": float(
                np.max(
                    np.abs(
                        models["selected_tree"].predict(Xv)
                        - models["tree_reference"].predict(Xv)
                    )
                )
            ),
            "reference_note": (
                "Float precision and tied splits can differ; "
                "compare analytic tests and metrics, "
                "not exact field equality."
            ),
        }
    )
    per_well = pd.concat([base["per_well"], pd.DataFrame(per_well)], ignore_index=True)
    destination = load_config("configs/force.toml").root / "artifacts/force_tree"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(candidates).to_csv(destination / "candidates.csv", index=False)
    per_well.to_csv(destination / "per_well.csv", index=False)
    return {
        "report": report,
        "training": training,
        "validation": validation,
        "models": {**base["models"], **models},
        "per_well": per_well,
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment()["report"], indent=2))
