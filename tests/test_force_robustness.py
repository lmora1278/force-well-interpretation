"""Test reserved-well isolation, baselines and subgroup accounting."""

import numpy as np
import pandas as pd
import pytest

from dsm_project.data.force import FEATURES, LITHOLOGY, validate_measurements
from dsm_project.experiments.force_robustness import (
    evaluate_development,
    fit_fixed_regressors,
    regression_slices,
)


def measurements():
    rng = np.random.default_rng(577)
    rows = []
    for well in range(6):
        for depth in range(30):
            x = rng.normal(size=3)
            rows.append(
                {
                    "WELL": f"well-{well}",
                    "DEPTH_MD": float(depth),
                    "GR": x[0],
                    "RHOB": x[1],
                    "NPHI": x[2],
                    "DTC": 80 + 3 * x[0] + 2 * x[1],
                    LITHOLOGY: 30000 if depth % 2 else 65000,
                }
            )
    return validate_measurements(pd.DataFrame(rows))


def test_mean_and_median_use_training_targets_only():
    frame = measurements().iloc[:5].copy()
    frame["DTC"] = [1.0, 2.0, 3.0, 4.0, 100.0]
    models = fit_fixed_regressors(frame)
    X = frame.loc[:, FEATURES] + 1000
    np.testing.assert_allclose(models["mean_baseline"].predict(X), 22.0)
    np.testing.assert_allclose(models["median_baseline"].predict(X), 3.0)


def test_reserved_target_changes_do_not_change_any_development_score():
    frame = measurements()
    altered = frame.copy()
    altered.loc[altered.WELL == "well-5", "DTC"] = 1e8
    altered.loc[altered.WELL == "well-5", "sandstone"] = 1
    kwargs = dict(seeds=(577, 578), validation_wells=1, rows_per_well=25)
    first = evaluate_development(frame, ["well-5"], **kwargs)
    second = evaluate_development(altered, ["well-5"], **kwargs)
    for name in ("regression", "classification", "per_well", "rock_type"):
        pd.testing.assert_frame_equal(first[name], second[name])
    for split in first["split_manifests"]:
        assert "well-5" not in split["training_wells"] + split["validation_wells"]
        assert not set(split["training_wells"]) & set(split["validation_wells"])
    assert set(first["regression"].model) == {
        "mean_baseline",
        "median_baseline",
        "linear",
        "tree",
        "pca2_linear",
    }


def test_missing_lithology_is_retained_in_regression_slice_counts():
    frame = measurements().iloc[:4].copy()
    frame.loc[frame.index[0], LITHOLOGY] = np.nan
    rows = regression_slices(frame, frame.DTC.to_numpy() + 2, LITHOLOGY)
    assert sum(row["rows"] for row in rows) == 4
    assert "missing" in {row["slice"] for row in rows}
    assert all(row["mae"] == pytest.approx(2) for row in rows)
    assert all(row["bias"] == pytest.approx(2) for row in rows)


def test_repeated_seed_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        evaluate_development(measurements(), ["well-5"], seeds=(577, 577))
