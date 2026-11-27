"""Exercise frozen fitting boundaries and trusted local artifact integrity."""

import json

import numpy as np
import pandas as pd
import pytest

from dsm_project.data.force import FEATURES, LITHOLOGY, validate_measurements
from dsm_project.experiments.force_final import (
    fit_candidates,
    load_candidates,
    save_candidates,
    support_mask,
)


def measurements():
    rng = np.random.default_rng(91)
    rows = []
    for well in range(4):
        for depth in range(260):
            x = rng.normal(size=3)
            rows.append(
                {
                    "WELL": f"well-{well}",
                    "DEPTH_MD": depth,
                    "GR": x[0],
                    "RHOB": x[1],
                    "NPHI": x[2],
                    "DTC": 80 + 3 * x[0] - x[1],
                    LITHOLOGY: 30000 if depth % 2 else 65000,
                }
            )
    return validate_measurements(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def fitted():
    frame = measurements()
    models, metadata = fit_candidates(frame, ["well-3"])
    return frame, models, metadata


def test_reserved_values_and_input_order_cannot_change_fit(fitted):
    frame, models, metadata = fitted
    altered = frame.copy()
    reserved = altered.WELL == "well-3"
    altered.loc[reserved, [*FEATURES, "DTC"]] = 1e8
    altered.loc[reserved, "sandstone"] = 1
    other, details = fit_candidates(altered.sample(frac=1, random_state=13), ["well-3"])
    assert metadata == details
    probe = frame.loc[~reserved, FEATURES].iloc[:20]
    for task, candidates in models.items():
        for name, model in candidates.items():
            np.testing.assert_array_equal(
                model.predict(probe), other[task][name].predict(probe)
            )
    assert metadata["development_wells"] == ["well-0", "well-1", "well-2"]
    assert not metadata["final_test_scored"]
    for task in ("regression", "classification"):
        assert metadata["tasks"][task]["sample_rows"] == 750
        assert metadata["tasks"][task]["eligible_rows"] == 780


def test_round_trip_preserves_all_predictions_and_probabilities(fitted, tmp_path):
    frame, models, metadata = fitted
    output = save_candidates(models, metadata, tmp_path / "bundle")
    loaded, record = load_candidates(output)
    probe = frame.loc[frame.WELL != "well-3", FEATURES].iloc[:30]
    for task, candidates in models.items():
        for name, model in candidates.items():
            np.testing.assert_array_equal(
                model.predict(probe), loaded[task][name].predict(probe)
            )
            if task == "classification":
                np.testing.assert_array_equal(
                    model.predict_proba(probe), loaded[task][name].predict_proba(probe)
                )
    assert record["tasks"] == metadata["tasks"]
    with pytest.raises(FileExistsError):
        save_candidates(models, metadata, output)


@pytest.mark.parametrize("filename", ["models.joblib", "metadata.json"])
def test_corruption_rejected_before_deserialization(
    fitted, tmp_path, filename, monkeypatch
):
    _, models, metadata = fitted
    output = save_candidates(models, metadata, tmp_path / "bundle")
    with (output / filename).open("ab") as stream:
        stream.write(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("Deserialization occurred before checksum validation")

    monkeypatch.setattr("dsm_project.experiments.force_final.joblib.load", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        load_candidates(output)


def test_support_bounds_and_named_feature_order():
    bounds = {"GR": [0, 100], "RHOB": [1, 3], "NPHI": [-0.1, 0.5]}
    frame = pd.DataFrame(
        {"NPHI": [-0.1, 0.5, 0.2], "GR": [0, 100, 101], "RHOB": [1, 3, 2]}
    )
    assert support_mask(frame, bounds).tolist() == [True, True, False]
    for invalid in (
        frame.drop(columns="GR"),
        frame.assign(extra=1),
        frame.assign(GR=np.nan),
        frame.assign(GR=np.inf),
        frame.assign(GR="12"),
        frame.assign(GR=True),
    ):
        with pytest.raises(ValueError):
            support_mask(invalid, bounds)


def test_no_converged_artifact_from_failed_optimization(monkeypatch):
    from dsm_project.algorithms.student.logistic_regression import (
        BatchLogisticClassifier,
    )

    real_fit = BatchLogisticClassifier.fit

    def nonconverged(self, X, y):
        real_fit(self, X, y)
        self.converged_ = False
        return self

    monkeypatch.setattr(BatchLogisticClassifier, "fit", nonconverged)
    with pytest.raises(ValueError, match="did not converge"):
        fit_candidates(measurements(), ["well-3"])


@pytest.mark.parametrize("reserved", [[], ["well-3", "well-3"], ["absent"]])
def test_invalid_reserved_manifest_rejected(reserved):
    with pytest.raises(ValueError):
        fit_candidates(measurements(), reserved)


def test_incomplete_artifact_is_not_loadable(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(FileNotFoundError):
        load_candidates(tmp_path)


def test_force_metric_declarations_agree():
    from dsm_project.config import load_config

    assert (
        load_config("configs/force.toml").tasks["regression"]["metric"]
        == "mean_absolute_error"
    )
