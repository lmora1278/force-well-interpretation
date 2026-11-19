"""Scientific boundary checks using small synthetic measurements, not field data."""

import numpy as np
import pandas as pd
import pytest

from dsm_project.data.force import (
    LITHOLOGY,
    development_partitions,
    eligible_rows,
    validate_measurements,
)


@pytest.fixture
def measurements():
    return pd.DataFrame(
        {
            "WELL": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "DEPTH_MD": [1, 2] * 4,
            "GR": [50.0] * 8,
            "RHOB": [2.4] * 8,
            "NPHI": [0.2] * 8,
            "DTC": [90.0] * 8,
            LITHOLOGY: [30000, 65030, 65000, np.nan] * 2,
        }
    )


def test_missing_targets_are_not_invented(measurements):
    measurements.loc[0, "DTC"] = np.nan
    measurements.loc[1, "NPHI"] = np.nan
    result = validate_measurements(measurements)
    assert len(eligible_rows(result, "regression")) == 6
    assert len(eligible_rows(result, "classification")) == 5
    assert result.loc[1, "sandstone"] == 0  # Mixed sandstone/shale is not sandstone.
    assert pd.isna(result.loc[3, "sandstone"])
    assert pd.isna(result.loc[0, "DTC"])


@pytest.mark.parametrize(
    "column,value",
    [
        ("GR", np.inf),
        ("DTC", -999.25),
        ("WELL", ""),
        ("DEPTH_MD", np.nan),
        (LITHOLOGY, 123),
    ],
)
def test_invalid_measurements_fail(measurements, column, value):
    measurements.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_measurements(measurements)


def test_duplicate_depth_and_missing_schema_fail(measurements):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_measurements(pd.concat([measurements, measurements.iloc[:1]]))
    with pytest.raises(ValueError, match="Missing FORCE"):
        validate_measurements(measurements.drop(columns="DTC"))


def test_finite_extremes_are_retained(measurements):
    measurements.loc[0, "NPHI"] = -0.03
    assert validate_measurements(measurements).loc[0, "NPHI"] == -0.03


def test_well_split_is_disjoint_and_order_independent(measurements):
    frame = validate_measurements(measurements)
    train, validation = development_partitions(frame, ["D"], validation_wells=1)
    assert set(train.WELL).isdisjoint(validation.WELL)
    assert "D" not in set(train.WELL) | set(validation.WELL)
    assert len(train) + len(validation) == 6
    other_train, other_validation = development_partitions(
        frame.sample(frac=1, random_state=3),
        ["D"],
        validation_wells=1,
    )
    assert set(train.observation_id) == set(other_train.observation_id)
    assert set(validation.observation_id) == set(other_validation.observation_id)


def test_bad_manifest_and_split_fail(measurements):
    for reserved in ([], ["D", "D"], ["UNKNOWN"]):
        with pytest.raises(ValueError):
            development_partitions(measurements, reserved, validation_wells=1)
    with pytest.raises(ValueError):
        development_partitions(measurements, ["D"], validation_wells=3)


def test_archive_adapter_checks_fingerprint_and_declares_tasks(tmp_path, measurements):
    import hashlib
    import json
    from dataclasses import replace

    from dsm_project.config import load_config
    from dsm_project.data import load_project_dataset

    archive = tmp_path / "train.zip"
    measurements.to_csv(
        archive,
        sep=";",
        index=False,
        compression={"method": "zip", "archive_name": "train.csv"},
    )
    manifest = tmp_path / "reserved.json"
    manifest.write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "reserved_test_wells": ["D"],
            }
        )
    )
    original = load_config("configs/force.toml")
    config = replace(
        original,
        dataset={**original.dataset, "path": str(archive)},
        split={**original.split, "reserved_manifest": str(manifest)},
    )
    dataset = load_project_dataset(config)
    assert len(dataset.observations) == 8
    assert dataset.tasks["regression"].feature_columns == ("GR", "RHOB", "NPHI")
    assert dataset.observations.observation_id.is_unique
    manifest.write_text(json.dumps({"source_sha256": "wrong"}))
    with pytest.raises(ValueError, match="checksum"):
        load_project_dataset(config)


def test_template_row_split_cannot_run_on_force():
    from sklearn.dummy import DummyRegressor

    from dsm_project.config import load_config
    from dsm_project.data import load_project_dataset
    from dsm_project.experiments.supervised import evaluate_supervised_estimator

    example = load_project_dataset(load_config("configs/example.toml"))
    with pytest.raises(ValueError, match="whole-well"):
        evaluate_supervised_estimator(
            example,
            load_config("configs/force.toml"),
            task_name="regression",
            estimator_name="dummy",
            estimator=DummyRegressor(),
        )
