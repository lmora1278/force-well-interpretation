"""Synthetic-only checks of final scoring; field holdout is never opened here."""

import numpy as np
import pandas as pd
import pytest

from dsm_project.data.force import LITHOLOGY, validate_measurements
from dsm_project.experiments.force_evaluate import score_reserved
from dsm_project.experiments.force_final import MODEL_NAMES


class FixedModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        assert (X.GR >= 10).all(), "Development rows reached scoring"
        return np.full(len(X), 10.0)

    def predict_proba(self, X):
        self.predict(X)
        return np.tile([0.25, 0.75], (len(X), 1))

    def fit(self, *args, **kwargs):
        pytest.fail("Final evaluation must never fit")


def fixture():
    frame = validate_measurements(
        pd.DataFrame(
            {
                "WELL": ["dev", "test-a", "test-a", "test-b", "test-b"],
                "DEPTH_MD": [0, 0, 1, 0, 1],
                "GR": [0, 10, 11, 12, 13],
                "RHOB": [2.0] * 5,
                "NPHI": [0.2] * 5,
                "DTC": [999, 8, 12, 14, np.nan],
                LITHOLOGY: [65000, 30000, np.nan, 65000, 30000],
            }
        )
    )
    models = {
        task: {name: FixedModel() for name in names}
        for task, names in MODEL_NAMES.items()
    }
    bounds = {"GR": [0, 10], "RHOB": [1, 3], "NPHI": [0, 1]}
    metadata = {
        "reserved_wells": ["test-a", "test-b"],
        "development_wells": ["dev"],
        "classification_threshold": 0.5,
        "selected_regressor": "tree",
        "tasks": {task: {"support": bounds} for task in MODEL_NAMES},
    }
    return frame, models, metadata


def test_metrics_use_reserved_rows_and_preserve_unsupported_cases():
    result = score_reserved(*fixture())
    metric = result["metrics"]
    tree = metric.query("task == 'regression' and model == 'tree'")
    pooled = tree.query("kind == 'pooled'").iloc[0]
    assert pooled.rows == 3
    assert pooled.mae == pytest.approx(8 / 3)
    assert pooled.rmse == pytest.approx(np.sqrt(8))
    assert pooled.bias == pytest.approx(-4 / 3)
    assert tree.query("kind == 'rock'").rows.sum() == 3
    assert "missing" in set(tree.query("kind == 'rock'")["slice"])
    assert tree.query("kind == 'support' and slice == 'outside'").iloc[0].rows == 2
    assert result["summary"]["equal_weight_well_mae"]["tree"] == 3
    assert (
        result["summary"]["paired_well_mae_differences"]["linear"]["median_difference"]
        == 0
    )
    classifier = metric.query(
        "task == 'classification' and model == 'logistic' and kind == 'pooled'"
    ).iloc[0]
    assert classifier.tp == 2 and classifier.fp == 1 and classifier.fn == 0
    assert classifier.precision == pytest.approx(2 / 3)
    for predictions in result["predictions"].values():
        assert "dev" not in set(predictions.WELL)
        assert len(predictions) == 3


def test_empty_support_and_single_class_metrics_are_not_invented():
    frame, models, metadata = fixture()
    for record in metadata["tasks"].values():
        record["support"]["GR"] = [0, 1]
    result = score_reserved(frame, models, metadata)
    empty = result["metrics"].query("kind == 'support' and slice == 'inside'")
    assert (empty.rows == 0).all()
    assert empty.mae.isna().all()
    single = result["metrics"].query("task == 'classification' and kind == 'rock'")
    assert single.balanced_accuracy.isna().all()
    negative = single[single["slice"] == "65000.0"]
    assert not negative.empty
    assert negative.recall.isna().all()


def test_development_targets_cannot_change_results():
    frame, models, metadata = fixture()
    first = score_reserved(frame, models, metadata)
    frame.loc[frame.WELL == "dev", ["DTC", "sandstone"]] = -99999
    second = score_reserved(frame, models, metadata)
    pd.testing.assert_frame_equal(first["metrics"], second["metrics"])


def test_overlapping_identities_and_changed_policy_rejected():
    frame, models, metadata = fixture()
    metadata["development_wells"].append("test-a")
    with pytest.raises(ValueError, match="disjoint"):
        score_reserved(frame, models, metadata)
    metadata["development_wells"] = ["dev"]
    metadata["classification_threshold"] = 0.2
    with pytest.raises(ValueError, match="threshold"):
        score_reserved(frame, models, metadata)


def test_command_saves_complete_report_without_overwriting(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from dsm_project.experiments import force_evaluate as module

    frame, models, metadata = fixture()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text("{}")
    manifest = tmp_path / "reserved.json"
    manifest.write_text(
        json.dumps(
            {
                "reserved_test_wells": metadata["reserved_wells"],
                "source_sha256": "fixture",
            }
        )
    )
    metadata["provenance"] = {
        "files_sha256": {},
        "archive_sha256": "fixture",
        "source_revision": "training-fixture",
    }
    metadata["models_sha256"] = "synthetic-model-identity"
    monkeypatch.setattr(
        module,
        "load_config",
        lambda *args: SimpleNamespace(
            root=tmp_path, split={"reserved_manifest": "reserved.json"}
        ),
    )
    monkeypatch.setattr(module, "load_candidates", lambda path: (models, metadata))
    monkeypatch.setattr(
        module,
        "load_project_dataset",
        lambda config: SimpleNamespace(observations=frame),
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **kwargs: "" if "status" in args else "fixture-revision",
    )
    output = module.evaluate_artifact(artifact, tmp_path / "evaluation")
    report = json.loads((output / "report.json").read_text())
    assert report["final_test_scored"] is True
    assert report["coverage"][0]["eligible_rows"] == 3
    checksums = json.loads((output / "COMPLETE.json").read_text())
    assert all(
        module.file_hash(output / name) == digest for name, digest in checksums.items()
    )
    with pytest.raises(FileExistsError):
        module.evaluate_artifact(artifact, output)
