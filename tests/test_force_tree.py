"""Exercise the development runner using synthetic wells, without field data."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dsm_project.experiments import force_tree


def test_runner_retains_grid_and_training_only_leaf_predictions(monkeypatch, tmp_path):
    # A constant training target makes validation-target leakage immediately visible.
    training = pd.DataFrame(
        {"GR": np.arange(220.0), "RHOB": 2.0, "NPHI": 0.2, "DTC": 7.0, "WELL": "train"}
    )
    validation = pd.DataFrame(
        {
            "GR": [0.0, 100.0],
            "RHOB": 2.0,
            "NPHI": 0.2,
            "DTC": [100.0, 200.0],
            "WELL": "validation",
        }
    )
    metadata = dict(
        source_sha256="fixture",
        sample_ids_sha256="fixture",
        training_rows=220,
        validation_rows=2,
        training_wells=["train"],
        validation_wells=["validation"],
        seed=577,
        rows_per_training_well=250,
        metrics={},
    )
    monkeypatch.setattr(
        force_tree,
        "run_linear",
        lambda: dict(
            training=training,
            validation=validation,
            report=metadata,
            models={},
            per_well=pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        force_tree, "load_config", lambda _: SimpleNamespace(root=tmp_path)
    )
    result = force_tree.run_experiment()
    assert len(result["report"]["candidates"]) == 6
    assert result["report"]["selected_candidate"]["model"] == "tree_depth2_leaf100"
    assert result["report"]["metrics"]["selected_tree"]["mae"] == 143.0
    assert (tmp_path / "artifacts/force_tree/candidates.csv").is_file()
    assert result["models"]["selected_tree"].tree_.prediction == 7.0
