"""Evaluate saved fixed candidates on reserved wells; never fit or select models."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import FEATURES, LITHOLOGY, eligible_rows
from dsm_project.experiments.force_final import (
    MODEL_NAMES,
    file_hash,
    load_candidates,
    support_mask,
)
from dsm_project.experiments.force_logistic import classification_metrics
from dsm_project.experiments.force_robustness import regression_metrics


def score_reserved(frame, models, metadata):
    """Produce metrics and row predictions using only reserved task-eligible rows."""
    reserved = set(metadata["reserved_wells"])
    development = set(metadata["development_wells"])
    if not reserved or reserved & development:
        raise ValueError("Reserved/development identities must be nonempty/disjoint")
    if set(frame.WELL.unique()) != reserved | development:
        raise ValueError("Dataset well identities differ from artifact")
    if set(models) != set(MODEL_NAMES) or any(
        set(models[task]) != names for task, names in MODEL_NAMES.items()
    ):
        raise ValueError("Unexpected candidate set")
    if metadata["classification_threshold"] != 0.5:
        raise ValueError("Frozen classification threshold must be 0.5")
    if metadata["selected_regressor"] != "tree":
        raise ValueError("Frozen selected regressor must remain tree")
    heldout = frame.loc[frame.WELL.isin(reserved)].copy()
    coverage, scores, predictions = [], [], {}
    for task, candidates in models.items():
        valid = eligible_rows(heldout, task).reset_index(drop=True)
        if valid.empty:
            raise ValueError(f"No eligible reserved rows for {task}")
        supported = support_mask(
            valid.loc[:, FEATURES], metadata["tasks"][task]["support"]
        )
        coverage.append(
            {
                "task": task,
                "raw_rows": len(heldout),
                "raw_wells": len(reserved),
                "eligible_rows": len(valid),
                "eligible_wells": valid.WELL.nunique(),
                "excluded_rows": len(heldout) - len(valid),
                "supported_rows": int(supported.sum()),
                "unsupported_rows": int((~supported).sum()),
                "zero_eligible_wells": sorted(reserved - set(valid.WELL)),
            }
        )
        target = "DTC" if task == "regression" else "sandstone"
        saved = valid[["observation_id", "WELL", "DEPTH_MD", LITHOLOGY, target]].copy()
        saved["supported"] = supported
        groups = [
            ("pooled", "all", np.ones(len(valid), dtype=bool)),
            ("support", "inside", supported),
            ("support", "outside", ~supported),
        ]
        for column, kind in (("WELL", "well"), (LITHOLOGY, "rock")):
            for label, subset in valid.groupby(column, dropna=False, sort=True):
                mask = valid.index.isin(subset.index)
                groups.append((kind, "missing" if pd.isna(label) else str(label), mask))
        for name, model in candidates.items():
            X = valid.loc[:, FEATURES]
            if task == "regression":
                prediction = np.asarray(model.predict(X), dtype=float)
            else:
                if not np.array_equal(model.classes_, [0, 1]):
                    raise ValueError("Expected binary class order [0, 1]")
                prediction = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
            if prediction.shape != (len(valid),) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions for {task}/{name}")
            saved[name] = prediction
            for kind, label, mask in groups:
                subset = valid.loc[mask]
                row = {
                    "task": task,
                    "model": name,
                    "kind": kind,
                    "slice": label,
                    "rows": len(subset),
                    "wells": subset.WELL.nunique(),
                }
                if len(subset):
                    y, p = subset[target].to_numpy(), prediction[mask]
                    if task == "regression":
                        row.update(regression_metrics(y, p))
                        row["nonpositive_predictions"] = int((p <= 0).sum())
                    else:
                        values = classification_metrics(y.astype(int), p)
                        matrix = values.pop("confusion_matrix")
                        row.update(values)
                        row.update(
                            dict(
                                zip(
                                    ("tn", "fp", "fn", "tp"),
                                    np.asarray(matrix).ravel().tolist(),
                                    strict=True,
                                )
                            )
                        )
                        row["prevalence"] = float(np.mean(y))
                        row["positive_flag_rate"] = float(np.mean(p >= 0.5))
                scores.append(row)
        predictions[task] = saved
    metrics = pd.DataFrame(scores)
    well = metrics.query("task == 'regression' and kind == 'well'")
    paired = well.pivot(index="slice", columns="model", values="mae")
    differences = {}
    for comparison in ("linear", "median_baseline"):
        delta = paired.tree - paired[comparison]
        differences[comparison] = {
            "wells": len(delta),
            "tree_better_wells": int((delta < 0).sum()),
            "median_difference": float(delta.median()),
            "minimum_difference": float(delta.min()),
            "maximum_difference": float(delta.max()),
        }
    pooled = metrics.query("task == 'regression' and kind == 'pooled'").set_index(
        "model"
    )
    summary = {
        "final_test_scored": True,
        "selected_regressor": "tree",
        "tree_beats_median_pooled_mae": bool(
            pooled.loc["tree", "mae"] < pooled.loc["median_baseline", "mae"]
        ),
        "paired_well_mae_differences": differences,
        "equal_weight_well_mae": well.groupby("model").mae.mean().to_dict(),
        "interpretation": "fixed held-out evaluation; no model reselection",
    }
    return {
        "metrics": metrics,
        "coverage": coverage,
        "predictions": predictions,
        "summary": summary,
    }


def evaluate_artifact(artifact, output):
    """Require clean compatible source and reserve an output before test scoring."""
    config = load_config("configs/force.toml")
    root = config.root
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Evaluation destination exists; preserve the previous run"
        )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain"):
        raise ValueError("Commit evaluator changes before final scoring")
    revision = git("rev-parse", "HEAD")
    models, metadata = load_candidates(artifact)
    provenance = metadata["provenance"]
    for name, digest in provenance["files_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Training source/configuration changed: {name}")
    manifest = json.loads(
        (root / config.split["reserved_manifest"]).read_text(encoding="utf-8-sig")
    )
    if set(manifest["reserved_test_wells"]) != set(metadata["reserved_wells"]):
        raise ValueError("Reserved manifest differs from artifact")
    if manifest["source_sha256"] != provenance["archive_sha256"]:
        raise ValueError("Dataset identity differs from artifact")
    output.mkdir(parents=True, exist_ok=False)
    run = {
        "started_at": datetime.now(UTC).isoformat(),
        "evaluation_revision": revision,
        "training_revision": provenance["source_revision"],
        "artifact_metadata_sha256": file_hash(Path(artifact) / "metadata.json"),
        "artifact_models_sha256": metadata["models_sha256"],
        "archive_sha256": provenance["archive_sha256"],
        "evaluator_sha256": file_hash(Path(__file__)),
    }
    (output / "STARTED.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    result = score_reserved(load_project_dataset(config).observations, models, metadata)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != revision:
        raise ValueError("Source changed during evaluation; preserve incomplete run")
    result["metrics"].to_csv(output / "metrics.csv", index=False)
    for task, table in result["predictions"].items():
        table.to_csv(output / f"{task}_predictions.csv", index=False)
    report = {
        **run,
        **result["summary"],
        "coverage": result["coverage"],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    hashes = {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}
    (output / "COMPLETE.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Saved final evaluation: {evaluate_artifact(args.artifact, args.output)}")


if __name__ == "__main__":
    main()
