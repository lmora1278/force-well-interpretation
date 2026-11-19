"""FORCE measurements and development splits with whole-well isolation."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)

FEATURES = ("GR", "RHOB", "NPHI")
LITHOLOGY = "FORCE_2020_LITHOFACIES_LITHOLOGY"
REQUIRED = ("WELL", "DEPTH_MD", *FEATURES, "DTC", LITHOLOGY)
LITHOLOGY_CODES = {
    30000,
    65030,
    65000,
    80000,
    74000,
    70000,
    70032,
    88000,
    86000,
    99000,
    90000,
    93000,
}


def validate_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate structure; retain missing measurements for task-specific filtering.

    Reject invalid numeric text, infinities, sentinel values and duplicate depths.
    Missing targets are never imputed. Finite extremes are retained for review.
    """
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing FORCE columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED].copy()
    if result.empty:
        raise ValueError("FORCE data contains no observations")
    if result.WELL.isna().any() or result.WELL.astype(str).str.strip().eq("").any():
        raise ValueError("Well identifiers must be present")
    result["WELL"] = result.WELL.astype(str)
    for column in REQUIRED[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise")
        present = result[column].dropna()
        if not np.isfinite(present).all() or present.isin([-999, -999.25, -9999]).any():
            raise ValueError(f"Invalid measurement or sentinel in {column}")
    if result.DEPTH_MD.isna().any():
        raise ValueError("Depth must be present")
    if result.duplicated(["WELL", "DEPTH_MD"]).any():
        raise ValueError("Duplicate well/depth observations")
    if not set(result[LITHOLOGY].dropna()).issubset(LITHOLOGY_CODES):
        raise ValueError("Unknown lithology code")
    result["sandstone"] = result[LITHOLOGY].eq(30000).astype("Float64")
    result.loc[result[LITHOLOGY].isna(), "sandstone"] = pd.NA
    result["observation_id"] = (
        result.WELL + ":" + result.DEPTH_MD.map(lambda value: format(value, ".17g"))
    )
    return result


class ForceAdapter:
    """Read the audited semicolon-separated training archive without extracting it."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Check the archive fingerprint and declare fixed scientific tasks."""
        path = config.root / str(config.dataset["path"])
        manifest_path = config.root / str(config.split["reserved_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest["source_sha256"]:
            raise ValueError("FORCE archive checksum does not match audited source")
        frame = validate_measurements(
            pd.read_csv(path, sep=";", usecols=list(REQUIRED))
        )
        tasks = {
            "regression": TaskDefinition(
                "reconstruct sonic slowness",
                TaskKind.REGRESSION,
                FEATURES,
                "DTC",
                "root_mean_squared_error",
            ),
            "classification": TaskDefinition(
                "screen sandstone labels",
                TaskKind.CLASSIFICATION,
                FEATURES,
                "sandstone",
                "balanced_accuracy",
            ),
        }
        dataset = ProjectDataset(
            frame,
            "observation_id",
            tasks,
            DatasetProvenance(
                "FORCE 2020 training partition",
                str(config.dataset["source"]),
                str(config.dataset["license"]),
                "one depth measurement in a well",
            ),
        )
        dataset.validate()
        return dataset


def development_partitions(
    frame: pd.DataFrame,
    reserved_wells: list[str],
    *,
    validation_wells: int = 20,
    seed: int = 577,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude reserved wells, then partition whole development wells.

    Assign wells before target filtering, keeping both tasks' partitions aligned.
    Return training and validation only; final-test observations are not returned.
    """
    wells = set(frame.WELL.unique())
    if not reserved_wells or len(set(reserved_wells)) != len(reserved_wells):
        raise ValueError("Reserved wells must be nonempty and unique")
    if not set(reserved_wells).issubset(wells):
        raise ValueError("Reserved manifest contains wells absent from the data")
    available = np.array(sorted(wells - set(reserved_wells)))
    if not isinstance(validation_wells, int) or not 0 < validation_wells < len(
        available
    ):
        raise ValueError("Validation well count must leave training wells")
    order = np.random.default_rng(seed).permutation(available)
    valid = set(order[:validation_wells])
    train = set(order[validation_wells:])
    return frame[frame.WELL.isin(train)].copy(), frame[frame.WELL.isin(valid)].copy()


def eligible_rows(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    """Retain complete predictors and the requested observed target, without fitting."""
    targets = {"regression": "DTC", "classification": "sandstone"}
    if task not in targets:
        raise ValueError(f"Unknown FORCE task: {task}")
    return frame.dropna(subset=[*FEATURES, targets[task]]).copy()
