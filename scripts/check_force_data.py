"""Report FORCE development coverage without training or scoring a model."""

import json

from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.force import development_partitions, eligible_rows


def main() -> None:
    config = load_config("configs/force.toml")
    dataset = load_project_dataset(config)
    manifest = json.loads(
        (config.root / config.split["reserved_manifest"]).read_text(
            encoding="utf-8-sig"
        )
    )
    train, validation = development_partitions(
        dataset.observations,
        manifest["reserved_test_wells"],
        validation_wells=config.split["validation_wells"],
        seed=config.split["random_state"],
    )
    report = {
        "source_rows": len(dataset.observations),
        "source_wells": dataset.observations.WELL.nunique(),
        "reserved_test_wells": len(manifest["reserved_test_wells"]),
        "partitions": {},
    }
    for name, frame in (("training", train), ("validation", validation)):
        report["partitions"][name] = {
            "rows": len(frame),
            "wells": frame.WELL.nunique(),
            "missing_fraction": frame[["GR", "RHOB", "NPHI", "DTC"]]
            .isna()
            .mean()
            .to_dict(),
            "eligible_regression_rows": len(eligible_rows(frame, "regression")),
            "eligible_classification_rows": len(eligible_rows(frame, "classification")),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
