# FORCE data milestone

Verified November 19, 2026 against the SHA256 recorded in
`configs/reserved_wells.json`. The original archive is stored at
`data/local/force-train.zip`, which Git ignores.

## Verified archive and development coverage

The source contains 1,170,511 depth observations from 98 wells. Twenty wells
remain reserved for final evaluation. No model was fitted or scored in this check.

| Partition | Wells | All rows | Complete regression rows | Complete classification rows |
| --- | ---: | ---: | ---: | ---: |
| Training | 58 | 706,823 | 413,532 | 431,307 |
| Validation | 20 | 240,136 | 135,154 | 145,345 |

Both tasks use GR, RHOB and NPHI. Regression also requires an observed DTC;
classification requires an observed lithology label. Sandstone is code 30000;
mixed sandstone/shale is in the other class. Missing labels remain missing.

Well assignment precedes task-specific filtering. Sorted development well names
are shuffled with NumPy's local generator and seed 577; the first 20 are
validation wells. This split is independent of row order. Final-test well
identities are preserved from the assessment manifest.

## Data rules

- Reject missing well/depth identifiers, duplicate well/depth observations,
  missing required columns, invalid numeric text, infinities, known numeric
  sentinels, and unknown lithology codes.
- Preserve missing measurements in the loaded table and report their coverage.
- Use complete predictors and the relevant target for each task. Do not impute
  missing targets or silently turn missing lithology into the negative class.
- Retain finite extremes, including negative neutron porosity, for scientific
  review rather than applying arbitrary clipping.
- Do not use well IDs, depth, lithology or the regression target as predictors.

## Reproduce

From the project root:

```powershell
uv run --locked python scripts/check_force_data.py
uv run --locked pytest tests/test_force_data.py -q
```

The check loads and validates the archive, then reports development coverage.
It does not evaluate final-test performance. The original template's supervised
row-splitting runner deliberately rejects FORCE until a dedicated whole-well
experiment runner is implemented. Wine demonstrations continue to work.

The automated unit tests use synthetic measurements, not sampled FORCE records.
The next milestone is a bounded training configuration and tested linear
regression following Lecture 10, with preprocessing learned on training only.
