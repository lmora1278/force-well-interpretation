# Dataset card: FORCE 2020 training archive

Owner: Liam Mora. Updated November 26, 2026.

## Identity and provenance

- Distribution: FORCE 2020 well-log/lithology competition training partition.
- Publisher/source: FORCE competition release maintained in the bolgebrygg
  repository; original government well-data provenance and acknowledgments
  are described by the source release. This project does not claim ownership.
- [Official repository](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition).
- [Archived release](https://zenodo.org/records/4351156).
- [Training archive](https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/train.zip).
- Source terms recorded in the project provenance: NLOD 2.0. Preserve the
  release's attribution and applicable terms; the code's MIT license does
  not replace the data license.
- Local archive verified November 19, 2026. The exact original download
  timestamp was not recorded; verification date is not asserted as retrieval date.
- SHA256: `e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9`.
- Local location: `data/local/force-train.zip`; semicolon-delimited CSV in ZIP.

The audited archive contains 1,170,511 rows, 98 wells, 29 original columns and
12 lithology codes. The broader release's 118 wells are not the size of this
project's archive. The adapter selects seven source fields; additional source
columns are deliberately not used as predictors. No external database is needed.

## Observation and sampling

One row is a measurement at a depth in a well. Neighboring depths are dependent;
rows are not independent experiments. Wells may also share geological conditions.
This is an existing collection of logged wells, not a randomized sample of all
oil-industry settings. No acquisition-date-based temporal split is supported by
the selected schema. This is reconstruction at an interval, not future forecasting.

The fixed manifest reserves 20 whole wells for final evaluation. Development
uses 78 others; the original split assigns 58 training and 20 validation wells.
Splitting precedes complete-case filtering. Models use at most 250 eligible
training rows per well with seed 577. Validation uses all eligible rows.
Final fitting on 78 wells is specified in EVALUATION_PLAN.md and has not yet run.

| Original development partition | Wells | Raw rows | Regression eligible | Classification eligible |
| --- | ---: | ---: | ---: | ---: |
| Training | 58 | 706823 | 413532 | 431307 |
| Validation | 20 | 240136 | 135154 | 145345 |

These are the recorded seed-577 development counts, not final-test statistics.
The held-out wells have not been scored. Repeated splits 577–581 overlap and
cannot be counted as five independent datasets.

## Schema, units and availability

| Field | Role | Unit or coding | Available to predictor? |
| --- | --- | --- | --- |
| WELL | Group/split identity | Source well identifier | No |
| DEPTH_MD | Depth and duplicate identity | Measured depth, metres | No |
| GR | Measured gamma ray | gAPI | Yes |
| RHOB | Measured bulk density | g/cm3 | Yes |
| NPHI | Measured neutron porosity | Fraction (m3/m3), not percent | Yes |
| DTC | Regression response | Compressional sonic slowness, microseconds/foot | No |
| FORCE_2020_LITHOFACIES_LITHOLOGY | Expert interpretation and diagnostic slice | Source integer code | No |
| sandstone (derived) | Classification response | 1 for 30000; 0 for other observed codes; missing stays missing | No |
| observation_id (derived) | Reproducibility identifier | Well plus full-precision depth string | No |

Code 65030 (mixed sandstone/shale) is negative under this specific binary
definition. The classifier is not a detector of any sand content. Unused source
formation, confidence and other interpretation columns must not leak into inputs.
All three measured predictors must already exist when a prediction is requested.

## Validation and missingness

The adapter verifies the archive fingerprint before parsing. It rejects missing
required columns, empty tables, absent/blank well identifiers, absent depth,
duplicate well/depth pairs, invalid numeric text, infinities, numeric sentinels
(-999, -999.25, -9999) and unknown lithology codes. It preserves ordinary missing
measurements for task-specific reporting rather than inventing observations.

Regression eligibility requires GR, RHOB, NPHI and DTC. Classification requires
the three predictors and a lithology label, not DTC. Missing lithology stays
missing rather than becoming class zero. No target is imputed. Finite extremes,
including negative NPHI, are retained; these checks do not prove every retained
value is physically plausible. No arbitrary clipping is performed.

Task eligibility and the planned serving envelope are distinct. Final evaluation
must score every eligible row, then additionally report training-envelope coverage
and subgroup metrics. Source measurements are never altered to improve test scores.

## Access and reproducibility

No account or secret is needed for the public archive. From the source repository,
download `lithology_competition/data/train.zip` and place it in the project as
`data/local/force-train.zip` without extracting or editing it. Check it from
the project root with:

```powershell
Get-FileHash -LiteralPath data/local/force-train.zip -Algorithm SHA256
uv run --locked python scripts/check_force_data.py
```

The fingerprint must match the value above and configs/reserved_wells.json.
If the upstream file changes, stop and investigate; do not edit the expected
hash merely to bypass the check. The existing loader validates the file and
the script reports development coverage without evaluating reserved performance.

Raw data, generated artifacts and the environment are Git-ignored. Commit code,
configurations, citations, synthetic fixtures and appropriately reviewed aggregate
evidence. GitHub is not the backup for the ignored local archive or model files;
the data remain recoverable from the cited source if its matching archive exists.
Final reproducibility/release evidence must include acquisition instructions and
model identity rather than assume a reviewer has local artifacts.

CI uses small synthetic measurements in tests/test_force_data.py and related
tests, including a synthetic ZIP for adapter checks. These are not sampled field
records. Full local-data notebook checks skip when the audited archive is absent;
passing lightweight CI alone therefore does not establish full-data reproduction.

## Representativeness and limitations

Complete-case selection can exclude systematically different intervals. We test
reconstruction where DTC is observed; genuinely missing DTC has no observed truth
for direct scoring. Expert lithology labels are interpretations, not infallible
ground truth. Unequal well lengths, rare rock labels and acquisition differences
limit pooled summaries. Well-level splitting reduces adjacent-depth leakage but
does not guarantee independence across wells or transfer to other basins.

The selected fields describe public geological records; no person-level labels
are used. Do not incorporate confidential operational logs or credentials into
this public repository. This dataset does not establish oil presence, reserves,
safe drilling choices, future production or reliable operation on new instruments,
units or populations. See PRODUCT_CONTRACT.md for the demonstration boundary.
