# Product contract: FORCE well interpretation

Owner: Liam Mora. Updated November 26, 2026.
This contract defines the intended product. Model experiments exist; the
FORCE-specific artifact, API, interface and operational controls below still
require implementation and verification. Template scaffolding is not evidence
that those requirements have already been met.

## Users and decisions

An analyst or student reviewing incomplete well records supplies gamma ray,
bulk density and neutron porosity measurements. The primary output is a
provisional compressional sonic-slowness estimate in microseconds per foot.
The supporting output is an exploratory sandstone-label score and binary flag.
The purpose is to study reconstruction and screening, with human review of
errors and limits. There is no prediction horizon: this estimates a measurement
at the same logged interval rather than forecasting future production.

No oil-presence, reserve, commercial drilling, autonomous geological or safety
decision is supported. A sandstone label is not evidence of hydrocarbons.
Success requires reproducible evidence and correct handling of unsupported
requests, not a predetermined score or an operational accuracy guarantee.

## Dataset and prediction boundary

Use only the audited FORCE 2020 training archive described in DATASET_CARD.md.
The three named inputs are GR (gAPI), RHOB (g/cm3), NPHI (fraction).
They must be measured and available at prediction time. DTC, lithology,
formation, confidence labels, well identity and depth are not predictors.
Well/depth identity supports offline splitting and diagnostics only.

Require all three finite numerical inputs with correct units. Reject missing
values, unexpected fields and unsupported values rather than impute or clip.
The service must map named inputs to the fixed model order GR, RHOB, NPHI.
For each task, store marginal minima/maxima from its final training sample;
refuse predictions outside those bounds. Negative NPHI is not automatically
invalid if it is inside that learned envelope. Marginal support does not
establish joint support, physical plausibility or accurate prediction.

## Task-algorithm matrix

| Family | Project role and disposition | Evidence or qualification |
| --- | --- | --- |
| Linear regression | Custom continuous DTC comparison | Full-MSE gradient descent; notebook 01, analytic/reference tests |
| Logistic regression | Custom sandstone-versus-other screening | Stable unregularized binary loss; notebook 02, gradient/reference tests |
| Decision trees | Custom squared-error DTC tree, selected candidate | Depth 4, leaf minimum 50; notebook 03, split/reference tests |
| Dimensionality reduction | Supporting library PCA | Training-only standardization/full SVD; notebook 04; not a fourth custom method |
| Nearest neighbors | Deferred, mathematically applicable | Additional modeling assumption outside the approved bounded comparison |
| Support vector machines/kernels | Deferred, mathematically applicable | Possible alternatives, not needed for the selected investigation |
| Ensembles | Deferred, mathematically applicable | Additional prediction complexity outside the current core scope |
| Neural networks | Deferred, mathematically applicable | No additional approved question justifies this expansion |
| Clustering | No separate clustering investigation | PCA addresses the chosen representation question; clusters would need their own interpretation |

Notebook 05 brings the fixed candidates together in robustness/failure analysis.
Library baselines, preprocessing and reference estimators remain distinguished
from custom implementations. The broader applicable-algorithm wording remains
an open compliance item in REQUIREMENTS_AUDIT.md; deferred is not incompatible.

## Evaluation contract

EVALUATION_PLAN.md is the authoritative pre-test protocol, merged in ca555b6
(decision commit 06e82d4). Twenty manifest wells are reserved. Final fitting
uses the other 78, capped at 250 eligible observations per well and task with
seed 577. No reserved targets influence fitting, preprocessing or support bounds.

The shallow tree is selected from development evidence. Retain linear,
mean/median baselines and PCA2-linear comparisons; do not select a new winner
using test scores. Primary regression metric is pooled MAE, with RMSE, bias,
per-well and rock-type evidence. The older config/adapter RMSE declarations
must be reconciled with this actual MAE selection during final-runner work.
Classification retains a prior baseline and logistic threshold 0.5, reporting
confusion counts, precision/recall, balanced accuracy and probability/ranking
metrics. No probability calibration is claimed.

Primary final metrics include all task-eligible rows, even outside service
support. Report supported/unsupported coverage and errors separately to avoid
selectively favorable reporting. Repeated development splits overlap; they
are descriptive sensitivity, not independent tests or confidence intervals.

If tree test MAE does not beat the median baseline, preserve that finding and
do not describe the product as validated reconstruction. Do not switch models
after observing it. Even improvement supports only an educational demonstration;
no domain-established acceptable error tolerance exists here. Logistic screening
remains exploratory because development recall was only about 0.22–0.38.

## System interfaces and acceptance requirements

| Boundary | Required behavior | Current state |
| --- | --- | --- |
| Ingestion | Check archive hash, schema, invalid values and identities | Implemented FORCE adapter and synthetic tests |
| Development | Whole-well fitting and comparison; reproducible outputs | Implemented dedicated experiment modules |
| Final training | Separate fitting from test scoring; save immutable candidates and provenance | Pending dedicated runner; do not use the template row-split training command |
| Artifact | Source/config/lockfile/data hashes, well/sample identities, parameters, transformations, support bounds and model hash | Template infrastructure requires FORCE adaptation and round-trip tests |
| Prediction API | Named measurements, explicit validation/support refusal, request and model identity, units and task outputs | Generic template API exists; FORCE contract pending |
| User interface | Three labeled inputs; estimates, score/flag, limitations and clear refusal messages | Generic template page requires adaptation |
| Deployment | Local demonstration using the exact evaluated artifact | No public production deployment claimed |
| Operations | Structured events, health/version checks, error/latency/unsupported-input counts and rollback | Exercise and FORCE evidence pending |

The primary sonic output must be finite and positive; reject invalid outputs
without silent correction. Classification must distinguish a model score from
calibrated confidence, and display the fixed threshold and label definition.
Show the artifact identity so results can be traced. Never load untrusted
serialized model files. Do not silently retrain when starting the service.

Existing development reproduction commands include
`uv run --locked python scripts/check_force_data.py` and
`uv run --locked python -m dsm_project.experiments.force_robustness`.
The final training and serving commands will be documented only after the
FORCE implementation is verified; generic Wine results do not satisfy this contract.

## Responsible use and operations boundary

Do not log credentials or raw submitted measurement records. Log request/model
identity, outcome, row count and duration; aggregate rejection/error signals.
An unavailable or invalid artifact must prevent prediction. Rollback must load
a previously verified compatible artifact and pass health/prediction checks.
OPERATIONS.md must record thresholds, a bounded failure exercise and recovery
evidence before release; this contract does not claim those exercises occurred.

Public well identifiers are scientific grouping labels, not evidence of personal
privacy risk in the selected fields. Do not add confidential field records to
this public project. Geological/source coverage is uneven, expert labels may
be imperfect, and complete-case evaluation selects a limited population.
No demographic fairness claim is made from these non-demographic measurements.
Users must review these limits; estimates cannot replace original measurements.
