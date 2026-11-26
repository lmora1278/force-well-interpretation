# Implementation plan: Reconstructing Missing Sonic Logs in Unseen Wells

Owner: Liam Mora

The authoritative definition is PROJECT_OVERVIEW.md, updated November 17, 2026. This plan records the implementation sequence.

## Project question

How reliably can gamma ray, bulk density, and neutron porosity measurements estimate compressional sonic slowness and reproduce sandstone labels in wells excluded from training, and where do these predictions fail?

The intended user is a student or analyst reviewing logged well intervals. The outputs support interpretation exercises, not oil discovery, reserves estimation, or drilling decisions. Rock labels are supplied expert interpretations, not infallible physical ground truth.

## Data contract

Use the official FORCE 2020 training archive, containing 1,170,511 measurements in 98 wells. Source: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition. Release: https://zenodo.org/records/4351156.

Archive SHA256: e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9.

Predictors: GR (gAPI), RHOB (g/cm3), NPHI (m3/m3). Regression target: DTC (microseconds per foot). Classification target: supplied lithology code 30000 versus other supplied codes, including mixed sandstone/shale. No well identifier, formation interpretation, confidence label, or withheld target enters the predictors.

Split by whole wells before model selection or fitting preprocessing. Preserve the 20 wells reserved in the feasibility audit. Use development-only tuning and bounded, reproducible training samples. Classification requires observed inputs and a label; regression additionally requires observed DTC. Report selection and coverage limitations separately for both tasks. Actual missing-DTC intervals have no observed target for scoring. No final-test evaluation before design freeze.

## Core implementation and useful additions

1. Custom linear regressor: scaled gradient descent, explicit objective and gradients, convergence checks, reusable fitted interface, analytic tests and trusted least-squares comparison. The template's existing exemplar does not count as a new project implementation.
2. Custom logistic classifier: stable binary loss and sigmoid, gradients, unregularized fitting, fixed 0.5 decision threshold, reusable interface, gradient and reference checks.
3. Custom shallow regression tree: implement squared-error split selection, recursive partitioning, stopping rules and leaf predictions. Verify hand-computable cases and trusted reference agreement. Compare with linear regression on identical well-level boundaries. A classification-tree extension remains optional.
4. PCA: examine redundancy and visualization in standardized inputs; test whether the projection helps interpret model limitations. Do not claim clusters or projections are geological truth.
5. Optional k-means or kNN only when it answers an additional useful question and does not displace required evidence.

## Requirements interpretation

The supplied syllabus explicitly requires at least two machine learning algorithms implemented as reusable functions or classes. It allocates 40% to functionality/implementation, 20% documentation, 20% testing, 10% examples/usability and 10% repository quality. It requires a public repository and presentation.

The original template separately asks for every applicable course algorithm and includes a broader software/product checklist. Those words are preserved in ASSIGNMENT.md and PROJECT_CHECKLIST.md. This plan follows a three-algorithm core-first implementation sequence; it does not assert that the wording difference is resolved or that deferred applicable families are scientifically incompatible. Confirm broader expectations before the final compliance sign-off. Add justified methods after completing the core, not to inflate model count.

## Evaluation and evidence

- Regression: training-median baseline, MAE and RMSE in source units, per-well residuals, comparisons with linear and nonlinear references.
- Classification: majority baseline, prevalence, balanced accuracy, precision/recall, average precision, confusion matrix and development-only threshold exploration; the retained threshold is 0.5. Scores must not be described as calibrated probabilities without calibration evidence.
- Uncertainty: respect well dependence. Repeated development splits and well-level summaries; no row-independent confidence interval pretending adjacent depths are independent.
- Failures: incomplete logs, outliers, rare rock types, mixed-label ambiguity, poor-performing wells, and measurements outside the supported range.
- Tests: analytic examples, finite-difference gradients, invalid inputs, train-only transformations, disjoint well splits, serialization, trusted-reference agreement and API behavior.

## Deliverables

Question-led notebooks for data/baselines, sonic estimation, sandstone-label classification, representation/error analysis, and the product demonstration. Reusable code belongs in the package. Include the dataset card, model card, product contract, provenance, operations exercise, working minimal interface, tests and CI, release identity and presentation. Adapt the original template rather than publish unchanged example results as project findings.

## Work sequence

1. Finalize source adapter, data validation, split manifest, CI fixture and baselines.
2. Complete and test custom linear regression; produce the first end-to-end case study.
3. Complete and test custom logistic regression; produce the second case study.
4. Implement and test the custom regression tree; complete the linear-versus-nonlinear comparison and supporting PCA analysis.
5. Resolve remaining applicable-method expectations; implement necessary additions.
6. Freeze selection, evaluate final wells, integrate the minimal interface, run operations and clean-clone checks.
7. Audit every rubric item, prepare the presentation and publish a tagged final release.

At each working session, save artifacts and record the next action. Once a new remote repository is confirmed, commit and push reviewed checkpoints. No old project repository history is part of this workspace.


## Pre-test decision record

EVALUATION_PLAN.md records the November 25 final-evaluation decisions.
REQUIREMENTS_AUDIT.md records remaining evidence and the unresolved broader
applicable-algorithm wording. These supplement, rather than replace, the approved
project question and the professor's original checklist.
