# Reconstructing Missing Sonic Logs in Unseen Wells

## A Machine-Learning Study Using FORCE 2020

**Student:** Liam Mora  
**Course:** INDE 577, Rice University  
**Format:** Solo final project  
**Updated:** November 17, 2026

This is the guiding project definition, superseding earlier topic and method proposals. PROJECT_PLAN.md records the implementation sequence; SESSION_CHECKPOINT.md records progress. Changes to the research questions, targets or core algorithms require an explicit scope decision.

## Primary research question

How accurately can gamma ray, bulk density, and neutron porosity measurements estimate compressional sonic slowness in wells excluded from training, compared with simple baseline predictions, and how does performance vary between wells and rock types?

## Model-comparison question

Does a shallow regression tree improve sonic-log reconstruction over linear regression in unseen wells, and is any improvement consistent across wells and rock types?

## Supporting research question

How effectively can the same three measurements distinguish sandstone-labelled intervals from other labelled rock types in unseen wells?

## Basic description and usefulness

The project investigates whether limited well-log measurements can support interpretation of incomplete well records. Regression estimates a sonic measurement; classification identifies intervals assigned a sandstone label. Both use the same three measured inputs and well-level evaluation boundaries.

The intended user is a student or analyst reviewing logged intervals. Outputs provide provisional sonic estimates and sandstone screening information. The study emphasizes transparent implementations, reproducibility, baselines, generalization to unseen wells, and error analysis.

Sonic reconstruction is evaluated against observed measurements deliberately withheld from model inputs. It does not establish accuracy wherever sonic measurements are genuinely missing. Classification measures agreement with expert interpretations, which can contain uncertainty and errors.

## Dataset and source

Use the official FORCE 2020 well-log and lithology dataset's training partition: **1,170,511 depth-indexed observations from 98 wells**. The broader release describes 118 wells across competition partitions; our starting archive contains 98.

- Official repository: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
- Dataset release: https://zenodo.org/records/4351156
- Training archive: https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/train.zip
- Audited archive SHA256: `e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9`

The source is a compressed CSV table. No external database server is necessary. Preserve source citations and applicable licence terms. The release identifies original government well data as distributed under NLOD 2.0. Original template attribution remains separate from data attribution.

## Observation, inputs and targets

One row represents measurements at a recorded depth in a well. Adjacent depths are dependent; the row count is not a count of independent experiments.

| Input | Field | Unit |
|---|---|---|
| Gamma ray | GR | gAPI |
| Bulk density | RHOB | g/cm3 |
| Neutron porosity | NPHI | m3/m3 |

**Regression target:** DTC, compressional sonic slowness in microseconds per foot.

**Classification target:** supplied lithology code 30000 (Sandstone) versus all other supplied codes, including mixed Sandstone/Shale. This identifies the dataset's sandstone label, not every interval containing any sand.

Well identity and depth support partitioning and diagnostics. Well identifiers, interpreted formations, label confidence and the withheld target are excluded from predictive inputs. Confirm units and quality rules during ingestion. Preserve original source values and document exclusions.

## Three core custom algorithms

| Algorithm | Project role | Implementation and verification |
|---|---|---|
| Linear regression | Estimate sonic slowness with a linear relationship | Implement objective, gradients, training, convergence checks and prediction. Verify analytic examples, gradients and trusted least-squares comparisons. |
| Logistic regression | Distinguish sandstone-labelled intervals from other labels | Implement stable sigmoid/log loss, gradients, training and decision behavior. Verify numerical stability, gradients and trusted classifier comparisons. |
| Shallow regression tree | Test whether nonlinear rules improve sonic reconstruction | Implement squared-error split selection, recursive partitioning, stopping rules and leaf predictions. Verify hand-computable splits, constant targets, repeated values, stopping behavior and trusted tree comparisons. |

The regression tree is a core custom implementation, not merely a library benchmark. Depth and minimum leaf size will be restricted using development evidence. Its value is the controlled linear-versus-nonlinear comparison on the primary target. The existing template exemplar does not count as a newly completed project implementation.

Trusted-library models provide verification and reference evidence and will be distinguished from custom code. A classification-tree extension is optional; it is not needed to establish the third algorithm.

## Supporting analysis

Use purposeful exploration and PCA to investigate measurement relationships, redundancy and failure patterns. PCA is a supporting representation tool, not one of the three core custom algorithms. Clustering, kNN, ensembles and other additions are outside the initial scope unless they answer a useful question or become necessary under clarified course requirements.

## Evaluation design

1. Separate training, validation and final-test data by whole wells before experimental sampling or fitting preprocessing. Preserve the 20 wells reserved during feasibility assessment.
2. Select preprocessing, thresholds and models using development data only. Use reproducible bounded training samples and identical boundaries for the regression comparison.
3. Classification requires observed inputs and labels; regression additionally requires observed DTC. Report coverage and selection limitations separately. Never impute missing targets for scoring.
4. Compare regression with a training-median baseline. Report MAE and RMSE in microseconds per foot, including per-well and rock-type error analysis.
5. Compare classification with a majority baseline. Report prevalence, precision, recall, balanced accuracy, average precision and confusion matrices. Do not claim calibrated probabilities without evidence.
6. Account for well dependence in uncertainty summaries. Include repeated grouped development evaluation and per-well reporting rather than treating neighboring rows as independent.
7. Diagnose incomplete logs, extreme readings, mixed-label ambiguity, false positives, poorly performing wells and unsupported input ranges.
8. Freeze the scientific design and selection rules before scoring reserved final-test wells. Retain baseline results and losing candidates.

Feasibility experiments support viability but are not final project results. A tree need not outperform linear regression for the comparison to be worthwhile.

## Deliverables

- Organized public GitHub repository, meaningful commit history and final release.
- Reusable Python package containing three custom algorithms and shared data-processing, evaluation and prediction logic.
- Tests for algorithm correctness, invalid inputs, splits, preprocessing, artifacts and relevant system behavior.
- Reproducible notebooks covering data preparation, sonic reconstruction, the linear-versus-tree comparison, sandstone classification, representation and failure analysis.
- Minimal interface demonstrating supported predictions and clearly reporting unsupported inputs.
- Dataset provenance, product/model documentation, reproducibility instructions and applicable operations/release evidence from the professor's template.
- Final presentation explaining the questions, mathematical ideas, results, judgments and limitations.

## Scope boundaries

Use one dataset, three predictors, two supervised targets and three core custom algorithms. Do not add oil discovery, reserve estimation, production forecasting, seismic interpretation, automated drilling decisions, live feeds or an elaborate dashboard. Sandstone identification is not evidence that oil is present.

The first interface requires all three inputs and reports unsupported requests instead of silently inventing missing measurements. Broader population or input support requires additional validation.

## Requirements basis and completion standard

The syllabus requires at least two reusable machine-learning algorithms; this scope includes three. Grading categories are functionality/implementation (40%), documentation/readability (20%), testing/reliability (20%), examples/usability (10%) and repository quality (10%). A public repository and presentation are required.

The template uses broader applicable-algorithm wording and includes additional product/operations evidence. Preserve those requirements and review them before final compliance sign-off. Three algorithms satisfy the stated numerical syllabus minimum; that alone does not establish compliance with every template requirement.

Success means tested implementations, reproducible experiments, understandable explanations, defensible evaluation, applicable deliverables and a presentation Liam can explain. It does not require a predetermined accuracy or a result favoring the more complex model.
