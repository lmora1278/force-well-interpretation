# Final evaluation protocol

Owner: Liam Mora. Prepared November 25, 2026.
Status: pre-test decision record; effective when this milestone is committed
and merged. No reserved-well scores were produced while preparing this document.
This record freezes the scientific evaluation, not a claim of rubric completion
or approval for operational use. Record any amendment before opening test scores.

## Question, targets and evidence

Retain PROJECT_OVERVIEW.md: reconstruct DTC from GR, RHOB and NPHI in unseen
wells, compare simple and nonlinear predictions, and diagnose well/rock variation.
Supporting classification is code 30000 versus all other observed lithology
codes, including mixed sandstone/shale. PCA remains supporting analysis.
No new targets, predictors or algorithm families are introduced by this protocol.

Select the custom squared-error tree for the primary sonic demonstration:
max_depth=4, min_samples_leaf=50. It beat linear pooled validation MAE on all
five development splits. Keep the failures visible: one split had worse RMSE,
some wells favor linear, and code 90000 underperformed the median baseline.
This is a development-based selection, not proof of final superiority.

Keep custom linear regression as the simpler comparison and both constant
baselines. Keep PCA2-linear as a fixed supporting comparison, not a route to
select a new winner after viewing test results. Logistic screening retains
threshold 0.5 and its exploratory status because development recall was low.

## Immutable data boundary

Use the archive and SHA256 in configs/reserved_wells.json and the existing
validated adapter. The manifest's 20 wells are the only final-test wells.
The other 78 are final fitting data; they include all prior development wells.
Split whole wells before eligibility filtering. Assert disjoint identities and
persist the well lists. Never replace reserved wells based on sample size or score.

Regression requires finite observed GR, RHOB, NPHI and DTC. Classification
requires the same inputs and an observed lithology label. Do not impute targets.
Use existing validation rules without new test-driven clipping or outlier removal.
Report raw and eligible row/well counts and exclusions separately for each task.
Preserve missing lithology as an explicit regression diagnostic slice if present.

## Final fitting, fixed before test scoring

Refit on all 78 development wells using sample_training_wells with seed 577
and a cap of 250 eligible rows per well, separately by task. This differs from
the earlier 58-well training fits; record final sample counts and sorted-ID hashes.
Do not reuse validation-fitted objects or expand the sample cap after seeing scores.
All candidates for a task use the identical final training sample.

| Candidate | Fixed procedure |
| --- | --- |
| Mean baseline | Mean of sampled development DTC |
| Median baseline | Median of sampled development DTC |
| Linear | StandardScaler then LectureLinearRegressor; learning_rate=0.05, max_iterations=5000, tolerance=1e-7 |
| Tree (selected sonic candidate) | SquaredErrorTreeRegressor; depth 4, minimum leaf 50; no scaling |
| PCA2-linear (supporting) | StandardScaler, PCA(2, full SVD), same linear estimator |
| Prior classification baseline | DummyClassifier(strategy="prior"), decision threshold 0.5 |
| Logistic | StandardScaler then BatchLogisticClassifier; learning_rate=0.2, max_iterations=20000, tolerance=1e-7, threshold=0.5; no regularization or class weighting |

Fit scalers, PCA axes and all other learned quantities only on the relevant
development sample. Use zero initialization and existing stopping rules.
Require convergence for iterative fits before opening the test scoring step.
If fitting fails, diagnose using development data, record a protocol amendment,
and commit it before proceeding. Do not silently replace a failed candidate.

## Metrics and reporting

Primary regression comparison is pooled MAE in microseconds per foot, consistent
with actual tree selection. Report RMSE and signed bias as secondary measures.
The earlier configs/force.toml metric field says root_mean_squared_error; it did
not control the dedicated tree runner's MAE selection. Align that field with MAE
in the final-runner implementation before scoring, preserving historical reports.

Score all eligible reserved rows once with the fixed candidate set. Report
pooled errors, each well's errors and equal-weight mean well MAE. Report the
paired per-well tree-minus-linear and tree-minus-median differences, their
median, range and the number of wells favoring the tree. These are descriptive
variation summaries, not row-independent confidence intervals. No p-value or
population confidence claim is required from only 20 potentially related wells.
Report lithology-slice MAE, RMSE, bias and support in rows and distinct wells.
Label small slices as limited evidence; do not hide failures or retune on them.

For classification report prevalence, accuracy, balanced accuracy, precision,
recall, average precision, log loss, Brier score and the fixed-order confusion
matrix. Include the prior baseline, per-well support and rock-type flag rates.
Use existing metric conventions: undefined single-class quantities are null,
not fabricated zeros; distinguish zero predicted-positive precision from absent
positive-class recall. Do not claim calibrated probabilities or optimize a new
threshold on test labels.

## Supported-input policy and demonstration gate

The future interface must require named GR, RHOB and NPHI values with displayed
units (gAPI, g/cm3, fraction). Reject missing, nonnumeric or nonfinite values
and unexpected fields. Do not silently reorder inputs, impute or clip them.
For each served task, store feature minima/maxima from its final training sample;
refuse predictions outside that rectangular envelope as unsupported. Negative
NPHI values are not automatically invalid when present in the training envelope.
Being inside marginal ranges is not proof of joint support or reliable prediction.

To avoid selectively flattering final accuracy, primary test metrics cover all
eligible rows, including out-of-envelope rows. Separately report supported
coverage and metrics inside/outside the training-defined envelope, including
empty-subset counts. Do not narrow the envelope after seeing test results.
This gate is a conservative demonstration policy, not a physical safety boundary.

If the selected tree fails to beat the median baseline on pooled test MAE,
retain the negative result and do not present it as a validated reconstruction
tool. Do not switch winners on test evidence. Even if it beats the baseline,
the interface remains an educational, analyst-reviewed demonstration without
a domain-established acceptable error tolerance. Suppress nonfinite outputs;
nonpositive sonic estimates must be reported as invalid, not silently corrected.
The classifier remains exploratory regardless of test performance. No oil,
reserves, drilling or autonomous geological decisions are supported.

## Execution and provenance gates

1. Merge this protocol and record its commit ID.
2. Implement the final fitter/evaluator, artifact metadata and input policy.
   Test them on synthetic fixtures and development data only. Verify reserved
   labels cannot affect fitting, preprocessing, sample selection or support bounds.
3. Complete the project-specific product/dataset contracts. Record source revision,
   lockfile/configuration hashes, archive hash, well/sample identities, estimator
   parameters, fitted preprocessing, convergence and model-file hash.
4. Save and verify the immutable fitted candidates before scoring. The demo must
   load the selected evaluated artifact rather than silently train another model.
5. Execute the fixed final evaluation and preserve every candidate's predictions,
   scores, coverage and failures. Produce a final report/notebook and model card.
6. Complete API/UI, operations exercise, clean-clone checks, release and presentation.

One-time means no test-guided model selection. A reproducibility rerun may use
the same frozen inputs and code; record it. If an implementation defect requires
correction after score access, preserve the original result, explain the defect,
record the corrected revision and disclose test reuse in the scientific report.
Do not pass a corrected rerun off as a newly untouched holdout.
