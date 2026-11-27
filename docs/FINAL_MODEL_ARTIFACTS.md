# Frozen candidate artifacts

Owner: Liam Mora. November 26, 2026.
Implements the fitting/storage stage of EVALUATION_PLAN.md. This module does
not score reserved wells or implement the final reporting/service layer.

## Scientific boundary

`dsm_project.experiments.force_final.fit_candidates` removes all reserved wells
before task eligibility, capped sampling, preprocessing, support bounds or
estimator fitting. All 78 development wells are available for final fitting.
Each task samples at most 250 eligible rows per well with seed 577. Candidate
sets and settings reuse the frozen robustness implementation, without tuning.
Regression retains mean, median, linear, depth-4/leaf-50 tree and PCA2-linear;
classification retains prior and unregularized logistic at threshold 0.5.

Iterative models must converge or fitting raises an error and produces no bundle.
The selected regression candidate remains the tree regardless of later scores.
Both the FORCE adapter and config now declare mean_absolute_error, matching
actual selection. Existing historical RMSE reports remain unchanged.

## Saved identity

Each new destination contains models.joblib, metadata.json and COMPLETE.json.
The model payload includes fitted estimators and preprocessing for every retained
candidate. Metadata records feature order/units, parameters, fitted scaler/PCA
and coefficient values, convergence, eligible/sample counts, well identities,
sample-ID hashes, training-only support limits and a false final_test_scored flag.
This flag describes artifact creation; future scoring must produce a separate
evaluation record, not silently rewrite this immutable training evidence.

The command additionally captures a clean Git source revision, byte hashes of
tracked source/configuration/lockfile files and protocol/contract documents,
archive hash, Python and key package versions. A changed source during fitting
causes failure. Working-tree byte hashes complement Git identity on Windows,
where checkout line endings may differ. The archive is validated by its adapter.

The destination must not exist. A failed write can leave an incomplete directory;
without its completion manifest it cannot be loaded. No existing artifact is
overwritten. Model and metadata hashes are checked before joblib loading. Hashes
detect corruption, not authenticity if an attacker replaces all files. Load only
trusted, locally produced bundles: joblib can execute code during deserialization.

## Input-support utility

`support_mask` accepts a DataFrame with exactly named GR, RHOB and NPHI numeric
columns, maps them to model order, and rejects missing, nonfinite, string and
Boolean measurements. It returns a mask for inclusive training-sample minima
and maxima. Negative NPHI can be supported. Being inside this rectangle does
not prove joint support or accuracy. The future service must refuse unsupported
requests; the final evaluator must still report all eligible observations plus
support-specific coverage/errors. Neither API policy nor final scoring is
implemented merely by this utility.

## Reproduce after committing and merging

Use a clean checkout at the intended revision. From the project root, with the
audited archive present, choose a new ignored artifact directory:

```powershell
uv run --locked python -m dsm_project.experiments.force_final --output artifacts/force_final/v1
```

This is fitting only. Do not run the generic template's row-split training
command for FORCE. If v1 already exists, inspect it; do not delete or overwrite
it merely to rerun. Keep the original artifact and document any revised version.
The command refuses uncommitted source changes so metadata names a recoverable
revision. The final field-data fit is deferred until this code milestone is merged.

## Verification and next work

Synthetic tests cover reserved-feature/target isolation, row-order stability,
sampling cap, prediction/probability round trips, overwrite refusal, corruption
before deserialization, incomplete bundles, input support and nonconvergence.
The existing dataset tests cover schema, fingerprint and whole-well boundaries.
Full verification results are recorded in SESSION_CHECKPOINT.md.

Next fit the actual development data from a clean merged revision, verify the
saved artifact, implement the reserved-well evaluator with synthetic boundary
checks, then follow the committed protocol for the one-time final report.
FORCE API/UI, model card, monitoring/rollback, release and presentation remain.
