# Final evaluation implementation and pre-score checkpoint

Owner: Liam Mora. November 26, 2026.
The real reserved wells remain unscored at this checkpoint.

## Verified training artifact

Local trusted bundle: artifacts/force_final/v1.
Source revision: 1234a2ddeaf7f4712838c55cd5e0a1472981313f.
Integrity checks and disjoint 78-development/20-reserved well identities passed.
Each task has 19,500 sampled training observations. Eligible development counts
were 548,686 for regression and 576,652 for classification. Linear converged in
510 updates, PCA2-linear in 204, and logistic in 3,243. These are training facts,
not final-performance results. The bundle retains all seven fixed candidates.

## Evaluator behavior

force_evaluate.score_reserved selects manifest wells before task eligibility,
calls predict/predict_proba only, and never fits, tunes or reselects models.
It verifies well identities, candidate sets and fixed selection/threshold.
It records every eligible observation's prediction, including unsupported inputs.
Pooled, support-specific, well and lithology metrics retain row/well counts;
empty support slices have zero rows and undefined metrics. Missing lithology is
an explicit regression slice. Single-class classification quantities follow the
existing explicit-null conventions.

Regression includes MAE, RMSE, signed bias, nonpositive-output counts,
equal-weight well MAE and paired well differences. Classification includes
confusion counts, prevalence, accuracy, balanced accuracy, precision, recall,
average precision, log loss, Brier score and positive flag rates. Rock-type
positive flag rate is recall for sandstone and false-positive rate otherwise.
The report tests the predeclared tree-versus-median MAE criterion without
changing the selected model. Results do not confer operational approval.

The command requires committed clean evaluator code and checks that prior
training source/configuration/protocol bytes still match the saved artifact.
It records evaluation revision, training revision, evaluator and artifact hashes.
A new destination is reserved with STARTED.json before scoring; COMPLETE.json
is written only after all reports/predictions are saved. Existing destinations
are never overwritten. Preserve incomplete runs and diagnose failures; do not
silently erase evidence or repeat selection after seeing scores.

## Next step after this code is merged

From a clean checkout, the following command opens the actual reserved scores:

```powershell
uv run --locked python -m dsm_project.experiments.force_evaluate --artifact artifacts/force_final/v1 --output artifacts/force_evaluation/v1
```

Do not run it before this implementation milestone is committed and merged.
No actual held-out scoring was used to develop these tests. All new evaluator
tests use hand-checkable synthetic data. The saved fitting artifact remains
unchanged; final evaluation is a separate, traceable record.

After scoring, inspect all results, write a final interpretation and model card,
and preserve losing candidates and failures. A reproducibility rerun is not a
fresh holdout. Follow EVALUATION_PLAN.md for any defect correction or amendment.
The reports and row predictions stay in ignored artifacts/; selected aggregate
evidence and final scientific interpretation will be committed separately.
