# Shallow regression tree: sonic-log reconstruction

## Question and implementation

Can a small nonlinear partition improve sonic-slowness reconstruction over the
linear model in unseen wells? SquaredErrorTreeRegressor is implemented in
src/dsm_project/algorithms/student/regression_tree.py and registered as
squared_error_tree. It follows Lecture 14, notebook
lecture-14-decision-trees/01-regression-trees.ipynb and the course's sorted-prefix
SSE approach in src/rice_dsm/ml/trees.py.

A leaf predicts its training-target mean, the constant minimizing squared error.
Each candidate split x_j <= t minimizes SSE(left) + SSE(right), where
SSE(S) = sum((y_i - mean(y_S))^2). Candidates are midpoints between distinct sorted
feature values. Prefix sums and sums of squares evaluate all valid splits.
Targets are centered within each node to reduce numerical cancellation without
changing SSE. Child SSEs are summed, not averaged without sample-size weighting.

Splits recurse until the depth limit, insufficient observations for two valid
leaves, or no positive reduction beyond numerical tolerance (1e-12). Ties favor
the first feature and lowest threshold. Root depth is zero. Predictions route
left at equality. Nodes expose sample counts, SSE, feature and threshold.
This implementation uses depth and minimum leaf size rather than post-pruning.
It needs no feature scaling; finite complete numeric inputs are required.

## Development design

- Same archived data hash, GR/RHOB/NPHI predictors and observed DTC target as linear regression.
- Same seed 577, 58 training wells and 20 validation wells.
- Same 14,500 sampled training observations, capped at 250 per well.
- Same 135,154 complete validation observations; no final-test scores.
- Prespecified grid: depths 2, 3, 4 crossed with minimum leaf counts 50, 100.
- Selection: lowest pooled validation MAE; ties favor lower depth then larger leaves.
- All six candidates retained. Selection is tuning on validation, so the winning
  score is not an unbiased final performance estimate.

Training sample SHA256:
053d868f3cdc82aea540bd173a6fe31f426efe6b834a465a9769acc1090413d9.

## Observed development results

| Depth | Minimum leaf count | Training MAE | Validation MAE | Validation RMSE |
| --- | --- | --- | --- | --- |
| 2 | 50 | 10.245 | 12.121 | 17.406 |
| 2 | 100 | 10.245 | 12.121 | 17.406 |
| 3 | 50 | 8.987 | 10.224 | 15.144 |
| 3 | 100 | 8.987 | 10.224 | 15.144 |
| 4 | 50 | 8.408 | 9.327 | 13.539 |
| 4 | 100 | 8.361 | 9.517 | 14.076 |

Errors are in microseconds per foot. The selected depth-4, minimum-leaf-50 tree
has 16 leaves. Its MAE is 9.327 versus 10.519 for linear regression and 21.574
for the training-mean baseline, approximately an 11.3% reduction relative to linear.
RMSE is 13.539 versus 14.936 for linear regression. The larger leaf constraint
does not monotonically increase training error: greedy split paths can change.

The scikit-learn squared-error reference uses the selected depth/leaf settings
and the same training sample. Maximum validation prediction difference is about
1.42e-13 in this run. Exact agreement is not required generally because floating
precision and tied-split policies can differ; analytic tests support correctness.

## Verification and limitations

Tests cover a hand-computable stump, threshold equality, constant features and
targets, depth/leaf constraints, duplicate features, trusted-reference agreement,
large target offsets, row permutation, refitting and invalid inputs. A synthetic
runner test ensures all candidates are retained, tie selection is reproducible,
and validation targets cannot become leaf predictions.

Whole-well splitting limits within-well leakage but does not remove geological
shift or complete-case selection bias. Pooled MAE favors wells with more observed
intervals. A tree is discontinuous and cannot extrapolate beyond its training
leaf means. Good reference agreement verifies computation, not scientific validity.
Per-well metrics are saved for the next comparative notebook. Repeated splits,
rock-type error analysis and final reserved-well evaluation remain pending.
This candidate is retained for further analysis, not promoted into production.

## Reproduce

From the project root, with the audited archive present:

```powershell
uv run --locked python -m dsm_project.experiments.force_tree
uv run --locked pytest tests/test_student_tree.py tests/test_force_tree.py -q
```

The runner also refreshes linear diagnostics for an identical-sample comparison.
Outputs in artifacts/force_tree/ include validation.json, candidates.csv and
per_well.csv; these remain local and excluded from Git.
