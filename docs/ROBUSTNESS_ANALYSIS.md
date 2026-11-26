# Robustness and rock-type development analysis

Updated November 25, 2026. Owner: Liam Mora.

## Protocol fixed before this run

Use split seeds 577, 578, 579, 580 and 581, each assigning 58 training and 20
validation wells from the 78 development wells. The same manifest reserves all
20 final-test wells; none enter fitting or scoring. Each split assigns wells
before task-specific complete-case filtering. Sampling uses seed 577 and at most
250 eligible rows per training well. Sampling is deterministic for each partition;
changing well membership may also change sampled rows in shared training wells.

Regression retains five fixed candidates: training-mean baseline, training-median
baseline, custom linear regression, custom tree (depth 4, minimum leaf size 50),
and two-component PCA plus custom linear regression. Scaling and PCA fit only
training data. No grid search or retuning occurs within this experiment. The
median baseline resolves the baseline specified in PROJECT_OVERVIEW.md; the mean
baseline is preserved for continuity. A training median minimizes training absolute
error but need not beat the mean on shifted validation data.

Classification retains the training-prior baseline and custom logistic model at
the existing 0.5 threshold. No threshold is selected from these new results.
All iterative fits converged. Source and sample hashes, well assignments, task
coverage and convergence records are saved beside the metric tables.

These are overlapping repeated holdouts following earlier model development.
Seed 577 was previously used for tuning. Other repeats may place previously
seen development wells in new roles. Consequently these are sensitivity checks,
not independent final estimates, nested cross-validation or confidence intervals.
Do not pool repeated rows into a fictitiously enlarged independent dataset.

## Regression: pooled MAE by split

All errors are microseconds per foot; negative tree-minus-linear favors the tree.

| seed | mean_baseline | median_baseline | linear | tree | pca2_linear | tree_minus_linear |
| --- | --- | --- | --- | --- | --- | --- |
| 577 | 21.574 | 21.888 | 10.519 | 9.327 | 10.271 | -1.193 |
| 578 | 21.149 | 19.763 | 8.661 | 8.427 | 8.691 | -0.235 |
| 579 | 18.630 | 18.426 | 8.821 | 8.752 | 8.813 | -0.069 |
| 580 | 24.010 | 26.011 | 9.964 | 8.741 | 9.932 | -1.222 |
| 581 | 25.482 | 27.191 | 10.225 | 9.180 | 10.129 | -1.045 |

The fixed tree improves MAE on all five splits, by approximately 0.07 to 1.22
us/ft. Its MAE ranges from 8.43 to 9.33, versus 8.66 to 10.52 for linear
regression. However, at seed 579 the tree has worse RMSE (11.83 versus 11.39),
showing that lower average absolute error does not imply better large-error
control. PCA improves linear MAE in four of five splits, with small differences;
this does not justify a new compression policy by itself.

## Equal-weight well MAE

The table below averages each well's MAE equally, rather than weighting wells by
eligible row count. This is descriptive between-well evidence, not a confidence
interval. It preserves the distinction between observation-level and well-level
objectives; selection was previously based on pooled validation MAE.

| seed | median_baseline | linear | tree | pca2_linear |
| --- | --- | --- | --- | --- |
| 577 | 18.718 | 10.166 | 9.000 | 9.921 |
| 578 | 16.559 | 8.432 | 8.209 | 8.438 |
| 579 | 16.458 | 8.623 | 8.489 | 8.620 |
| 580 | 20.848 | 9.537 | 8.591 | 9.503 |
| 581 | 22.139 | 9.130 | 8.589 | 9.034 |

## Rock-type error slices on the original development split

Use supplied FORCE lithology codes as identifiers. Code 30000 is sandstone,
65030 is mixed sandstone/shale. The codes are external audit labels, never model
inputs. Rows with missing lithology remain an explicit missing slice for regression
if present; missing targets remain excluded. All splits' slice metrics, RMSE and
signed bias are retained in rock_type.csv; the table below uses seed 577 only.

| slice | rows | wells | median_baseline | linear | tree |
| --- | --- | --- | --- | --- | --- |
| 30000 | 18694 | 20 | 17.461 | 7.558 | 7.663 |
| 65000 | 82999 | 20 | 25.062 | 11.624 | 9.812 |
| 65030 | 18634 | 20 | 14.913 | 7.489 | 6.950 |
| 70000 | 8712 | 20 | 17.725 | 8.628 | 9.431 |
| 70032 | 222 | 1 | 14.517 | 7.669 | 6.420 |
| 74000 | 78 | 4 | 11.568 | 7.472 | 8.477 |
| 80000 | 3553 | 10 | 13.113 | 11.441 | 11.078 |
| 90000 | 637 | 11 | 17.355 | 37.393 | 29.417 |
| 99000 | 1625 | 6 | 35.466 | 21.067 | 19.131 |

Support is essential: code 70032 has only one represented validation well, and
code 74000 has only 78 eligible observations. Their errors are not reliable
population comparisons. Codes 90000 and 99000 have substantially higher tree MAE
than the common sandstone and shale-labelled groups. This flags heterogeneous
performance for further review; neither a rock-type cause nor a deployment rule
can be inferred. Compare models on identical rows within each slice.

## Classification across splits

| seed | prevalence | balanced_accuracy | precision | recall | average_precision |
| --- | --- | --- | --- | --- | --- |
| 577 | 0.144 | 0.653 | 0.456 | 0.383 | 0.370 |
| 578 | 0.157 | 0.590 | 0.374 | 0.263 | 0.362 |
| 579 | 0.163 | 0.633 | 0.770 | 0.283 | 0.635 |
| 580 | 0.157 | 0.592 | 0.513 | 0.224 | 0.529 |
| 581 | 0.138 | 0.609 | 0.685 | 0.236 | 0.559 |

The prior baseline has balanced accuracy 0.5 and zero recall at threshold 0.5
on every split. Logistic balanced accuracy ranges from 0.590 to 0.653, but recall
is only 0.224 to 0.383. Precision varies widely (0.374 to 0.770). This is inadequate
evidence for autonomous interpretation, even where probability losses improve.
Full baseline metrics and confusion counts remain in classification.csv.

### Flag rates by supplied rock label, seed 577

For code 30000, positive flag rate is sandstone recall. For every other code it
is a false-positive rate under the declared binary label, including mixed 65030.
It is not precision within a rock type. Counts and distinct wells are included.

| lithology_code | rows | wells | positive_flag_rate |
| --- | --- | --- | --- |
| 30000 | 20901 | 20 | 0.383 |
| 65000 | 88860 | 20 | 0.041 |
| 65030 | 19229 | 20 | 0.075 |
| 70000 | 9919 | 20 | 0.318 |
| 70032 | 222 | 1 | 0.077 |
| 74000 | 192 | 6 | 0.000 |
| 80000 | 3649 | 11 | 0.138 |
| 86000 | 17 | 1 | 0.000 |
| 90000 | 711 | 12 | 0.322 |
| 99000 | 1645 | 6 | 0.323 |

## Limitations and next decision

The tree remains a promising candidate, with linear retained as a simple reference.
Do not promote a model or claim final accuracy from these repeats. Per-well
variation, complete-case selection, depth dependence, rare lithologies, limited
classification recall and out-of-range input behavior remain material limitations.
No thresholds, inputs or candidate settings were changed after observing results.

Next, present the robustness evidence in a concise applied notebook, then complete
the model-selection record, supported-input contract and template product documents.
Freeze the evaluation protocol before the one-time reserved-well evaluation.

## Reproduce and verification

```powershell
uv run --locked python -m dsm_project.experiments.force_robustness
uv run --locked pytest tests/test_force_robustness.py -q
```

The runner uses the existing custom algorithms and loader; no new algorithm or
source dataset is introduced. Outputs remain in ignored artifacts/force_robustness/.
Tests verify that changing reserved targets cannot alter development results,
that mean/median baselines use training targets, that subgroup counts include
missing labels, and that repeated seeds cannot masquerade as separate runs.
