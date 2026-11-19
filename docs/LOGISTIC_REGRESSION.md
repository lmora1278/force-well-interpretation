# Sandstone classification development experiment

## Question and interpretation

How effectively can gamma ray, bulk density and neutron porosity distinguish
sandstone-labelled intervals from other labelled rock types in unseen wells?
The positive class is FORCE code 30000. Mixed sandstone/shale (65030) remains
negative. Missing labels are excluded, never converted to negatives.
This is screening for agreement with supplied lithology labels, not a test of
oil presence, a substitute for geological interpretation, or an assessment of
independent physical ground truth.

## Mathematical method and course reference

The implementation follows Randy Davila's Lecture 11
`00-logistic-model-and-decision-policy.ipynb` at reviewed commit `b697d54`:
affine scores, stable sigmoid, mean binary cross-entropy, zero initialization,
and full-batch gradient descent. For design A including a column of ones:

- z = A theta; p = sigmoid(z).
- L = mean(softplus(z) - y z), equivalently the class-specific softplus form
  used in code to avoid subtraction of large similar values.
- gradient = A-transpose (p - y) / n.
- theta is updated by subtracting learning_rate times gradient.

`BatchLogisticClassifier` supplies fit, decision_function, predict_proba and
predict, plus aligned classes_, coefficients and convergence evidence. Its
registry name is `batch_logistic_regression`. Probability columns are [0, 1].
The classifier is unregularized and unweighted. Inputs are standardized with
training-only statistics. Predictions at the initial cutoff use p >= 0.5.
Parameters: learning_rate=0.2, max_iterations=20000, tolerance=1e-7.
These were fixed before examining this run's validation scores.

## Data and evaluation boundaries

Use the same 58 training wells, 20 validation wells and 20 reserved final-test
wells as the regression study. Select up to 250 complete classification rows
per training well with seed 577: 14,500 training observations. Classification
does not require DTC, so its eligible rows differ from regression. Evaluate all
145,345 eligible validation observations. Sandstone prevalence is 19.08% in
the bounded training sample and 14.38% in validation. Equal-cap well sampling
and complete-case filtering affect representativeness.

The baseline returns training class frequencies and predicts the majority at
0.5. The reference is scikit-learn LogisticRegression(C=infinity), with no
regularization, the same rows and the same training-only scaling.
No final-test well is fitted or scored; no classification-specific re-split is made.

## Initial validation results

Verified November 19, 2026. These are development results from one split.

| Measure | Prior/majority baseline | Batch logistic | Trusted reference |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8562 | 0.8455 | 0.8455 |
| Balanced accuracy | 0.5000 | 0.6530 | 0.6530 |
| Sandstone precision | 0.0000 | 0.4556 | 0.4556 |
| Sandstone recall | 0.0000 | 0.3829 | 0.3829 |
| Average precision | 0.1438 | 0.3697 | 0.3697 |
| Log loss | 0.4195 | 0.3635 | 0.3635 |
| Brier score | 0.1253 | 0.1109 | 0.1109 |

Baseline precision is reported as zero because it predicts no positives.
For wells with only one observed class, balanced accuracy and average precision
are reported as null; positive recall is null if no positives are present.

Confusion counts for the custom model at 0.5:

| Observed / predicted | Other | Sandstone |
| --- | ---: | ---: |
| Other | 114,880 | 9,564 |
| Sandstone | 12,898 | 8,003 |

The baseline's higher ordinary accuracy illustrates the minority-class problem:
it misses every sandstone interval. Logistic regression improves balanced
accuracy, ranking and probability losses, but still misses most sandstone at
0.5 and has more false positives than true positives. Improved Brier/log-loss
values do not establish probability calibration.

Optimization reached the gradient tolerance after 4,016 updates. The largest
validation probability difference from the unregularized reference is 3.39e-6;
both yield identical decisions at 0.5. This supports implementation correctness,
not geological validity. Per-well balanced accuracy ranges from about 0.447 to
0.936. Well 31/3-3 falls below the 0.5 majority baseline.

## Decision threshold trade-offs

The experiment reports a predeclared sweep 0.1, 0.2, ..., 0.9 for exploration,
without changing the primary threshold or selecting a production cutoff.
For illustration, a cutoff of 0.2 yields balanced accuracy 0.7647, precision
0.3481 and recall 0.7725 on this validation set. Higher recall comes with more
false alarms. A threshold chosen using these labels would be a tuning result,
not independent evidence; freeze its rule before final-test evaluation.
No acceptable false-alarm cost or geological acceptance threshold is established.

## Verification and reproduction

Tests cover extreme logits, finite-difference gradients, an intercept-only
prevalence solution, noisy reference agreement, probability normalization,
threshold semantics, invalid inputs/settings, fitted-state behavior, exhausted
iteration budgets, training-only scaling and exact confusion counts.
Completely separable data can lack a finite unregularized optimum; callers must
inspect converged_ rather than assume every fit converges.

```powershell
uv run --locked pytest tests/test_student_logistic.py -q
uv run --locked python -m dsm_project.experiments.force_logistic
```

Local artifacts under `artifacts/force_logistic/` include validation.json,
per_well.csv, thresholds.csv and loss_history.csv. The JSON records optimizer
settings, archive/sample fingerprints, well identities, scaling and coefficients.
Tests use synthetic inputs, so automated checks need no field-data download.

Next: an applied classification notebook with convergence, confusion matrix,
precision-recall and threshold plots; review false positives by original rock
label and assess probability calibration and stability across development wells.
The supporting classification question does not replace the primary sonic
reconstruction question or expand this project to twelve-class lithology prediction.
