# Linear regression development experiment

## Method and course alignment

The project estimator `LectureLinearRegressor` follows Randy Davila's Lecture 10
`00-gradient-descent-from-functions-to-neuron.ipynb`, reviewed at course commit
`b697d54`: prediction Xw + b, mean squared error, separate weight and bias
gradients, zero initialization, and full-batch negative-gradient updates.

For n observations and residual r = Xw + b - y:

- Loss = mean(r squared).
- Weight gradient = 2 X-transpose r / n.
- Bias gradient = 2 mean(r).
- Update both parameters by subtracting learning_rate times their gradient.

The new implementation is under `algorithms/student/linear_regression.py` and
registered as `lecture_linear_regression`. The inherited template exemplar
`batch_linear_regression` remains a separate demonstration; it uses half-MSE.
Input checks use scikit-learn utilities, but parameter updates are implemented
directly with NumPy. The estimator records initial and final losses, iteration
count, convergence status and final gradient norm.

## Evaluation design

Use the existing whole-well partition. Only complete GR/RHOB/NPHI/DTC training
observations are eligible. Draw 250 observations without replacement from each
of 58 training wells using seed 577, after sorting well and depth. This gives
14,500 training observations and gives each well equal training weight. It
does not make correlated depth measurements independent.

Fit StandardScaler on this training sample only. Fit all three comparisons on
the same rows, then evaluate on all 135,154 eligible observations from the 20
validation wells. No final-test well is fitted or scored. StandardScaler is
part of each linear model's pipeline; validation only calls predict.

Fixed optimization settings: learning_rate=0.05, max_iterations=5000,
tolerance=1e-7 (joint gradient norm). No validation search was performed for
these settings. The baseline predicts the mean of the sampled training targets.
The trusted reference is unregularized least squares with the same scaling.

## Initial development results

Verified November 19, 2026. Errors are in microseconds per foot.

| Model | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| Training-mean baseline | 21.5744 | 26.4162 |
| Lecture-based batch gradient descent | 10.5195 | 14.9360 |
| Trusted least-squares reference | 10.5195 | 14.9360 |

Gradient descent reached the declared tolerance after 587 updates. Its maximum
validation prediction difference from the reference was 7.85e-7 microseconds
per foot. Agreement supports optimization correctness, not accuracy in all wells.
Validation well MAE ranges from 4.33 to 24.41; errors are uneven across wells.
These pooled metrics weight wells by their number of eligible validation rows.
Future reports must also compare equal-weight well summaries and rock types.

Complete-case exclusions and equal-cap training sampling change which intervals
are represented. This is one development split, not final-test evidence or a
claim of reliability where sonic measurements are genuinely missing.

## Verification and reproduction

Tests cover finite-difference gradients, an exact affine rule, noisy multivariate
least-squares agreement, constant inputs, invalid parameters and observations,
unfitted prediction, limited iteration budgets, order-independent sampling, and
training-only scaling. No field-data archive is needed for the unit tests.

```powershell
uv run --locked pytest tests/test_student_linear.py -q
uv run --locked python -m dsm_project.experiments.force_linear
```

The experiment writes `validation.json`, `per_well.csv`, and `loss_history.csv`
under ignored `artifacts/force_linear/`. The report records archive and sample
fingerprints, well lists, scaling parameters and coefficients. The local
audited archive must be present at `data/local/force-train.zip`.

Next: add a question-led notebook with convergence, prediction/residual and
well-level plots; implement the supporting logistic model and shallow tree;
complete PCA and sensitivity analyses before freezing final-test evaluation.
