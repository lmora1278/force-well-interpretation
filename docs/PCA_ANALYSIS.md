# PCA representation study

Updated November 24, 2026.

## Purpose and course alignment

Investigate redundancy among GR, RHOB and NPHI and the effect of compressing
these inputs on sonic reconstruction. PCA remains a supporting library-based
analysis, not an additional core custom algorithm. The professor's project
template uses StandardScaler followed by scikit-learn PCA in
src/dsm_project/experiments/unsupervised.py::run_pca_example. We preserve that
approach but fit exclusively on training wells. The reviewed course snapshot's
Lecture 9 landscape notebook introduces PCA; no dedicated implementation
notebook was found in the available snapshot. This is not a claim about updates
that may exist outside the reviewed local material.

## Method and boundaries

The runner reuses the regression sample: 14,500 observations in 58 training wells
and 135,154 observations in 20 validation wells. All 20 final-test wells remain
excluded. This deliberately represents the regression complete-case population,
not every row with available predictor measurements. DTC availability is a
population filter, but DTC values do not fit PCA or standardization.

Standardize with training means and scales. Full SVD on centered standardized
training predictors supplies orthonormal axes and explained-variance fractions.
No whitening, clipping or validation-fitted preprocessing is used. Retain 1, 2
and 3 components as diagnostic alternatives. Each feeds the same custom
unregularized linear estimator. PCA minimizes predictor reconstruction error;
it does not optimize DTC prediction. All three axes provide a lossless rotation.

## Development evidence

| Components | Training variance retained | Validation MAE (us/ft) | Validation RMSE (us/ft) | Standardized reconstruction MSE |
| --- | --- | --- | --- | --- |
| 1 | 52.84% | 10.833 | 15.139 | 0.46723 |
| 2 | 91.02% | 10.271 | 14.460 | 0.15548 |
| 3 | 100% | 10.519 | 14.936 | approximately 0 |

The two-axis representation slightly improves this development comparison
(about 2.4% lower MAE than uncompressed linear regression). This is not evidence
of consistent benefit across new wells. No component count is promoted or
selected as the production policy. The original three-input models remain intact.
Full-rotation predictions agree with the uncompressed model within 5.69e-14.

PC1 mainly contrasts RHOB with NPHI; PC2 is dominated by GR with a RHOB
contribution. Axis signs are arbitrary and coefficients are not causal effects.
Plots use up to 200 validation rows per well solely for display; metrics use
the full eligible validation population. Reconstruction errors are descriptive,
not calibrated anomaly or prediction-uncertainty scores.

## Verification and limitations

Synthetic tests check training-only scaling, unchanged fitted state on new
inputs, independence from target values, orthonormality, full reconstruction,
monotone reconstruction error with retained dimension and linear equivalence.
The notebook checks train/validation/test well disjointness and runs from a
clean kernel. Its integration test skips when the ignored field archive is absent.

Complete-case selection, correlated depths, finite extremes and covariance shift
limit interpretation. Validation was already used for other development decisions;
these results cannot be relabeled as final testing. Repeated whole-well robustness
and rock-type diagnostics remain required before final evaluation.

## Reproduce

```powershell
uv run --locked python -m dsm_project.experiments.force_pca
uv run --locked pytest tests/test_force_pca.py tests/test_pca_notebook.py -q
```

Open notebooks/04-pca-representation.ipynb with DSM Project. Outputs are cleared
for Git. The executed copy and numeric diagnostics remain in artifacts/force_pca/.

## Requirement audit note

The project overview specifies a training-median regression baseline, while
completed experiments currently use a training-mean baseline. Preserve the mean
baseline and add the planned median comparison in the robustness milestone.
This discrepancy must be resolved before claiming full requirement compliance.
