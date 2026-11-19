# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: C:\Users\lmora\INDE577\force-well-interpretation.
Branch: feature/logistic-regression.

## Completed

- Regression notebook merged into main (22bc95a).
- Added BatchLogisticClassifier and registry name batch_logistic_regression.
- Stable sigmoid and cross-entropy, full-batch gradients, explicit threshold,
  fitted-state checks and convergence evidence implemented following Lecture 11.
- Same whole-well boundaries; 14,500 training rows and 145,345 validation rows.
- Scaling fitted only on training sample; fixed initial cutoff 0.5.
- Validation balanced accuracy 0.6530 versus majority baseline 0.5000;
  precision 0.4556 and recall 0.3829. Limitations are documented.
- Custom model converged after 4,016 updates and closely matches the reference.
- Exploratory threshold sweep saved without selecting a new primary cutoff.
- Methods and results saved in docs/LOGISTIC_REGRESSION.md.
- Full local suite: 106 passed, two existing dependency warnings; Ruff passed.
- Artifacts remain local under artifacts/force_logistic/.

## Next

User reviews and commits the local milestone, pushes feature/logistic-regression,
then opens a pull request and merges only after checks pass.
No commit or remote publication of this milestone was performed during implementation.
Next: applied classification notebook with convergence, confusion matrix,
precision-recall and threshold trade-offs. The shallow regression tree, PCA,
rock-type diagnostics and repeated-split sensitivity remain planned.
Keep all 20 final-test wells excluded until choices are frozen.

Naming preference: PR titles, descriptions and commit messages describe methods
and changes. Specific course references remain in code comments and documentation.
