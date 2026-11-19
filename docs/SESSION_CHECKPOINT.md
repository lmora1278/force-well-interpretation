# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: C:\Users\lmora\INDE577\force-well-interpretation.
Branch: feature/linear-regression.

## Completed

- FORCE data milestone merged into main (242f363).
- Implemented and registered LectureLinearRegressor using Lecture 10 full-MSE gradients.
- Added reproducible 250-row-per-well training sampling and training-only scaling.
- Compared custom regression with a training-mean baseline and least squares.
- Validation uses 20 whole wells; all 20 final-test wells remain excluded.
- Validation MAE: baseline 21.5744; custom linear 10.5195 microseconds per foot.
- Optimization converged after 587 updates and agrees closely with least squares.
- Full suite: 86 passed, two existing dependency warnings. Ruff checks passed.
- Methods and limitations saved in docs/LINEAR_REGRESSION.md.
- Detailed reproducible outputs are local in artifacts/force_linear/ (ignored by Git).

## Next

User reviews and commits this local milestone, pushes feature/linear-regression,
then opens a pull request and waits for passing checks before merging.
No commit or remote publication of this milestone was performed during implementation.
Next scientific work: a question-led regression notebook with convergence,
prediction/residual and well-level plots, followed by logistic regression and a
shallow regression tree. PCA and sensitivity analyses remain planned.

Do not score final-test wells until methods and choices are frozen.
The existing Wine notebooks are inherited examples, not FORCE results.
