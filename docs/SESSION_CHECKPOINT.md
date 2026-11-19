# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: C:\Users\lmora\INDE577\force-well-interpretation.
Branch: feature/regression-tree.

## Completed

- Classification notebook merged into main (5a8f2e7).
- Added SquaredErrorTreeRegressor and squared_error_tree registry entry.
- Mean leaves and sorted-prefix squared-error splits follow Lecture 14.
- Depth and leaf-size stopping, deterministic ties, input validation and
  inspectable fitted nodes implemented and tested.
- Compared a prespecified six-candidate grid using the same 14,500 training
  observations and 135,154 validation observations as linear regression.
- Selected depth 4 / minimum leaf size 50 by pooled validation MAE.
- Development MAE 9.327 versus linear 10.519 and mean baseline 21.574 us/ft.
- Trusted reference predictions match within approximately 1.42e-13.
- All candidates and per-well diagnostics saved locally in artifacts/force_tree/.
- Method, results and limitations saved in docs/REGRESSION_TREE.md.

## Next

Review, commit and push feature/regression-tree, then open a pull request and
merge after both remote checks pass. No commit or remote publication was performed
during implementation.

Next: comparative sonic-regression notebook, including tree complexity, per-well
errors, residuals and extrapolation limits. The tree's validation score is tuning
evidence, not a final-test estimate. PCA, rock-type diagnostics, repeated-split
sensitivity and product/release requirements remain pending. Keep all 20 reserved
wells excluded until modeling and evaluation choices are frozen.

PR titles, descriptions and commit messages describe methodology and changes.
Specific course references remain in code comments and supporting documentation.

## Verification

Full local suite: 121 tests passed in 71.04 seconds, with two existing dependency warnings. Ruff and Git whitespace checks passed. The development experiment completed successfully; the final-test wells remain excluded.
