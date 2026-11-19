# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Branch: feature/tree-comparison-notebook.

## Completed

- Regression tree merged into main (339223b).
- Added notebooks/03-tree-comparison.ipynb using the package experiment runner.
- Covers provenance, missing-data coverage, model mathematics, six retained
  candidates, validation selection, residuals, per-well errors and extrapolation.
- Five figures executed and reviewed. The selected tree improves 13 of 20 wells;
  a post-hoc omission diagnostic exposes dependence on the largest-gain well.
- Primary comparison remains unchanged: pooled MAE 9.327 versus linear 10.519
  microseconds per foot. These are validation-tuning results, not final scores.
- Notebook outputs are cleared for Git. Executed review copy and figures remain
  local under artifacts/force_tree/03-tree-comparison.executed.ipynb.
- Added structural and clean-kernel notebook tests, with full-data execution
  skipped in environments without the ignored archive.

## Next

Review, commit and push this branch, then merge after remote checks pass.
No commit or publication was performed during notebook implementation.
Next: planned representation study and further robustness/rock-type diagnostics.
Check the project overview and course alignment before choosing the next scope.
Keep all 20 reserved final-test wells excluded until decisions are frozen.
Product integration, final evaluation and release requirements remain pending.

Naming: PRs and commit messages describe methodology; specific course references
remain in code comments and supporting documentation.

## Verification

Full local suite: 123 tests passed in 85.10 seconds, with two existing dependency warnings. Ruff passed. The final notebook executed from a clean kernel and all five figures were reviewed. Git whitespace checks passed.
