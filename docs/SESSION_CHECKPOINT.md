# Session checkpoint

Updated November 25, 2026. Owner: Liam Mora.
Branch: feature/robustness-analysis.

## Completed

- PCA merged into main (0701430).
- Added fixed-model robustness runner: five whole-well development holdouts,
  seeds 577–581, with all 20 final-test wells excluded.
- Added training-median baseline alongside the existing mean baseline.
- Retained linear, depth-4/leaf-50 tree and PCA2-linear candidates without retuning.
- Repeated logistic screening at fixed 0.5 with the prior baseline.
- Saved pooled, per-well and lithology-slice metrics, coverage, split/sample
  identities, convergence and classification flag rates locally.
- All iterative fits converged. Tree MAE beats linear on all five repeats,
  but RMSE is worse on one; classification recall remains limited.
- Detailed tables and caveats saved in docs/ROBUSTNESS_ANALYSIS.md.

## Next

Commit and push this milestone, then merge after both remote checks pass.
No commit or remote publication was performed during implementation.
Next: applied robustness notebook, followed by model-selection freeze and product
documentation/interface work. Keep every reserved well excluded until the final
evaluation protocol is frozen. Repeated development splits are not final testing.

The project still needs the applicable template product/operations evidence,
final evaluation, tagged release and presentation. Do not claim full rubric
completion until the checklist is reviewed against the final deliverables.

## Verification

Full local suite: 132 tests passed in 134.35 seconds, with two existing dependency warnings. Ruff and Git whitespace checks passed. The five-split field experiment completed; every iterative fit converged.
