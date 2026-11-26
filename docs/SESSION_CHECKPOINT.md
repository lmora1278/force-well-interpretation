# Session checkpoint

Updated November 25, 2026. Owner: Liam Mora.
Branch: feature/robustness-notebook.

## Completed

- Robustness runner and report merged into main (85ecb4b).
- Added notebooks/05-robustness-analysis.ipynb, interpreting the five fixed-model
  whole-well development splits without retuning or final-test scoring.
- Included eligibility coverage, pooled and equal-weight well errors, individual
  well comparisons, rock-type support and errors, and classification limitations.
- Four figures and an executed notebook are saved locally under
  artifacts/force_robustness/. Tracked notebook outputs remain cleared.
- The tree improves pooled MAE over linear on all five repeats, but has worse
  RMSE on one. Both models underperform the median baseline for rock code 90000
  on seed 577. The classifier's recall remains limited at threshold 0.5.
- Repeated splits overlap and follow development decisions; their ranges are
  descriptive sensitivity, not confidence intervals or independent test estimates.

## Next

Commit and push this notebook milestone, then merge after remote checks pass.
Implementation did not commit, push or publish remotely.
Next review the assignment and project checklist, record the model-selection
and supported-input decisions, and freeze the final-evaluation protocol.
Keep all 20 reserved wells excluded until that record is complete.

Remaining work includes applicable product/interface, model and dataset cards,
monitoring/operations evidence, final evaluation, release and presentation.
Do not claim full rubric completion before auditing those deliverables.

## Verification

Six targeted checks passed in 46.01 seconds, including complete fresh-kernel notebook execution and reserved-well isolation tests. One existing Windows event-loop warning was non-blocking. Ruff and Git whitespace checks passed; all four figures were visually reviewed. The full suite was not rerun for this notebook-only milestone.
