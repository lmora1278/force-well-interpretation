# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: C:\Users\lmora\INDE577\force-well-interpretation.
Branch: feature/classification-notebook.

## Completed

- Logistic regression merged into main (7ae32b9).
- Added notebooks/02-sandstone-classification.ipynb as a self-contained case study.
- Documents provenance, complete-case coverage, whole-well splits, training-only
  scaling, model mathematics, implementation verification and limitations.
- Five figures cover training distributions, convergence, confusion and ranking,
  threshold sensitivity, and between-well variation.
- The fixed 0.5 policy remains unchanged. Balanced accuracy is 0.6530,
  precision 0.4556 and recall 0.3829; ordinary accuracy is below the baseline.
- No final-test wells were scored. Exploratory threshold comparisons are not
  a newly selected policy. The classifier is not promoted for autonomous use.
- Committed notebook outputs are cleared per docs/NOTEBOOK_CASE_STUDY_STANDARD.md.
  The executed local review copy and figures are in artifacts/force_logistic/.
- Added notebook contract and clean-kernel execution tests. The full-data test
  skips when the ignored field archive is unavailable, as on CI.

## Next

Review the notebook, commit and push feature/classification-notebook, then open
a pull request and merge after both remote checks pass. No commit or remote
publication was performed during this implementation step.

Next implementation: shallow squared-error regression tree for sonic prediction,
using the professor's tree approach, with whole-well validation and a trusted
reference. PCA, rock-type diagnostics, repeated-split sensitivity, final evaluation
and product/release requirements remain pending. Preserve all 20 reserved wells
until modeling and evaluation choices are frozen.

PR titles, descriptions and commit messages describe methodology and changes.
Specific course references remain in code comments and supporting documentation.

## Verification

Full local suite: 108 tests passed in 71.24 seconds, with the two existing dependency warnings. Ruff passed. The new notebook executed from a clean kernel; all five figures were visually reviewed. Git whitespace checks passed.
