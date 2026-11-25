# Session checkpoint

Updated November 24, 2026. Owner: Liam Mora.
Branch: feature/pca-analysis.

## Completed

- Prior tree-comparison notebook merged; started from clean main.
- Added training-only standardized PCA experiment and representation notebook.
- Reused the fixed regression sample and well boundaries; no final-test scoring.
- PCA uses the professor template's library-based approach, not a new core custom algorithm.
- Two components retain 91.02% of training variance. Validation MAE is 10.271
  versus uncompressed linear 10.519 us/ft. This modest gain is provisional.
- Full three-axis rotation matches the uncompressed linear model within 5.69e-14.
- Added synthetic invariants and clean-kernel notebook tests.
- Documented results, source alignment and limitations in docs/PCA_ANALYSIS.md.
- Outputs cleared for Git. Executed notebook, three reviewed figures and numeric
  diagnostics remain locally in artifacts/force_pca/.

## Next

Commit and push feature/pca-analysis, open a PR and merge after remote checks pass.
No commit or remote publication was performed during implementation.

Next milestone: robustness and rock-type error analysis. Add the training-median
baseline required by PROJECT_OVERVIEW.md while retaining the existing mean baseline.
Evaluate repeated grouped development splits without accessing reserved final wells.
Do not expand the candidate grid merely to chase a favorable validation result.
No PCA dimension or new production policy has been selected.

Final evaluation, dataset/model/product documentation, interface and operations
evidence, final release and presentation remain pending. Review the full checklist
before claiming rubric completion. Keep all 20 final-test wells reserved.

Naming preference: methodology-focused PRs and commits; course citations in
supporting documentation and code comments.

## Verification

Full suite completed with 127 passing tests and one notebook-heading check failure. The heading was corrected and that check passed on rerun: all 128 checks are now resolved. The clean-kernel PCA integration passed. Two existing dependency warnings remain. Ruff passed; three notebook figures were visually reviewed.
