# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: C:\Users\lmora\INDE577\force-well-interpretation.
Branch: feature/regression-notebook.

## Completed

- Data and linear-regression milestones merged into main (b20772f).
- Added notebooks/01-sonic-regression.ipynb with saved executed outputs.
- Notebook imports package experiment code and uses only development wells.
- Includes source, scope, training-only scaling, mathematical model, verification,
  convergence, pooled metrics, prediction/residual density plots, and per-well errors.
- Linear regression improves MAE in 19 of 20 validation wells; high DTC values
  show systematic underprediction. Limitations and unfinished analyses are explicit.
- Experiment now returns results to the notebook and retains CLI JSON output.
- Clean-kernel notebook execution passed; all three plots visually inspected.
- Full local suite: 88 passed, two existing dependency warnings. Ruff passed.
- CI checks notebook structure; full FORCE execution is explicitly skipped when
  the ignored field archive is absent. Local full-data execution passed.

## Next

User reviews saved notebook, commits and pushes feature/regression-notebook,
then opens a pull request and waits for checks before merging.
No commit or push of this milestone was performed during implementation.
Next model: logistic regression for sandstone-labelled interval screening.
Shallow regression tree, PCA, rock-type diagnostics and repeated-split sensitivity
remain planned. Do not score reserved final-test wells before freezing choices.

Naming preference: PR titles, descriptions and commit messages describe methods
and changes. Specific course references remain in code comments and documentation.
