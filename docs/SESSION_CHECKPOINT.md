# Session checkpoint

Updated November 26, 2026. Owner: Liam Mora.
Branch: feature/final-model-artifacts.

## Completed

- Product/data contracts merged in e57cacb.
- Added frozen candidate fitting, training-only support bounds and provenance.
- Added non-overwriting bundles with integrity-checked trusted local loading.
- Require convergence and a clean source revision for the field-data artifact command.
- Aligned FORCE adapter/config metric declarations with actual MAE selection.
- Saved reproduction steps in FINAL_MODEL_ARTIFACTS.md.
- Synthetic checks passed; full field-data artifact creation is deferred until merge.
- No final-test scoring, commit, push or remote publication performed here.

## Next

Commit/push and merge this milestone after automated checks pass. Pull merged
main locally, then create the next branch. With a clean revision, fit the actual
78 development wells and inspect the saved artifact without scoring the 20
reserved wells. Implement/test final evaluation separately before opening scores.
See EVALUATION_PLAN.md for the immutable evaluation decisions.

API/UI adaptation, model card, operations exercise, ownership documentation,
release and presentation remain. Broader algorithm-scope wording is still open.

## Verification

Focused artifact/data checks: 23 passed. Full suite: 145 passed in 326.41 seconds, with two existing non-blocking dependency warnings. Ruff and whitespace checks passed. The command help entry point was checked without starting a field-data fit.
