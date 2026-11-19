# Session checkpoint

Updated November 19, 2026. Owner: Liam Mora.
Working folder: `C:\Users\lmora\INDE577\force-well-interpretation`.
Branch: `feature/force-data`.

## Completed

- Fresh professor-template project and approved three-method scope saved.
- User published starting main branch to lmora1278/force-well-interpretation.
- Course alignment reviewed through b697d54; template at 27ba89c.
- Official Python 3.12 environment and stable notebook kernel configured.
- FORCE adapter added with archive checksum and measurement validation.
- Audited archive stored in ignored data/local/force-train.zip.
- Whole-well split: 58 training, 20 validation, 20 reserved final-test wells.
- Task-specific complete-case filtering preserves missing targets.
- Template row-splitting supervised evaluator rejects FORCE.
- Verification: 75 tests passed, two existing dependency warnings; Ruff passed.
- Full archive validation succeeded. Coverage is recorded in FORCE_DATA_CHECK.md.

## Next

User reviews and commits the local data milestone and pushes feature/force-data.
No commit or push of this milestone was performed during implementation.
Then implement bounded development sampling, training-only preprocessing, and
linear regression following Lecture 10; use a dedicated whole-well runner.
Do not score reserved final-test wells during model selection.

The existing Wine examples are template demonstrations, not FORCE findings.
No FORCE model has yet been fitted in the new project. Read PROJECT_OVERVIEW.md,
COURSE_ALIGNMENT.md and FORCE_DATA_CHECK.md before continuing.
