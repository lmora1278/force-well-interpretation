# Student project case studies

This directory holds the applied scientific narrative of the project. Begin by reading
[`ASSIGNMENT.md`](../ASSIGNMENT.md), then duplicate
`00-guided-case-study-template.ipynb` for each coherent investigation.

## What “one notebook” means

A notebook is not automatically one algorithm and it is not automatically one
course week. It is one complete, readable investigation that uses data
analysis and machine learning to answer one real-world or scientific question.
It may:

- teach and evaluate one algorithm in depth;
- compare several algorithms that answer the same question; or
- compose methods into a justified pipeline.

For example, a regression notebook might compare a student linear model with a
tree ensemble because the scientific question is whether nonlinear structure
materially improves prediction. A representation notebook might use PCA before
clustering because the question concerns stable low-dimensional structure.

Split the work when methods answer unrelated questions. Combine it when the
comparison itself is the story. Do not force every course algorithm onto data
for which it is meaningless; explain the incompatibility in the project scope.

## How to use the template

1. Copy `00-guided-case-study-template.ipynb` to a numbered, descriptive name,
   such as `02-linear-and-tree-regression.ipynb`.
2. Replace every **STUDENT WORK** prompt. Do not leave the example question,
   configuration, figure, metric, or interpretation unchanged.
3. Implement reusable behavior in `src/dsm_project/` and tests in `tests/`.
4. Import that behavior into the notebook through the public package API.
5. Restart the kernel and run the complete notebook from top to bottom.
6. Ask a teammate to review both the scientific argument and package code.

The notebook must stand on its own: define notation, explain why code is being
run, interpret every important result, and state limitations. A reader should
not need the lecture or commit history to understand the investigation.

Reference reports under `examples/` show completed course-topic stories. They
are models of reasoning, not submissions. Follow the full
[`notebook case-study standard`](../docs/NOTEBOOK_CASE_STUDY_STANDARD.md).

## FORCE regression study

Open 01-sonic-regression.ipynb with the DSM Project kernel. Saved outputs show the development analysis. Full reruns require the audited archive in data/local/force-train.zip. The 20 final-test wells remain excluded. See docs/LINEAR_REGRESSION.md for the method and reproduction steps.

## FORCE sandstone classification study

Open 02-sandstone-classification.ipynb with the DSM Project kernel. It explains class imbalance, convergence, confusion patterns, threshold trade-offs and per-well variation. Run all cells with the audited local archive available. Committed outputs are cleared per the notebook standard; an executed review copy is saved locally under artifacts/force_logistic/02-sandstone-classification.executed.ipynb. The primary threshold remains 0.5 and final-test wells remain excluded.

## FORCE tree comparison

03-tree-comparison.ipynb compares the shallow tree, linear regression and mean baseline on development wells. It covers complexity selection, residuals, per-well failures and extrapolation. Outputs are cleared; the executed review copy is local under artifacts/force_tree/.

## FORCE PCA representation

04-pca-representation.ipynb examines standardized measurement redundancy, two-axis visualization, reconstruction loss and downstream sonic prediction. PCA fits training predictors only. Outputs are cleared; the executed review copy is local under artifacts/force_pca/.

## FORCE robustness study

05-robustness-analysis.ipynb compares fixed models across five overlapping whole-well development splits, with per-well and rock-type failures, eligibility coverage, and sandstone-screening limits. No final-test scoring or retuning occurs. Outputs are cleared; the executed review copy and figures are local under artifacts/force_robustness/.
