# Course alignment and update review

Reviewed November 19, 2026 against professor course commit
`b697d54af35e43f9d5032d01d77fbc291d947aeb` and project-template commit `27ba89c`.

## Implementation references

| Project method | Course source | Approach to preserve |
| --- | --- | --- |
| Linear regression | Lecture 10, `00-gradient-descent-from-functions-to-neuron.ipynb` | Linear prediction with separate bias, mean squared error, full-batch gradients, negative-gradient updates, loss history, finite-difference verification. |
| Logistic regression | Lecture 11, `00-logistic-model-and-decision-policy.ipynb` | Logistic probabilities, stable binary cross-entropy, gradient training, decision threshold selected using development data, classification metrics separate from probability estimates. |
| Regression tree | Lecture 14, `01-regression-trees.ipynb`; `src/rice_dsm/ml/trees.py` | Squared-error recursive partitioning, mean prediction in each leaf, depth and minimum leaf constraints, validation-based complexity selection, baseline and trusted-library comparisons. |

Reusable project estimators belong in `src/dsm_project/algorithms/student/`.
Notebooks import the tested implementation. Preserve attribution for any adapted
course code. Scikit-learn comparisons verify behavior; they do not replace the
three planned student implementations. Standardization and any other learned
preprocessing are fitted on training wells only.

The course examples commonly split observations. FORCE requires a deliberate
dataset-specific adaptation: split whole wells, because neighboring depth rows
are dependent. The reserved test wells remain sealed until model selection ends.

## Changes since the previous review

- `46166e9`: Lecture 15 now includes released random-forest, boosting, and
  voting/stacking notebooks and a tested ensemble implementation. These are
  additional references, not an automatic expansion of the approved scope.
- `b697d54`: the course pins `ipykernel>=6.29,<7` and
  `jupyter-client>=8.6,<8.7` to address local VS Code kernel readiness failures.
  Apply these constraints to the fresh project environment.
- The project-template checkout remains at `27ba89c`; no newer template change
  was returned by the upstream fetch.

The broader applicable-algorithm wording in the template remains a final rubric
review item. The approved core is linear regression, logistic regression, and a
regression tree; no claim of full final-project compliance is made at setup time.

Sources: https://github.com/RandyRDavila/Data-Science-Machine-Learning-Fall-2026
and https://github.com/RandyRDavila/Data-Science-Machine-Learning-Project-Template.
