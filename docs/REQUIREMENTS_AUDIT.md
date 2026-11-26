# Requirements audit before final evaluation

Reviewed November 25, 2026 by artifact inspection. Owner: Liam Mora.
Sources: ASSIGNMENT.md, PROJECT_CHECKLIST.md, NOTEBOOK_CASE_STUDY_STANDARD.md,
PROJECT_OVERVIEW.md and PROJECT_PLAN.md. This is an audit of the local supplied
requirements, not a claim that newer online course instructions were checked.
The recorded syllabus weights/minimum are summarized in PROJECT_OVERVIEW.md;
the original syllabus was not re-extracted in this review.

| Requirement | Evidence/status | Remaining action |
| --- | --- | --- |
| Coherent question and scope | Project overview and five applied notebooks | Consolidate final README and presentation |
| Data access, provenance and whole-well boundaries | FORCE adapter, manifest, data report and synthetic tests | Complete DATASET_CARD.md and reproducible acquisition instructions |
| Regression, classification, representation | Custom linear/tree, logistic, supporting library PCA | Preserve task limits and custom/library distinctions |
| Correct custom implementations | Analytic, gradient, trusted-reference and failure tests | Include final integrated checks |
| Clean-kernel case studies | Notebooks 01–05 and execution tests | Final evaluation/product narrative; review unused template notebook |
| Baselines, losing candidates and failures | Mean/median/prior comparisons and method/robustness reports | Preserve all in final evidence |
| Uncertainty and dependence | Whole-well repeats and per-well diagnostics | Final paired well summaries; no independent-row claims |
| Selection before test access | EVALUATION_PLAN.md prepared; 20 wells unscored | Commit/merge, then implement and verify frozen runner |
| Product and dataset contracts | PRODUCT_CONTRACT.md and DATASET_CARD.md still contain prompts | Replace with FORCE-specific content |
| Model card and versioned artifact | Template bundle infrastructure only; MODEL_CARD.md still prompts | FORCE artifact with evaluated identity, metadata and round-trip tests |
| Tested prediction API and UI | Generic template service/client exists | Named FORCE inputs, units, support rejection, model identity, tests |
| Logs, monitoring, rollback, incident | Template facilities; OPERATIONS.md still prompts | Exercise a real bounded failure and save diagnosis/recovery evidence |
| Solo ownership | TEAM_OWNERSHIP.md still prompts | Record lmora1278 responsibilities and actual integration decisions |
| Repository process | Feature branches, commits, PRs and checks reported through workflow | Final CI, clean-clone verification, issues/review evidence as applicable |
| Tagged reproducible release | Not yet established | Source/config/data/model identities, release instructions and tag |
| Presentation | Not inspected or completed here | Prepare explanation of mathematics, results, failures and decisions |

## Algorithm scope: numerical minimum versus broader wording

The recorded syllabus minimum is two implementations; three custom core methods
exist. The template also says every applicable course algorithm. This wording
is not automatically satisfied by meeting the numerical minimum.

| Family | Project disposition | Reason and remaining qualification |
| --- | --- | --- |
| Linear regression | Implemented | Transparent sonic relationship and baseline comparison |
| Logistic regression | Implemented | Binary sandstone-label screening |
| Trees | Regression tree implemented | Controlled nonlinear sonic comparison; classification-tree extension omitted |
| Dimensionality reduction | Supporting library PCA implemented | Measurement redundancy and downstream comparison; not a custom fourth method |
| Nearest neighbors | Deferred, not incompatible | Could predict these targets; omitted for bounded scope |
| SVM/kernel methods | Deferred, not incompatible | Could fit these targets; not needed for the current comparison |
| Ensembles | Deferred, not incompatible | Could improve prediction; beyond the controlled shallow-tree study |
| Neural networks | Deferred, not incompatible | Numerically applicable; additional complexity without a new approved question |
| Clustering | No separate clustering question | PCA addresses the chosen representation question; clustering is possible but would require its own scientific justification |

Resolve with the professor whether the coherent three-method scope plus PCA is
acceptable under the broader template wording before claiming full compliance.
Do not relabel useful but deferred algorithms as mathematically incompatible.
Continue the bounded implementation while this compliance item remains open.

## Prioritized finish sequence

1. Merge the evaluation decision record; complete FORCE product/dataset contracts.
2. Build and test final fitting/artifacts/input support without test-label access.
3. Evaluate reserved wells under the committed protocol and complete the model card.
4. Adapt the minimal API/UI; exercise monitoring and rollback.
5. Audit every checklist item, reproducibility and presentation; publish a final release.

No checklist item is marked complete merely because template scaffolding exists.
