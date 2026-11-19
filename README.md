# Reconstructing Missing Sonic Logs in Unseen Wells

A machine-learning study using FORCE 2020, by Liam Mora, for Rice University INDE 577.

## Research question

How accurately can gamma ray, bulk density, and neutron porosity measurements estimate compressional sonic slowness in wells excluded from training, and how does performance vary between wells and rock types?

## Planned methods

- Linear regression using batch gradient descent.
- A shallow regression tree for nonlinear sonic reconstruction.
- Logistic regression for the supporting sandstone classification question.

Whole wells are held out to avoid leakage between neighboring depth observations.

## Project status

Initial setup and course-alignment review. FORCE-specific implementations and final results are not yet complete. Existing Wine examples demonstrate the inherited template only.

- [Approved overview](docs/PROJECT_OVERVIEW.md)
- [Course alignment](docs/COURSE_ALIGNMENT.md)
- [Resume instructions](docs/START_NEXT_SESSION.md)
- [Professor template guide](TEMPLATE_GUIDE.md)

## Sources and attribution

Dataset: [FORCE 2020](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition).
Project scaffolding: [Randy Davila's course template](https://github.com/RandyRDavila/Data-Science-Machine-Learning-Project-Template), retained under its included license.
