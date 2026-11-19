"""Independent mathematical checks and training-boundary tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from dsm_project.algorithms.student.linear_regression import (
    LectureLinearRegressor,
    mse_and_gradient,
)
from dsm_project.experiments.force_linear import (
    fit_linear_models,
    sample_training_wells,
)


def test_gradients_match_central_differences():
    X = np.array([[1.0, 2.0], [-1.0, 0.0], [2.0, -3.0]])
    y, w, b = np.array([2.0, 1.0, -4.0]), np.array([0.2, -0.4]), 0.7
    _, dw, db = mse_and_gradient(X, y, w, b)
    eps = 1e-6
    for j in range(2):
        delta = np.eye(2)[j] * eps
        numeric = (
            mse_and_gradient(X, y, w + delta, b)[0]
            - mse_and_gradient(X, y, w - delta, b)[0]
        ) / (2 * eps)
        assert dw[j] == pytest.approx(numeric, rel=1e-6)
    numeric = (
        mse_and_gradient(X, y, w, b + eps)[0] - mse_and_gradient(X, y, w, b - eps)[0]
    ) / (2 * eps)
    assert db == pytest.approx(numeric, rel=1e-6)


def test_exact_affine_solution_and_history():
    X = np.linspace(-1, 1, 31).reshape(-1, 1)
    y = 3 * X[:, 0] + 7
    model = LectureLinearRegressor().fit(X, y)
    assert model.coef_[0] == pytest.approx(3, abs=1e-5)
    assert model.intercept_ == pytest.approx(7, abs=1e-5)
    assert model.converged_
    assert len(model.loss_history_) == model.n_iter_ + 1
    assert np.all(np.diff(model.loss_history_) <= 1e-10)
    assert model.loss_history_[-1] == pytest.approx(
        np.mean((model.predict(X) - y) ** 2), abs=1e-12
    )


def test_noisy_multivariate_agrees_with_least_squares():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(120, 3))
    y = X @ np.array([2.0, -3.0, 0.4]) + 4 + rng.normal(size=120)
    model = LectureLinearRegressor().fit(X, y)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-5)


def test_constant_features_and_response():
    X = np.zeros((10, 2))
    model = LectureLinearRegressor().fit(X, np.full(10, 6.0))
    np.testing.assert_allclose(model.predict(X), 6, atol=1e-5)


@pytest.mark.parametrize(
    "parameters",
    [
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"tolerance": -1},
        {"max_iterations": 1.5},
        {"max_iterations": True},
    ],
)
def test_bad_parameters(parameters):
    with pytest.raises(ValueError):
        LectureLinearRegressor(**parameters).fit([[1], [2]], [1, 2])


def test_invalid_inputs_unfitted_and_exhausted_budget():
    with pytest.raises(NotFittedError):
        LectureLinearRegressor().predict([[1]])
    with pytest.raises(ValueError):
        LectureLinearRegressor().fit([[np.nan]], [1])
    with pytest.raises(ValueError):
        LectureLinearRegressor().fit([[1], [2]], [1])
    model = LectureLinearRegressor(max_iterations=1).fit([[0], [1]], [3, 7])
    assert not model.converged_
    assert model.n_iter_ == 1
    with pytest.raises(ValueError):
        model.predict([[1, 2]])


def test_sample_is_bounded_reproducible_and_scaling_uses_training_only():
    frame = pd.DataFrame(
        {
            "WELL": ["A"] * 6 + ["B"] * 6,
            "DEPTH_MD": list(range(6)) * 2,
            "GR": np.arange(12.0),
            "RHOB": np.arange(12.0) + 2,
            "NPHI": np.arange(12.0) / 10,
            "DTC": np.arange(12.0) + 50,
        }
    )
    sample = sample_training_wells(frame, rows_per_well=3)
    other = sample_training_wells(frame.sample(frac=1, random_state=7), rows_per_well=3)
    pd.testing.assert_frame_equal(sample, other)
    assert sample.groupby("WELL").size().tolist() == [3, 3]
    models = fit_linear_models(sample)
    scaler = models["lecture_linear"].steps[0][1]
    np.testing.assert_allclose(scaler.mean_, sample[["GR", "RHOB", "NPHI"]].mean())
    before = scaler.mean_.copy()
    models["lecture_linear"].predict(
        pd.DataFrame([[1e6, 1e6, 1e6]], columns=["GR", "RHOB", "NPHI"])
    )
    np.testing.assert_array_equal(scaler.mean_, before)
