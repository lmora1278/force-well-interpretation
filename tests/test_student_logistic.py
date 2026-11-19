"""Mathematical, numerical and evaluation-boundary checks for binary screening."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from dsm_project.algorithms.student.logistic_regression import (
    BatchLogisticClassifier,
    logistic_loss_and_gradient,
    stable_sigmoid,
)
from dsm_project.experiments.force_logistic import (
    classification_metrics,
    fit_logistic_models,
)


def test_extreme_scores_are_stable():
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(stable_sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
        assert stable_sigmoid(0.0) == 0.5
        loss, gradient = logistic_loss_and_gradient(
            np.array([[1.0, -1000.0], [1.0, 1000.0]]), [1, 0], [0, 1]
        )
    assert loss == pytest.approx(1000)
    assert np.isfinite(gradient).all()


def test_gradient_matches_finite_differences():
    design = np.array(
        [[1.0, -1.2, 0.4], [1.0, 0.3, -0.8], [1.0, 1.7, 1.1], [1.0, 0.2, 0.9]]
    )
    y, weights = np.array([0, 1, 1, 0]), np.array([0.15, -0.35, 0.22])
    _, actual = logistic_loss_and_gradient(design, y, weights)
    for j, basis in enumerate(np.eye(3)):
        plus = logistic_loss_and_gradient(design, y, weights + 1e-6 * basis)[0]
        minus = logistic_loss_and_gradient(design, y, weights - 1e-6 * basis)[0]
        assert actual[j] == pytest.approx((plus - minus) / 2e-6, abs=1e-8)


def test_intercept_only_recovers_prevalence_and_history():
    X, y = np.zeros((20, 2)), np.array([1] * 5 + [0] * 15)
    fitted = BatchLogisticClassifier().fit(X, y)
    assert fitted.converged_
    np.testing.assert_allclose(fitted.predict_proba(X)[:, 1], 0.25, atol=1e-6)
    assert fitted.intercept_[0] == pytest.approx(np.log(1 / 3), abs=1e-5)
    assert len(fitted.loss_history_) == fitted.n_iter_ + 1
    assert np.all(np.diff(fitted.loss_history_) <= 1e-12)
    final, _ = logistic_loss_and_gradient(
        np.column_stack([np.ones(len(X)), X]),
        y,
        np.r_[fitted.intercept_, fitted.coef_[0]],
    )
    assert final == pytest.approx(fitted.loss_history_[-1])


def test_probabilities_match_unregularized_reference():
    rng = np.random.default_rng(577)
    X = rng.normal(size=(300, 3))
    y = rng.binomial(1, stable_sigmoid(X @ [1.0, -0.7, 0.4] - 0.5))
    model = BatchLogisticClassifier().fit(X, y)
    trusted = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)
    assert model.converged_
    np.testing.assert_allclose(
        model.predict_proba(X), trusted.predict_proba(X), atol=1e-5
    )
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict(X), model.predict_proba(X)[:, 1] >= 0.5)


def test_threshold_changes_decisions_not_probabilities():
    X = np.zeros((10, 1))
    model = BatchLogisticClassifier().fit(X, [1] * 3 + [0] * 7)
    before = model.predict_proba(X)
    assert (model.predict(X) == 0).all()
    model.set_params(threshold=0.2)
    assert (model.predict(X) == 1).all()
    np.testing.assert_array_equal(model.predict_proba(X), before)
    model.set_params(threshold=1)
    with pytest.raises(ValueError, match="threshold"):
        model.predict(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": np.nan},
        {"tolerance": 0},
        {"max_iterations": True},
        {"max_iterations": 1.2},
        {"threshold": -0.1},
    ],
)
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        BatchLogisticClassifier(**kwargs).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    "X,y",
    [
        ([[0], [1]], [0, 0]),
        ([[0], [1]], [0, 2]),
        ([[0], [np.nan]], [0, 1]),
        ([[0], [1]], [0]),
    ],
)
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        BatchLogisticClassifier().fit(X, y)


def test_unfitted_invalid_shape_budget_and_refit():
    model = BatchLogisticClassifier(max_iterations=1)
    with pytest.raises(NotFittedError):
        model.predict([[0]])
    model.fit([[-1], [1]], [0, 1])
    assert not model.converged_ and model.n_iter_ == 1
    with pytest.raises(ValueError):
        model.predict([[0, 1]])
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [0, 0])
    with pytest.raises(NotFittedError):
        model.predict([[0]])


def test_known_confusion_matrix_and_single_class_reporting():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["balanced_accuracy"] == 0.5
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    result = classification_metrics([0, 0], [0.1, 0.2])
    assert result["balanced_accuracy"] is None
    assert result["average_precision"] is None
    assert result["recall"] is None
    assert result["precision"] == 0
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.1, 1.2])


def test_scaling_and_baseline_learn_only_training_sample():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=["GR", "RHOB", "NPHI"])
    frame["sandstone"] = [1] * 20 + [0] * 60
    models = fit_logistic_models(frame)
    scaler = models["batch_logistic"].steps[0][1]
    mean = frame[["GR", "RHOB", "NPHI"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, mean)
    validation = pd.DataFrame([[100, 200, 300]], columns=["GR", "RHOB", "NPHI"])
    models["batch_logistic"].predict_proba(validation)
    np.testing.assert_allclose(scaler.mean_, mean)
    np.testing.assert_allclose(
        models["prior_baseline"].predict_proba(validation), [[0.75, 0.25]]
    )
