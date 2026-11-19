"""Analytic split, boundary, regularization and trusted-reference checks."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from dsm_project.algorithms.student.regression_tree import SquaredErrorTreeRegressor


def test_hand_computable_stump_and_threshold_boundary():
    model = SquaredErrorTreeRegressor(1, 1).fit([[0], [1], [2], [3]], [1, 1, 5, 7])
    assert model.tree_.threshold == 1.5
    assert model.tree_.sse == 27
    np.testing.assert_allclose(
        model.predict([[1.5], [1.50001], [-100], [100]]), [1, 6, 1, 6]
    )
    assert model.depth_ == 1 and model.n_leaves_ == 2


@pytest.mark.parametrize(
    "X,y", [([[1], [1], [1]], [1, 2, 6]), ([[0], [1], [2]], [3, 3, 3])]
)
def test_no_useful_split_returns_mean(X, y):
    model = SquaredErrorTreeRegressor(4, 1).fit(X, y)
    assert model.n_leaves_ == 1
    np.testing.assert_allclose(model.predict(X), np.mean(y))


def test_depth_zero_and_minimum_leaf_stop_growth():
    X, y = np.arange(10).reshape(-1, 1), np.arange(10)
    for model in [SquaredErrorTreeRegressor(0, 1), SquaredErrorTreeRegressor(4, 6)]:
        model.fit(X, y)
        np.testing.assert_allclose(model.predict(X), 4.5)


def test_matches_reference_without_split_ties():
    rng = np.random.default_rng(577)
    X = rng.normal(size=(180, 3))
    y = 3 * (X[:, 0] > 0.3) + X[:, 1] ** 2 + rng.normal(0, 0.1, 180)
    ours = SquaredErrorTreeRegressor(3, 8).fit(X, y)
    reference = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=8, random_state=577
    ).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), reference.predict(X), atol=1e-10)
    assert ours.n_leaves_ == reference.get_n_leaves()


def test_leaf_constraints_and_duplicate_features():
    X = np.repeat(np.arange(12), 3).reshape(-1, 1)
    y = np.sin(X[:, 0])
    model = SquaredErrorTreeRegressor(3, 5).fit(X, y)

    def check(node):
        if node.feature is None:
            assert node.n_samples >= 5
            return
        assert node.left.n_samples + node.right.n_samples == node.n_samples
        check(node.left)
        check(node.right)

    check(model.tree_)
    assert model.depth_ <= 3
    assert model.predict(X).min() >= y.min()
    assert model.predict(X).max() <= y.max()


def test_target_offset_and_row_permutation():
    X, y = np.arange(20).reshape(-1, 1), np.repeat([1.0, 5.0, -2.0, 8.0], 5)
    first = SquaredErrorTreeRegressor(3, 2).fit(X, y)
    second = SquaredErrorTreeRegressor(3, 2).fit(X[::-1], y[::-1] + 1e9)
    np.testing.assert_allclose(second.predict(X) - 1e9, first.predict(X))


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_depth": -1},
        {"max_depth": True},
        {"max_depth": 1.5},
        {"min_samples_leaf": 0},
    ],
)
def test_rejects_invalid_controls(parameters):
    with pytest.raises(ValueError):
        SquaredErrorTreeRegressor(**parameters).fit([[0], [1]], [1, 2])


def test_rejects_missing_and_invalid_prediction_features():
    model = SquaredErrorTreeRegressor(1, 1)
    with pytest.raises(NotFittedError):
        model.predict([[0]])
    with pytest.raises(ValueError):
        model.fit([[0], [np.nan]], [0, 1])
    model.fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([[0, 1]])
    model.fit([[0, 1], [1, 2]], [4, 4])
    assert model.n_features_in_ == 2 and model.n_leaves_ == 1
