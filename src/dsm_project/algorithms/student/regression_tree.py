"""Greedy squared-error tree following Lecture 14, regression-trees notebook.

Uses the course's sorted-prefix SSE calculation and mean leaves. Depth and
minimum leaf size provide structural regularization; no post-pruning is used.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class RegressionNode:
    """Inspectable fitted region; counts refer only to training observations."""

    prediction: float
    n_samples: int
    sse: float
    feature: int | None = None
    threshold: float | None = None
    left: "RegressionNode | None" = None
    right: "RegressionNode | None" = None


class SquaredErrorTreeRegressor(RegressorMixin, BaseEstimator):
    """Deterministic CART-style regression with finite, complete numeric inputs.

    Ties retain the first feature and lowest threshold encountered. Root depth
    is zero. A split must strictly reduce SSE. Predictions cannot extrapolate
    outside the training target range. This is a single tree, not an ensemble.
    """

    def __init__(self, max_depth=3, min_samples_leaf=50):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        for name, minimum in (("max_depth", 0), ("min_samples_leaf", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        for name in ("tree_", "n_features_in_", "depth_", "n_leaves_"):
            self.__dict__.pop(name, None)

        def grow(indices, depth):
            targets = y[indices]
            mean = float(targets.mean())
            centered = targets - mean
            parent_sse = float(centered @ centered)
            if not np.isfinite(parent_sse):
                raise ValueError("Targets exceed stable squared-error range")
            node = RegressionNode(mean, len(indices), parent_sse)
            if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
                return node, depth, 1
            best_loss, best = parent_sse, None
            # Centering preserves SSE while reducing cancellation for large offsets.
            for feature in range(X.shape[1]):
                order = np.argsort(X[indices, feature], kind="stable")
                values, ordered_y = X[indices[order], feature], centered[order]
                positions = np.arange(
                    self.min_samples_leaf, len(indices) - self.min_samples_leaf + 1
                )
                positions = positions[values[positions - 1] < values[positions]]
                if not len(positions):
                    continue
                sums, squares = np.cumsum(ordered_y), np.cumsum(ordered_y**2)
                left_sum, left_sq = sums[positions - 1], squares[positions - 1]
                right_sum, right_sq = sums[-1] - left_sum, squares[-1] - left_sq
                losses = np.maximum(left_sq - left_sum**2 / positions, 0)
                losses += np.maximum(
                    right_sq - right_sum**2 / (len(indices) - positions), 0
                )
                k = int(np.argmin(losses))
                loss = float(losses[k])
                if loss < best_loss - 1e-12:
                    position = positions[k]
                    low, high = values[position - 1], values[position]
                    threshold = low / 2 + high / 2
                    # Adjacent floating-point values may round their midpoint up.
                    if threshold >= high or threshold < low:
                        threshold = low
                    best_loss = loss
                    best = (
                        feature,
                        float(threshold),
                        indices[order[:position]],
                        indices[order[position:]],
                    )
            if best is None:
                return node, depth, 1
            node.feature, node.threshold, left_ids, right_ids = best
            node.left, left_depth, left_leaves = grow(left_ids, depth + 1)
            node.right, right_depth, right_leaves = grow(right_ids, depth + 1)
            return node, max(left_depth, right_depth), left_leaves + right_leaves

        self.tree_, self.depth_, self.n_leaves_ = grow(np.arange(len(y)), 0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted data")
        predictions = np.empty(len(X))

        def visit(node, indices):
            if node.feature is None:
                predictions[indices] = node.prediction
                return
            left = X[indices, node.feature] <= node.threshold
            visit(node.left, indices[left])
            visit(node.right, indices[~left])

        visit(self.tree_, np.arange(len(X)))
        return predictions
