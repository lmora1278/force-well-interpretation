"""Affine regression following Lecture 10's full-MSE batch updates.

Reference: Randy Davila, Lecture 10, gradient-descent-from-functions-to-neuron.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def mse_and_gradient(features, targets, weights, bias):
    """Return MSE and its weight/bias derivatives at the supplied parameters."""
    residual = features @ weights + bias - targets
    return (
        float(np.mean(residual**2)),
        2 * features.T @ residual / len(targets),
        float(2 * residual.mean()),
    )


class LectureLinearRegressor(RegressorMixin, BaseEstimator):
    """Fit y = Xw + b with zero initialization and full-batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Finite positive step size. Standardize inputs before fitting.
    max_iterations : int
        Maximum number of full-data parameter updates.
    tolerance : float
        Nonnegative stopping threshold for the joint gradient norm.

    Notes
    -----
    Cost is O(iterations * rows * features); history uses O(iterations) memory.
    Correlated inputs may converge slowly. A finite iteration budget can leave
    the solution short of least squares; inspect converged_ and gradient_norm_.
    """

    def __init__(self, learning_rate=0.05, max_iterations=5000, tolerance=1e-7):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, X, y):
        """Fit validated finite observations and retain initial/final MSE history."""
        for name in (
            "coef_",
            "intercept_",
            "n_features_in_",
            "loss_history_",
            "n_iter_",
            "converged_",
            "gradient_norm_",
        ):
            self.__dict__.pop(name, None)
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        weights, bias = np.zeros(X.shape[1]), 0.0
        history = []
        with np.errstate(over="raise", invalid="raise"):
            try:
                for step in range(self.max_iterations + 1):
                    loss, dw, db = mse_and_gradient(X, y, weights, bias)
                    norm = float(np.linalg.norm(np.r_[dw, db]))
                    if not np.isfinite(loss) or not np.isfinite(norm):
                        raise FloatingPointError
                    history.append(loss)
                    if norm <= self.tolerance or step == self.max_iterations:
                        break
                    weights -= self.learning_rate * dw
                    bias -= self.learning_rate * db
            except FloatingPointError as error:
                raise ValueError(
                    "Optimization diverged; scale inputs or reduce step"
                ) from error
        self.coef_, self.intercept_ = weights, bias
        self.n_features_in_ = X.shape[1]
        self.loss_history_ = np.asarray(history)
        self.n_iter_, self.gradient_norm_ = step, norm
        self.converged_ = norm <= self.tolerance
        return self

    def predict(self, X):
        """Return sonic predictions when trained with a sonic response."""
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        return X @ self.coef_ + self.intercept_
