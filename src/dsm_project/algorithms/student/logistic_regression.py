"""Binary logistic regression using the Lecture 11 batch-gradient approach.

Reference: Randy Davila, 00-logistic-model-and-decision-policy.ipynb.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def stable_sigmoid(logits):
    """Evaluate sigmoid without overflowing the exponential for extreme scores."""
    values = np.asarray(logits, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def logistic_loss_and_gradient(design, targets, parameters):
    """Return mean unregularized cross-entropy and its full-batch gradient.

    The design includes an intercept column. Softplus evaluated separately for
    each binary class avoids both log(0) and subtraction of large similar values.
    """
    matrix, target = check_X_y(design, targets, dtype=float, y_numeric=True)
    weights = np.asarray(parameters, dtype=float)
    if weights.shape != (matrix.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("parameters must be finite and align with design columns")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("targets must contain only 0 and 1")
    logits = matrix @ weights
    probabilities = stable_sigmoid(logits)
    loss = np.mean(
        (1 - target) * np.logaddexp(0, logits) + target * np.logaddexp(0, -logits)
    )
    return float(loss), matrix.T @ (probabilities - target) / len(target)


class BatchLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Fit binary labels 0/1 using unregularized batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Finite positive update size; scale features before fitting.
    max_iterations : int
        Positive maximum number of full-batch updates.
    tolerance : float
        Finite positive stopping threshold for the gradient norm.
    threshold : float
        Decision cutoff in (0, 1); predict uses probability >= threshold.

    Notes
    -----
    Training costs O(iterations * rows * features), with O(iterations) trace
    storage. Complete separation may prevent a finite maximum-likelihood
    solution. Inspect converged_ rather than assuming the budget suffices.
    Probabilities are model estimates, not a claim of calibration.
    """

    def __init__(
        self, learning_rate=0.2, max_iterations=20000, tolerance=1e-7, threshold=0.5
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.threshold = threshold

    def _check_threshold(self):
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one")

    def fit(self, X, y):
        """Fit both binary classes and retain initial and final loss evidence."""
        for name in (
            "coef_",
            "intercept_",
            "classes_",
            "n_features_in_",
            "loss_history_",
            "gradient_norm_",
            "converged_",
            "n_iter_",
        ):
            self.__dict__.pop(name, None)
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("training requires both binary classes 0 and 1")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self._check_threshold()
        design = np.column_stack([np.ones(len(X)), X])
        parameters = np.zeros(design.shape[1])
        history = []
        with np.errstate(over="raise", invalid="raise"):
            try:
                for step in range(self.max_iterations + 1):
                    loss, gradient = logistic_loss_and_gradient(design, y, parameters)
                    norm = float(np.linalg.norm(gradient))
                    if not np.isfinite(loss) or not np.isfinite(norm):
                        raise FloatingPointError
                    history.append(loss)
                    if norm <= self.tolerance or step == self.max_iterations:
                        break
                    parameters -= self.learning_rate * gradient
            except FloatingPointError as error:
                raise ValueError(
                    "Optimization diverged; check scaling and step size"
                ) from error
        self.intercept_ = np.array([parameters[0]])
        self.coef_ = parameters[1:].reshape(1, -1)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.n_iter_, self.gradient_norm_ = step, norm
        self.converged_ = norm <= self.tolerance
        self.loss_history_ = np.asarray(history)
        return self

    def decision_function(self, X):
        """Return affine log-odds scores for finite, aligned input features."""
        check_is_fitted(self, "coef_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        """Return columns P(class 0), P(class 1), aligned with classes_."""
        probability = stable_sigmoid(self.decision_function(X))
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        """Apply the explicit probability cutoff without changing fitted weights."""
        self._check_threshold()
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
