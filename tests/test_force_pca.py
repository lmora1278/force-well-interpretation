"""Verify training-only PCA and rotation/reconstruction invariants."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dsm_project.data.force import FEATURES
from dsm_project.experiments.force_pca import fit_representation_models


def fixture():
    rng = np.random.default_rng(577)
    X = rng.normal(size=(180, 3)) * [10, 0.2, 0.1] + [70, 2.4, 0.3]
    frame = pd.DataFrame(X, columns=FEATURES)
    frame["DTC"] = 20 + X @ np.array([0.1, 3.0, 5.0])
    return frame


def test_training_only_scaling_and_transform_preserves_fit():
    training = fixture()
    model = fit_representation_models(training)[2]
    scaler, pca = model.steps[0][1], model.steps[1][1]
    np.testing.assert_allclose(scaler.mean_, training.loc[:, FEATURES].mean())
    old_components = pca.components_.copy()
    old_mean = scaler.mean_.copy()
    model.predict(training.loc[:, FEATURES] + 100)
    np.testing.assert_array_equal(pca.components_, old_components)
    np.testing.assert_array_equal(scaler.mean_, old_mean)


def test_full_rotation_reconstructs_and_matches_uncompressed_regression():
    frame = fixture()
    models = fit_representation_models(frame)
    X = frame.loc[:, FEATURES]
    full = models[3]
    scaled = full.steps[0][1].transform(X)
    pca = full.steps[1][1]
    np.testing.assert_allclose(
        pca.inverse_transform(pca.transform(scaled)), scaled, atol=1e-12
    )
    np.testing.assert_allclose(
        pca.components_ @ pca.components_.T, np.eye(3), atol=1e-12
    )
    reference = LinearRegression().fit(X, frame.DTC)
    np.testing.assert_allclose(full.predict(X), reference.predict(X), atol=1e-5)
    errors = []
    for model in models.values():
        pca = model.steps[1][1]
        errors.append(
            np.mean((scaled - pca.inverse_transform(pca.transform(scaled))) ** 2)
        )
    assert errors[0] >= errors[1] >= errors[2]


def test_pca_axes_do_not_depend_on_target_labels():
    first = fixture()
    altered = first.assign(DTC=first.DTC.to_numpy()[::-1])
    a = fit_representation_models(first)[2].steps[1][1]
    b = fit_representation_models(altered)[2].steps[1][1]
    np.testing.assert_allclose(a.components_, b.components_)
    np.testing.assert_allclose(a.explained_variance_ratio_, b.explained_variance_ratio_)
