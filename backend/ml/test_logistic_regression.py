"""Unit Tests for Manual Logistic Regression Implementation.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.1 - Manual Logistic Regression Classifier Implementation.

Strict Constraints:
- Pure Python and NumPy/Pandas testing only.
- Strictly NO Scikit-learn (sklearn) is imported or used.
- Uses Python's built-in unittest module.
- All test fixtures are isolated, small, and deterministic.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from backend.ml.eda import CANONICAL_FEATURES, TARGET_COLUMN
from backend.ml.logistic_regression import ManualLogisticRegression


class TestManualLogisticRegression(unittest.TestCase):
    """Automated unit test suite verifying mathematical correctness, prediction behavior,
    input validation, and data integrity for ManualLogisticRegression.
    """

    # -------------------------------------------------------------------------
    # Mathematical Correctness Tests (Requirements 1 - 7)
    # -------------------------------------------------------------------------

    def test_01_sigmoid_zero_is_half(self) -> None:
        """Requirement 1: Verify sigmoid output for z = 0 is exactly 0.5."""
        clf = ManualLogisticRegression()

        # Scalar float input
        self.assertEqual(clf._sigmoid(0.0), 0.5)

        # Integer input
        self.assertEqual(clf._sigmoid(0), 0.5)

        # NumPy 1D array input
        z_arr = np.array([0.0])
        res = clf._sigmoid(z_arr)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (1,))
        self.assertAlmostEqual(res[0], 0.5, places=12)

    def test_02_sigmoid_extreme_values_stability(self) -> None:
        """Requirement 2: Verify sigmoid handles extreme positive and negative values safely."""
        clf = ManualLogisticRegression()

        # Extreme positive logits should saturate asymptotically to 1.0 without overflow
        z_large_pos = np.array([100.0, 500.0, 709.0, 1000.0, 1e9])
        prob_pos = clf._sigmoid(z_large_pos)
        self.assertTrue(np.all(np.isfinite(prob_pos)))
        self.assertTrue(np.all(prob_pos >= 0.5))
        self.assertTrue(np.all(prob_pos <= 1.0))
        np.testing.assert_allclose(prob_pos, 1.0, atol=1e-12)

        # Extreme negative logits should saturate asymptotically to 0.0 without underflow/warnings
        z_large_neg = np.array([-100.0, -500.0, -709.0, -1000.0, -1e9])
        prob_neg = clf._sigmoid(z_large_neg)
        self.assertTrue(np.all(np.isfinite(prob_neg)))
        self.assertTrue(np.all(prob_neg >= 0.0))
        self.assertTrue(np.all(prob_neg <= 0.5))
        np.testing.assert_allclose(prob_neg, 0.0, atol=1e-12)

        # Ensure range [0, 1] is strictly maintained across arbitrary values
        z_mixed = np.array([-1e12, -10.0, -1.0, 0.0, 1.0, 10.0, 1e12])
        prob_mixed = clf._sigmoid(z_mixed)
        self.assertTrue(np.all((prob_mixed >= 0.0) & (prob_mixed <= 1.0)))

    def test_03_initial_cross_entropy_is_ln2(self) -> None:
        """Requirement 3: Verify initial binary cross-entropy with zero weights and bias is approximately ln(2)."""
        # When w = 0 and b = 0, z = 0 and y_hat = sigma(0) = 0.5 for all samples.
        # Loss per sample = - [y * log(0.5) + (1 - y) * log(0.5)] = - log(0.5) = ln(2) ~= 0.693147.
        expected_ln2 = math.log(2.0)

        # Test directly via _compute_loss
        clf = ManualLogisticRegression()
        y_fixture = np.array([1.0, 0.0, 1.0, 0.0])
        y_hat_half = np.full(len(y_fixture), 0.5)
        z_zeros = np.zeros(len(y_fixture))

        # Via logit-based formula
        loss_logit = clf._compute_loss(y_fixture, y_hat_half, z=z_zeros)
        self.assertAlmostEqual(loss_logit, expected_ln2, places=10)

        # Via probability-based formula
        loss_prob = clf._compute_loss(y_fixture, y_hat_half, z=None)
        self.assertAlmostEqual(loss_prob, expected_ln2, places=10)

        # Test via fit() initialization tracking
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [5.0, 6.0, 7.0, 8.0]})
        y_series = pd.Series([1, 0, 1, 0])
        clf_fitted = ManualLogisticRegression(epochs=1)
        clf_fitted.fit(X_df, y_series)

        self.assertGreater(len(clf_fitted.loss_history_), 0)
        self.assertAlmostEqual(clf_fitted.loss_history_[0], expected_ln2, places=10)

    def test_04_forward_propagation_correctness(self) -> None:
        """Requirement 4: Verify forward propagation produces correct logits and probabilities."""
        # Controlled fixture:
        # X = [[1.0, 2.0], [-1.0, 0.5]]
        # w = [0.5, -0.2], b = 0.1
        # z1 = 1.0*0.5 + 2.0*(-0.2) + 0.1 = 0.5 - 0.4 + 0.1 = 0.2
        # z2 = -1.0*0.5 + 0.5*(-0.2) + 0.1 = -0.5 - 0.1 + 0.1 = -0.5
        # y_hat_1 = 1 / (1 + exp(-0.2))
        # y_hat_2 = 1 / (1 + exp(0.5))
        clf = ManualLogisticRegression()
        clf.weights_ = np.array([0.5, -0.2])
        clf.bias_ = 0.1

        X_mat = np.array([[1.0, 2.0], [-1.0, 0.5]])
        z, y_hat = clf._forward(X_mat)

        expected_z = np.array([0.2, -0.5])
        expected_y_hat = np.array([1.0 / (1.0 + math.exp(-0.2)), 1.0 / (1.0 + math.exp(0.5))])

        np.testing.assert_allclose(z, expected_z, atol=1e-12)
        np.testing.assert_allclose(y_hat, expected_y_hat, atol=1e-12)

    def test_05_gradient_calculation_matches_manual_math(self) -> None:
        """Requirement 5: Verify gradient calculations match manually calculated analytical values."""
        # Controlled test fixture:
        # X = [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]] (m = 3 samples, n = 2 features)
        # y = [1.0, 0.0, 1.0]
        # At initial zero weights (w = [0, 0], b = 0):
        # z = [0, 0, 0] -> y_hat = [0.5, 0.5, 0.5]
        # error = y_hat - y = [-0.5, 0.5, -0.5]
        # dw = (1/3) * X^T * error
        #    = (1/3) * [ [1*(-0.5) + 2*(0.5) + 3*(-0.5)], [2*(-0.5) + 1*(0.5) + 4*(-0.5)] ]
        #    = (1/3) * [ -1.0, -2.5 ] = [-1/3, -5/6]
        # db = (1/3) * sum(error) = (1/3) * (-0.5 + 0.5 - 0.5) = -0.5 / 3 = -1/6
        clf = ManualLogisticRegression()
        X_mat = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
        y_vec = np.array([1.0, 0.0, 1.0])
        y_hat = np.array([0.5, 0.5, 0.5])

        dw, db = clf._compute_gradients(X_mat, y_vec, y_hat)

        expected_dw = np.array([-1.0 / 3.0, -5.0 / 6.0])
        expected_db = -1.0 / 6.0

        np.testing.assert_allclose(dw, expected_dw, atol=1e-12)
        self.assertAlmostEqual(db, expected_db, places=12)

    def test_06_single_update_matches_manual_math(self) -> None:
        """Requirement 6: Verify a single gradient descent update produces expected weight and bias."""
        # Using the fixture from test_05:
        # Initial: w = [0.0, 0.0], b = 0.0
        # alpha = 0.1, epochs = 1
        # dw = [-1/3, -5/6], db = -1/6
        # w_new = w - alpha * dw = [0, 0] - 0.1 * [-1/3, -5/6] = [0.1/3, 0.5/6] = [1/30, 1/12]
        # b_new = b - alpha * db = 0 - 0.1 * (-1/6) = 0.1/6 = 1/60
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [2.0, 1.0, 4.0]})
        y_series = pd.Series([1, 0, 1])

        clf = ManualLogisticRegression(learning_rate=0.1, epochs=1)
        clf.fit(X_df, y_series)

        expected_weights = np.array([1.0 / 30.0, 1.0 / 12.0])
        expected_bias = 1.0 / 60.0

        np.testing.assert_allclose(clf.weights_, expected_weights, atol=1e-12)
        self.assertAlmostEqual(clf.bias_, expected_bias, places=12)
        self.assertEqual(clf.n_iter_, 1)
        self.assertEqual(len(clf.loss_history_), 1)

    def test_07_loss_decreases_during_training(self) -> None:
        """Requirement 7: Verify loss decreases monotonically during training on a linearly separable dataset."""
        # Simple linearly separable 2D dataset
        X_df = pd.DataFrame({
            "x1": [1.0, 1.5, 2.0, 2.5, 5.0, 5.5, 6.0, 6.5],
            "x2": [1.0, 1.2, 1.8, 2.2, 5.0, 5.2, 6.1, 6.8],
        })
        y_series = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        clf = ManualLogisticRegression(learning_rate=0.2, epochs=80)
        clf.fit(X_df, y_series)

        self.assertEqual(len(clf.loss_history_), 80)
        initial_loss = clf.loss_history_[0]
        mid_loss = clf.loss_history_[40]
        final_loss = clf.loss_history_[-1]

        # Initial loss is ~ln(2)
        self.assertAlmostEqual(initial_loss, math.log(2.0), places=6)

        # Loss decreases progressively
        self.assertLess(mid_loss, initial_loss)
        self.assertLess(final_loss, mid_loss)

        # Verify no divergence / monotonicity (each step decreases or is within numerical tolerance)
        loss_diffs = np.diff(clf.loss_history_)
        self.assertTrue(np.all(loss_diffs <= 1e-6))

    # -------------------------------------------------------------------------
    # Prediction Behavior Tests (Requirements 8 - 12)
    # -------------------------------------------------------------------------

    def test_08_fit_returns_self(self) -> None:
        """Requirement 8: Verify fit() returns the classifier instance (self)."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y_series = pd.Series([0, 1])
        clf = ManualLogisticRegression(epochs=5)
        fitted_instance = clf.fit(X_df, y_series)
        self.assertIs(fitted_instance, clf)

    def test_09_predict_proba_two_columns(self) -> None:
        """Requirement 9: Verify predict_proba() returns a 2-column array (P(y=0), P(y=1))."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 1.0, 4.0, 3.0]})
        y_series = pd.Series([0, 1, 0, 1])
        clf = ManualLogisticRegression(epochs=10).fit(X_df, y_series)

        proba = clf.predict_proba(X_df)
        self.assertIsInstance(proba, np.ndarray)
        self.assertEqual(proba.shape, (4, 2))
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_10_probabilities_sum_to_one(self) -> None:
        """Requirement 10: Verify class probabilities for every sample sum to approximately 1."""
        X_df = pd.DataFrame({"f1": [-5.0, 0.0, 5.0, 10.0], "f2": [3.0, -2.0, 1.0, 8.0]})
        y_series = pd.Series([0, 0, 1, 1])
        clf = ManualLogisticRegression(epochs=25).fit(X_df, y_series)

        proba = clf.predict_proba(X_df)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(X_df)), atol=1e-12)

    def test_11_predictions_only_binary(self) -> None:
        """Requirement 11: Verify predictions contain only binary values {0, 1}."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 3.0, 4.0, 5.0]})
        y_series = pd.Series([0, 0, 1, 1])
        clf = ManualLogisticRegression(epochs=20).fit(X_df, y_series)

        preds = clf.predict(X_df)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 1)
        self.assertEqual(len(preds), 4)
        self.assertTrue(np.issubdtype(preds.dtype, np.integer))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_12_threshold_behavior(self) -> None:
        """Requirement 12: Verify configurable decision threshold alters binary classifications correctly."""
        # Construct a scenario where model produces known probabilities
        X_df = pd.DataFrame({"f1": [0.0], "f2": [0.0]})
        y_series = pd.Series([0, 1])  # Dummy fit data
        clf = ManualLogisticRegression(epochs=1)
        clf.fit(pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}), y_series)

        # Manually configure weights and bias so logit is 0.4054651081 (P(y=1) = 0.60)
        # z = ln(0.6 / 0.4) = ln(1.5) ~= 0.4054651081081644
        clf.weights_ = np.array([0.0, 0.0])
        clf.bias_ = math.log(1.5)

        proba = clf.predict_proba(X_df)
        p1 = proba[0, 1]
        self.assertAlmostEqual(p1, 0.6, places=6)

        # Test threshold = 0.5 -> p1 (0.6) >= 0.5 -> 1
        clf.threshold = 0.5
        self.assertEqual(clf.predict(X_df)[0], 1)

        # Test threshold = 0.7 -> p1 (0.6) < 0.7 -> 0
        clf.threshold = 0.7
        self.assertEqual(clf.predict(X_df)[0], 0)

        # Test boundary: threshold = 0.6 -> p1 >= 0.6 -> 1
        clf.threshold = 0.6
        self.assertEqual(clf.predict(X_df)[0], 1)

        # Test boundary: threshold = 0.6000001 -> p1 < threshold -> 0
        clf.threshold = 0.6000001
        self.assertEqual(clf.predict(X_df)[0], 0)

    # -------------------------------------------------------------------------
    # Input Validation Tests (Requirements 13 - 22)
    # -------------------------------------------------------------------------

    def test_13_rejects_empty_inputs(self) -> None:
        """Requirement 13: Verify classifier rejects empty feature DataFrames and target Series."""
        clf = ManualLogisticRegression()

        # Empty DataFrame
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame(), pd.Series([0, 1]))

        # DataFrame with 0 rows
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": []}), pd.Series([]))

        # Empty target Series
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0]}), pd.Series([]))

    def test_14_rejects_missing_feature_values(self) -> None:
        """Requirement 14: Verify classifier rejects NaN values in features (imputation must precede)."""
        clf = ManualLogisticRegression()
        X_nan = pd.DataFrame({"f1": [1.0, np.nan, 3.0], "f2": [4.0, 5.0, 6.0]})
        y = pd.Series([0, 1, 0])

        with self.assertRaises(ValueError) as ctx:
            clf.fit(X_nan, y)
        self.assertIn("missing (nan) values", str(ctx.exception).lower())

        # Also reject during predict()
        clf_fitted = ManualLogisticRegression(epochs=5).fit(
            pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}),
            pd.Series([0, 1]),
        )
        with self.assertRaises(ValueError):
            clf_fitted.predict(X_nan)

    def test_15_rejects_infinite_values(self) -> None:
        """Requirement 15: Verify classifier rejects infinite (+inf / -inf) feature and target values."""
        clf = ManualLogisticRegression()

        # Positive infinity in X
        X_pos_inf = pd.DataFrame({"f1": [1.0, np.inf], "f2": [3.0, 4.0]})
        with self.assertRaises(ValueError):
            clf.fit(X_pos_inf, pd.Series([0, 1]))

        # Negative infinity in X
        X_neg_inf = pd.DataFrame({"f1": [1.0, -np.inf], "f2": [3.0, 4.0]})
        with self.assertRaises(ValueError):
            clf.fit(X_neg_inf, pd.Series([0, 1]))

        # Infinity in y
        X_valid = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        with self.assertRaises(ValueError):
            clf.fit(X_valid, pd.Series([0.0, np.inf]))

    def test_16_rejects_non_numeric_features(self) -> None:
        """Requirement 16: Verify classifier rejects non-numeric (string / object) feature columns."""
        clf = ManualLogisticRegression()
        X_str = pd.DataFrame({"f1": [1.0, 2.0], "f2": ["potable", "acidic"]})
        y = pd.Series([0, 1])

        with self.assertRaises(TypeError) as ctx:
            clf.fit(X_str, y)
        self.assertIn("numeric dtype", str(ctx.exception))

    def test_17_rejects_invalid_target_labels(self) -> None:
        """Requirement 17: Verify classifier rejects non-binary or out-of-domain target labels."""
        clf = ManualLogisticRegression()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

        # Multiclass labels {0, 1, 2}
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([0, 1, 2]))

        # Non-{0, 1} binary labels {-1, 1}
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([-1, 1, 1]))

        # Continuous target values {0.2, 0.8, 0.5}
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([0.2, 0.8, 0.5]))

        # String target labels {"no", "yes"}
        with self.assertRaises(TypeError):
            clf.fit(X, pd.Series(["no", "yes", "no"]))

        # Missing values in target
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([0, np.nan, 1]))

    def test_18_rejects_mismatched_sample_counts(self) -> None:
        """Requirement 18: Verify classifier rejects mismatched sample counts between X and y."""
        clf = ManualLogisticRegression()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})  # 3 rows
        y_short = pd.Series([0, 1])  # 2 rows

        with self.assertRaises(ValueError) as ctx:
            clf.fit(X, y_short)
        self.assertIn("sample count mismatch", str(ctx.exception).lower())

    def test_19_rejects_mismatched_or_reordered_feature_columns(self) -> None:
        """Requirement 19: Verify prediction rejects missing, extra, or reordered feature columns."""
        X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y_train = pd.Series([0, 1])
        clf = ManualLogisticRegression(epochs=5).fit(X_train, y_train)

        # Missing column
        with self.assertRaises(ValueError):
            clf.predict(pd.DataFrame({"f1": [1.0, 2.0]}))

        # Extra column
        with self.assertRaises(ValueError):
            clf.predict(pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]}))

        # Reordered columns
        with self.assertRaises(ValueError) as ctx:
            clf.predict(pd.DataFrame({"f2": [3.0, 4.0], "f1": [1.0, 2.0]}))
        self.assertIn("feature mismatch", str(ctx.exception).lower())

    def test_20_rejects_prediction_before_fitting(self) -> None:
        """Requirement 20: Verify prediction and decision function methods reject calls before fitting."""
        clf = ManualLogisticRegression()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

        with self.assertRaises(RuntimeError):
            clf.decision_function(X)

        with self.assertRaises(RuntimeError):
            clf.predict_proba(X)

        with self.assertRaises(RuntimeError):
            clf.predict(X)

    def test_21_rejects_single_class_target(self) -> None:
        """Requirement 21: Verify classifier rejects training data containing only one target class."""
        clf = ManualLogisticRegression()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

        # All 0s
        with self.assertRaises(ValueError) as ctx1:
            clf.fit(X, pd.Series([0, 0, 0]))
        self.assertIn("only one class", str(ctx1.exception).lower())

        # All 1s
        with self.assertRaises(ValueError) as ctx2:
            clf.fit(X, pd.Series([1, 1, 1]))
        self.assertIn("only one class", str(ctx2.exception).lower())

    def test_22_rejects_incompatible_feature_dimensions(self) -> None:
        """Requirement 22: Verify classifier rejects non-DataFrame inputs and multi-dimensional targets."""
        clf = ManualLogisticRegression()

        # X is a raw NumPy array instead of a pandas DataFrame
        with self.assertRaises(TypeError):
            clf.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), pd.Series([0, 1]))  # type: ignore

        # y is 2D NumPy array with shape (2, 2)
        X_df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        with self.assertRaises(ValueError):
            clf.fit(X_df, np.array([[0, 1], [1, 0]]))

    # -------------------------------------------------------------------------
    # Data Integrity and Reproducibility Tests (Requirements 23 - 25)
    # -------------------------------------------------------------------------

    def test_23_inputs_remain_unmodified(self) -> None:
        """Requirement 23: Verify original input DataFrames and target Series are not modified in-place."""
        X_df = pd.DataFrame({
            "f1": [1.0, 2.5, 3.8],
            "f2": [4.2, 5.1, 6.9],
        }, index=[10, 20, 30])
        y_series = pd.Series([0, 1, 0], index=[10, 20, 30], name="custom_target")

        X_copy = X_df.copy(deep=True)
        y_copy = y_series.copy(deep=True)

        clf = ManualLogisticRegression(learning_rate=0.05, epochs=15)
        clf.fit(X_df, y_series)

        pd.testing.assert_frame_equal(X_df, X_copy)
        pd.testing.assert_series_equal(y_series, y_copy)

    def test_24_deterministic_reproducibility(self) -> None:
        """Requirement 24: Verify identical inputs and parameters produce perfectly identical results."""
        X_df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        y_series = pd.Series([0, 0, 1, 1, 1])

        clf1 = ManualLogisticRegression(learning_rate=0.1, epochs=50)
        clf1.fit(X_df, y_series)

        clf2 = ManualLogisticRegression(learning_rate=0.1, epochs=50)
        clf2.fit(X_df, y_series)

        np.testing.assert_array_equal(clf1.weights_, clf2.weights_)
        self.assertEqual(clf1.bias_, clf2.bias_)
        self.assertEqual(clf1.loss_history_, clf2.loss_history_)
        np.testing.assert_array_equal(clf1.predict_proba(X_df), clf2.predict_proba(X_df))
        np.testing.assert_array_equal(clf1.predict(X_df), clf2.predict(X_df))

    def test_25_preserves_feature_ordering(self) -> None:
        """Requirement 25: Verify fitted model preserves and enforces exact feature ordering."""
        features = ["Turbidity", "Conductivity", "Hardness"]
        X_df = pd.DataFrame({
            "Turbidity": [2.5, 3.1, 4.0],
            "Conductivity": [400.0, 420.0, 390.0],
            "Hardness": [180.0, 195.0, 210.0],
        })
        y_series = pd.Series([0, 1, 0])

        clf = ManualLogisticRegression(epochs=5).fit(X_df, y_series)
        self.assertEqual(clf.feature_names_in_, features)
        self.assertEqual(clf.n_features_in_, 3)

        # Predicting with original order succeeds
        preds = clf.predict(X_df)
        self.assertEqual(len(preds), 3)

        # Predicting with shuffled order raises ValueError
        X_shuffled = X_df[["Hardness", "Turbidity", "Conductivity"]]
        with self.assertRaises(ValueError):
            clf.predict(X_shuffled)

    # -------------------------------------------------------------------------
    # Additional Purity Pipeline Integration Tests (Requirements 26 - 30)
    # -------------------------------------------------------------------------

    def test_26_rejects_target_leakage_potability_column(self) -> None:
        """Requirement 26: Verify Potability column is strictly rejected if passed in feature matrix X."""
        X_leak = pd.DataFrame({
            "ph": [7.0, 6.5],
            TARGET_COLUMN: [0, 1],  # "Potability" must not be in X!
        })
        y = pd.Series([0, 1])

        clf = ManualLogisticRegression()
        with self.assertRaises(ValueError) as ctx:
            clf.fit(X_leak, y)
        self.assertIn("target leakage", str(ctx.exception).lower())

    def test_27_rejects_mismatched_series_index(self) -> None:
        """Requirement 27: Verify Series target with mismatched index to X is rejected."""
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=[0, 1])
        y_mismatched = pd.Series([0, 1], index=[10, 20])

        clf = ManualLogisticRegression()
        with self.assertRaises(ValueError) as ctx:
            clf.fit(X, y_mismatched)
        self.assertIn("index mismatch", str(ctx.exception).lower())

    def test_28_rejects_non_canonical_feature_ordering(self) -> None:
        """Requirement 28: Verify shuffled canonical water features in X are rejected during fit()."""
        # If all canonical features are present, they must be in canonical order
        shuffled_cols = list(reversed(CANONICAL_FEATURES))
        data = {col: [1.0, 2.0] for col in shuffled_cols}
        X_shuffled = pd.DataFrame(data)
        y = pd.Series([0, 1])

        clf = ManualLogisticRegression()
        with self.assertRaises(ValueError) as ctx:
            clf.fit(X_shuffled, y)
        self.assertIn("canonical ordering", str(ctx.exception).lower())

    def test_29_constructor_parameter_validation(self) -> None:
        """Requirement 29: Verify constructor parameter validation for learning rate, epochs, and threshold."""
        # Invalid learning_rate
        with self.assertRaises(ValueError):
            ManualLogisticRegression(learning_rate=-0.01)
        with self.assertRaises(ValueError):
            ManualLogisticRegression(learning_rate=0.0)
        with self.assertRaises(TypeError):
            ManualLogisticRegression(learning_rate=True)  # Boolean rejected
        with self.assertRaises(TypeError):
            ManualLogisticRegression(learning_rate="0.01")  # type: ignore

        # Invalid epochs
        with self.assertRaises(ValueError):
            ManualLogisticRegression(epochs=-10)
        with self.assertRaises(ValueError):
            ManualLogisticRegression(epochs=0)
        with self.assertRaises(TypeError):
            ManualLogisticRegression(epochs=10.5)  # Float rejected
        with self.assertRaises(TypeError):
            ManualLogisticRegression(epochs=True)  # Boolean rejected

        # Invalid threshold
        with self.assertRaises(ValueError):
            ManualLogisticRegression(threshold=-0.1)
        with self.assertRaises(ValueError):
            ManualLogisticRegression(threshold=1.5)
        with self.assertRaises(TypeError):
            ManualLogisticRegression(threshold=False)  # Boolean rejected
        with self.assertRaises(TypeError):
            ManualLogisticRegression(threshold="0.5")  # type: ignore

    def test_30_refit_resets_state_cleanly(self) -> None:
        """Requirement 30: Verify calling fit() multiple times re-initializes model state cleanly."""
        X1 = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y1 = pd.Series([0, 1])

        clf = ManualLogisticRegression(learning_rate=0.1, epochs=10)
        clf.fit(X1, y1)
        self.assertEqual(clf.n_features_in_, 2)
        self.assertEqual(len(clf.loss_history_), 10)

        # Refit on a 3-feature dataset
        X2 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        y2 = pd.Series([1, 0])
        clf.fit(X2, y2)

        self.assertEqual(clf.n_features_in_, 3)
        self.assertEqual(clf.feature_names_in_, ["a", "b", "c"])
        self.assertEqual(clf.weights_.shape, (3,))
        self.assertEqual(len(clf.loss_history_), 10)
        self.assertEqual(clf.n_iter_, 10)


if __name__ == "__main__":
    unittest.main()
