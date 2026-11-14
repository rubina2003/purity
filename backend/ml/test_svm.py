"""Unit Tests for Manual Support Vector Machine (SVM) Implementation.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.2 - Manual Support Vector Machine Classifier Implementation.

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
from backend.ml.svm import ManualSVM


class TestManualSVM(unittest.TestCase):
    """Automated unit test suite verifying kernel mathematics, SMO optimization,
    classification behavior, input validation, and data integrity for ManualSVM.
    """

    # -------------------------------------------------------------------------
    # Kernel Mathematics Tests (Requirements 1 - 7)
    # -------------------------------------------------------------------------

    def test_01_linear_kernel_dot_product(self) -> None:
        """Requirement 1: Verify linear kernel computes the exact dot product."""
        clf = ManualSVM(kernel="linear")
        x1 = np.array([[1.0, 2.0]])
        x2 = np.array([[3.0, 4.0]])

        # Dot product: 1.0 * 3.0 + 2.0 * 4.0 = 3.0 + 8.0 = 11.0
        K = clf._compute_kernel(x1, x2)
        self.assertEqual(K.shape, (1, 1))
        self.assertAlmostEqual(K[0, 0], 11.0, places=12)

        # Batch matrix dot product: (2, 2) @ (3, 2)^T -> (2, 3)
        X1 = np.array([[1.0, 0.0], [0.0, 2.0]])
        X2 = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        expected_K = np.array([[1.0, 2.0, 0.0], [2.0, 0.0, 6.0]])
        np.testing.assert_allclose(clf._compute_kernel(X1, X2), expected_K, atol=1e-12)

    def test_02_rbf_kernel_manual_calculation(self) -> None:
        """Requirement 2: Verify RBF kernel returns the exact manually calculated value."""
        # For x1 = [1.0, 2.0], x2 = [3.0, 4.0]:
        # ||x1 - x2||^2 = (1 - 3)^2 + (2 - 4)^2 = (-2)^2 + (-2)^2 = 4 + 4 = 8.0
        # With gamma = 0.5: K = exp(-0.5 * 8.0) = exp(-4.0) ~= 0.01831563888873418
        gamma = 0.5
        clf = ManualSVM(kernel="rbf", gamma=gamma)
        clf.gamma_ = gamma

        x1 = np.array([[1.0, 2.0]])
        x2 = np.array([[3.0, 4.0]])
        expected_val = math.exp(-0.5 * 8.0)

        K = clf._compute_kernel(x1, x2)
        self.assertEqual(K.shape, (1, 1))
        self.assertAlmostEqual(K[0, 0], expected_val, places=12)

    def test_03_kernel_matrix_is_symmetric(self) -> None:
        """Requirement 3: Verify kernel Gram matrix is symmetric for both Linear and RBF kernels."""
        X = np.array([[1.0, 2.0], [3.0, 5.0], [0.5, -1.2], [-2.0, 4.0]])

        # Linear kernel symmetry
        clf_lin = ManualSVM(kernel="linear")
        K_lin = clf_lin._compute_kernel(X, X)
        np.testing.assert_allclose(K_lin, K_lin.T, atol=1e-12)

        # RBF kernel symmetry
        clf_rbf = ManualSVM(kernel="rbf", gamma=0.3)
        clf_rbf.gamma_ = 0.3
        K_rbf = clf_rbf._compute_kernel(X, X)
        np.testing.assert_allclose(K_rbf, K_rbf.T, atol=1e-12)

    def test_04_rbf_diagonal_values_approximately_one(self) -> None:
        """Requirement 4: Verify RBF self-similarity diagonal values equal 1.0 (K(x_i, x_i) = 1)."""
        X = np.array([[-10.0, 20.0], [0.0, 0.0], [100.5, -50.2], [3.14, 2.71]])
        clf = ManualSVM(kernel="rbf", gamma=0.1)
        clf.gamma_ = 0.1

        K = clf._compute_kernel(X, X)
        diagonal = np.diag(K)
        np.testing.assert_allclose(diagonal, np.ones(len(X)), atol=1e-12)

    def test_05_squared_distances_non_negative(self) -> None:
        """Requirement 5: Verify squared Euclidean distances remain non-negative after numerical handling."""
        # Two identical points could produce -1e-16 due to floating-point rounding if unhandled
        X_identical = np.array([[1.23456789, 9.87654321], [1.23456789, 9.87654321]])
        clf = ManualSVM(kernel="rbf", gamma=1.0)
        clf.gamma_ = 1.0

        K = clf._compute_kernel(X_identical, X_identical)
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0))
        self.assertAlmostEqual(K[0, 1], 1.0, places=12)

    def test_06_gamma_numeric_configuration(self) -> None:
        """Requirement 6: Verify numeric gamma configuration is resolved and stored correctly."""
        clf = ManualSVM(kernel="rbf", gamma=0.25)
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y_series = pd.Series([0, 1, 0])

        clf.fit(X_df, y_series)
        self.assertAlmostEqual(clf.gamma_, 0.25, places=12)

    def test_07_gamma_scale_calculated_from_training_only(self) -> None:
        """Requirement 7: Verify 'scale' gamma is calculated strictly from training data variance."""
        # For X with n_features = 2:
        # X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        # Var(X) across all elements is 5.25
        # Expected gamma = 1.0 / (2 * 5.25) = 1.0 / 10.5 ~= 0.09523809523809523
        X_train = pd.DataFrame({"f1": [1.0, 3.0, 5.0, 7.0], "f2": [2.0, 4.0, 6.0, 8.0]})
        y_train = pd.Series([0, 0, 1, 1])

        clf = ManualSVM(kernel="rbf", gamma="scale", max_iter=10)
        clf.fit(X_train, y_train)

        expected_var = float(np.var(X_train.to_numpy()))
        expected_gamma = 1.0 / (2.0 * expected_var)

        self.assertAlmostEqual(clf.gamma_, expected_gamma, places=10)

        # Test inference with wildly different data scale does NOT change fitted gamma_
        X_test = pd.DataFrame({"f1": [1000.0, 2000.0], "f2": [3000.0, 4000.0]})
        clf.predict(X_test)
        self.assertAlmostEqual(clf.gamma_, expected_gamma, places=10)

    # -------------------------------------------------------------------------
    # SMO Optimization Tests (Requirements 8 - 15)
    # -------------------------------------------------------------------------

    def test_08_alphas_within_box_bounds(self) -> None:
        """Requirement 8: Verify all learned Lagrange multipliers reside within box bounds [0, C]."""
        C_val = 2.5
        clf = ManualSVM(C=C_val, kernel="linear", max_iter=200)
        X_df = pd.DataFrame({
            "x1": [1.0, 2.0, 2.0, 5.0, 6.0, 5.0],
            "x2": [1.0, 1.0, 2.0, 5.0, 5.0, 6.0],
        })
        y_series = pd.Series([0, 0, 0, 1, 1, 1])

        clf.fit(X_df, y_series)

        # Every alpha must satisfy 0 <= alpha <= C (within numerical epsilon)
        self.assertTrue(np.all(clf.alphas_ >= -clf.alpha_eps))
        self.assertTrue(np.all(clf.alphas_ <= C_val + clf.alpha_eps))

    def test_09_dual_equality_constraint(self) -> None:
        """Requirement 9: Verify dual equality constraint sum_{i=1}^m alpha_i * y_i = 0 is satisfied."""
        clf = ManualSVM(C=1.0, kernel="linear", max_iter=200)
        X_df = pd.DataFrame({
            "x1": [1.0, 2.0, 1.5, 6.0, 7.0, 6.5],
            "x2": [1.0, 1.5, 2.0, 6.0, 6.5, 7.0],
        })
        y_series = pd.Series([0, 0, 0, 1, 1, 1])

        clf.fit(X_df, y_series)

        y_svm = np.where(y_series.to_numpy() == 1, 1.0, -1.0)
        dual_equality_sum = float(np.sum(clf.alphas_ * y_svm))

        self.assertAlmostEqual(dual_equality_sum, 0.0, places=4)

    def test_10_bias_update_rules(self) -> None:
        """Requirement 10: Verify bias update strictly follows the standard three-branch SMO rule."""
        clf = ManualSVM(C=1.0)
        b_old = 0.5

        # Case 1: First alpha is non-bound (0 < a_i_new < C) -> b_new must equal b1
        b_res1 = clf._compute_bias_update(
            b_old=b_old, i=0, j=1,
            alpha_i_old=0.0, alpha_i_new=0.4,
            alpha_j_old=0.0, alpha_j_new=0.0,
            y_i=1.0, y_j=-1.0,
            E_i=-1.0, E_j=1.0,
            k_ii=1.0, k_jj=1.0, k_ij=0.5
        )
        # Expected b1 = b_old - E_i - y_i * (a_i_new - a_i_old) * k_ii - y_j * (a_j_new - a_j_old) * k_ij
        # b1 = 0.5 - (-1.0) - 1.0 * (0.4 - 0.0) * 1.0 - (-1.0) * (0.0) = 1.5 - 0.4 = 1.1
        self.assertAlmostEqual(b_res1, 1.1, places=10)

        # Case 2: Second alpha is non-bound (0 < a_j_new < C) while first is at bound (0) -> b_new = b2
        b_res2 = clf._compute_bias_update(
            b_old=b_old, i=0, j=1,
            alpha_i_old=0.0, alpha_i_new=0.0,
            alpha_j_old=0.0, alpha_j_new=0.6,
            y_i=1.0, y_j=-1.0,
            E_i=-1.0, E_j=1.0,
            k_ii=1.0, k_jj=1.0, k_ij=0.5
        )
        # b2 = b_old - E_j - y_i * (0) - y_j * (0.6 - 0) * k_jj = 0.5 - 1.0 - (-1.0) * 0.6 * 1.0 = -0.5 + 0.6 = 0.1
        self.assertAlmostEqual(b_res2, 0.1, places=10)

        # Case 3: Both alphas at bound (e.g. 0 and C) -> b_new = (b1 + b2) / 2
        b_res3 = clf._compute_bias_update(
            b_old=b_old, i=0, j=1,
            alpha_i_old=0.0, alpha_i_new=0.0,
            alpha_j_old=0.0, alpha_j_new=1.0,  # Bound at C=1.0
            y_i=1.0, y_j=-1.0,
            E_i=-1.0, E_j=1.0,
            k_ii=1.0, k_jj=1.0, k_ij=0.5
        )
        # b1 = 0.5 - (-1.0) - 0 - (-1.0) * 1.0 * 0.5 = 1.5 + 0.5 = 2.0
        # b2 = 0.5 - 1.0 - 0 - (-1.0) * 1.0 * 1.0 = -0.5 + 1.0 = 0.5
        # avg = (2.0 + 0.5) / 2 = 1.25
        self.assertAlmostEqual(b_res3, 1.25, places=10)

    def test_11_error_cache_consistency(self) -> None:
        """Requirement 11: Verify error cache updates remain consistent with the decision function."""
        clf = ManualSVM(C=1.0, kernel="rbf", max_iter=50)
        X_df = pd.DataFrame({
            "f1": [1.0, 1.5, 2.0, 5.0, 5.5, 6.0],
            "f2": [1.0, 1.2, 1.8, 5.0, 5.2, 5.8],
        })
        y_series = pd.Series([0, 0, 0, 1, 1, 1])

        clf.fit(X_df, y_series)

        # Compute decision scores from scratch using decision_function()
        scores = clf.decision_function(X_df)
        y_svm = np.where(y_series.to_numpy() == 1, 1.0, -1.0)
        expected_errors = scores - y_svm

        # Ensure active support vectors with 0 < alpha < C have f(x) ~= y (error ~= 0)
        for idx in range(len(X_df)):
            if clf.alpha_eps < clf.alphas_[idx] < (clf.C - clf.alpha_eps):
                self.assertAlmostEqual(expected_errors[idx], 0.0, places=2)

    def test_12_stopping_conditions(self) -> None:
        """Requirement 12: Verify optimization terminates according to max_passes stopping condition."""
        clf = ManualSVM(kernel="linear", max_passes=3, max_iter=200)
        X_df = pd.DataFrame({"x1": [1.0, 2.0, 5.0, 6.0], "x2": [1.0, 1.0, 5.0, 5.0]})
        y_series = pd.Series([0, 0, 1, 1])

        clf.fit(X_df, y_series)

        self.assertLessEqual(clf.n_iter_, 200)
        self.assertGreaterEqual(clf.n_passes_, 3)
        self.assertTrue(clf.converged_)

    def test_13_convergence_status_distinction(self) -> None:
        """Requirement 13: Verify convergence status distinguishes convergence from iteration-limit truncation."""
        X_df = pd.DataFrame({
            "x1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            "x2": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        })
        y_series = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])  # Alternating, non-trivial

        # Artificially constrain max_iter to 1 pass so it cannot converge
        clf_truncated = ManualSVM(kernel="linear", max_iter=1, max_passes=10)
        clf_truncated.fit(X_df, y_series)
        self.assertFalse(clf_truncated.converged_)
        self.assertEqual(clf_truncated.n_iter_, 1)

        # Standard separable dataset allowing full convergence
        clf_converged = ManualSVM(kernel="linear", max_iter=200, max_passes=5)
        X_sep = pd.DataFrame({"x1": [1.0, 2.0, 8.0, 9.0], "x2": [1.0, 2.0, 8.0, 9.0]})
        y_sep = pd.Series([0, 0, 1, 1])
        clf_converged.fit(X_sep, y_sep)
        self.assertTrue(clf_converged.converged_)

    def test_14_deterministic_reproducibility(self) -> None:
        """Requirement 14: Verify repeated fitting with the same data and parameters is strictly deterministic."""
        X_df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            "f2": [2.0, 1.0, 3.0, 6.0, 5.0, 7.0],
        })
        y_series = pd.Series([0, 0, 0, 1, 1, 1])

        clf1 = ManualSVM(C=1.0, kernel="rbf", gamma=0.5, max_iter=100)
        clf1.fit(X_df, y_series)

        clf2 = ManualSVM(C=1.0, kernel="rbf", gamma=0.5, max_iter=100)
        clf2.fit(X_df, y_series)

        np.testing.assert_array_equal(clf1.alphas_, clf2.alphas_)
        self.assertEqual(clf1.bias_, clf2.bias_)
        self.assertEqual(clf1.n_updates_, clf2.n_updates_)
        self.assertEqual(clf1.n_iter_, clf2.n_iter_)
        np.testing.assert_array_equal(clf1.decision_function(X_df), clf2.decision_function(X_df))
        np.testing.assert_array_equal(clf1.predict(X_df), clf2.predict(X_df))

    def test_15_degenerate_duplicate_samples_handled(self) -> None:
        """Requirement 15: Verify degenerate or duplicate training samples are handled safely without crash."""
        # Contains identical duplicate rows -> eta = 2*K_ij - K_ii - K_jj = 0
        X_dup = pd.DataFrame({
            "f1": [1.0, 1.0, 5.0, 5.0],
            "f2": [2.0, 2.0, 6.0, 6.0],
        })
        y_dup = pd.Series([0, 0, 1, 1])

        clf = ManualSVM(kernel="linear", max_iter=50)
        # Must not raise ZeroDivisionError or FloatingPointError
        clf.fit(X_dup, y_dup)
        preds = clf.predict(X_dup)
        self.assertEqual(len(preds), 4)

    # -------------------------------------------------------------------------
    # Classification Tests (Requirements 16 - 21)
    # -------------------------------------------------------------------------

    def test_16_linear_svm_classifies_linearly_separable(self) -> None:
        """Requirement 16: Verify Linear SVM achieves 100% classification on linearly separable data."""
        X_df = pd.DataFrame({
            "x1": [1.0, 2.0, 1.5, 6.0, 7.0, 6.5],
            "x2": [1.0, 1.5, 2.0, 6.0, 6.5, 7.0],
        })
        y_series = pd.Series([0, 0, 0, 1, 1, 1])

        clf = ManualSVM(kernel="linear", C=1.0, max_iter=200)
        clf.fit(X_df, y_series)

        preds = clf.predict(X_df)
        np.testing.assert_array_equal(preds, y_series.to_numpy())

    def test_17_rbf_svm_classifies_nonlinear_fixture(self) -> None:
        """Requirement 17: Verify RBF SVM achieves 100% classification on a nonlinear concentric ring fixture."""
        # Non-linear concentric fixture:
        # Center points: (0, 0), (+-0.5, 0), (0, +-0.5) -> Label 1
        # Outer ring points: (+-2, 0), (0, +-2), (+-1.5, +-1.5) -> Label 0
        X_df = pd.DataFrame({
            "x1": [0.0, 0.5, -0.5, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 1.5, -1.5, 1.5, -1.5],
            "x2": [0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 2.0, -2.0, 1.5, 1.5, -1.5, -1.5],
        })
        y_series = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        clf = ManualSVM(kernel="rbf", C=5.0, gamma="scale", max_iter=250)
        clf.fit(X_df, y_series)

        preds = clf.predict(X_df)
        np.testing.assert_array_equal(preds, y_series.to_numpy())

    def test_18_decision_function_shape_and_type(self) -> None:
        """Requirement 18: Verify decision_function() returns a 1D NumPy array with correct sample shape."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [5.0, 6.0, 7.0, 8.0]})
        y_series = pd.Series([0, 0, 1, 1])

        clf = ManualSVM(kernel="linear", max_iter=50).fit(X_df, y_series)
        scores = clf.decision_function(X_df)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.ndim, 1)
        self.assertEqual(len(scores), 4)

    def test_19_decision_score_sign_maps_to_classes(self) -> None:
        """Requirement 19: Verify positive decision scores map to class 1, negative to class 0."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 6.0, 7.0], "f2": [1.0, 2.0, 6.0, 7.0]})
        y_series = pd.Series([0, 0, 1, 1])

        clf = ManualSVM(kernel="linear", max_iter=100).fit(X_df, y_series)
        scores = clf.decision_function(X_df)
        preds = clf.predict(X_df)

        for s, p in zip(scores, preds):
            if s >= 0.0:
                self.assertEqual(p, 1)
            else:
                self.assertEqual(p, 0)

    def test_20_predict_returns_binary_only(self) -> None:
        """Requirement 20: Verify predict() returns strictly binary integers {0, 1}."""
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [2.0, 1.0, 4.0, 3.0]})
        y_series = pd.Series([0, 1, 0, 1])

        clf = ManualSVM(max_iter=50).fit(X_df, y_series)
        preds = clf.predict(X_df)

        self.assertTrue(np.issubdtype(preds.dtype, np.integer))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_21_support_vectors_and_dual_coef_aligned(self) -> None:
        """Requirement 21: Verify support vectors, support indices, and dual coefficients remain aligned."""
        X_df = pd.DataFrame({"x1": [1.0, 2.0, 5.0, 6.0], "x2": [1.0, 1.5, 5.0, 5.5]})
        y_series = pd.Series([0, 0, 1, 1])

        clf = ManualSVM(kernel="linear", max_iter=100).fit(X_df, y_series)

        self.assertEqual(clf.n_support_, len(clf.support_))
        self.assertEqual(len(clf.support_vectors_), clf.n_support_)
        self.assertEqual(len(clf.dual_coef_), clf.n_support_)

        y_svm = np.where(y_series.to_numpy() == 1, 1.0, -1.0)
        for k, sv_idx in enumerate(clf.support_):
            expected_dual = clf.alphas_[sv_idx] * y_svm[sv_idx]
            self.assertAlmostEqual(clf.dual_coef_[k], expected_dual, places=12)
            np.testing.assert_array_equal(clf.support_vectors_[k], X_df.iloc[sv_idx].to_numpy())

    # -------------------------------------------------------------------------
    # Validation and Integrity Tests (Requirements 22 - 34)
    # -------------------------------------------------------------------------

    def test_22_rejects_empty_inputs(self) -> None:
        """Requirement 22: Verify classifier rejects empty feature DataFrames and target Series."""
        clf = ManualSVM()

        # Empty DataFrame
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame(), pd.Series([0, 1]))

        # DataFrame with 0 rows
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": []}), pd.Series([]))

        # Empty target Series
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0]}), pd.Series([]))

    def test_23_rejects_nan_and_infinite_values(self) -> None:
        """Requirement 23: Verify classifier rejects NaN and infinite values in features and targets."""
        clf = ManualSVM()

        # NaN in X
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0, np.nan], "f2": [2.0, 3.0]}), pd.Series([0, 1]))

        # +inf in X
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0, np.inf], "f2": [2.0, 3.0]}), pd.Series([0, 1]))

        # -inf in X
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0, -np.inf], "f2": [2.0, 3.0]}), pd.Series([0, 1]))

        # NaN in target
        with self.assertRaises(ValueError):
            clf.fit(pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}), pd.Series([0, np.nan]))

    def test_24_rejects_non_numeric_features(self) -> None:
        """Requirement 24: Verify classifier rejects non-numeric (string / object) feature columns."""
        clf = ManualSVM()
        X_str = pd.DataFrame({"f1": [1.0, 2.0], "f2": ["potable", "non-potable"]})
        y = pd.Series([0, 1])

        with self.assertRaises(TypeError) as ctx:
            clf.fit(X_str, y)
        self.assertIn("numeric dtype", str(ctx.exception))

    def test_25_rejects_invalid_target_labels(self) -> None:
        """Requirement 25: Verify classifier rejects non-binary, continuous, string, and out-of-domain targets."""
        clf = ManualSVM()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

        # Multiclass labels {0, 1, 2}
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([0, 1, 2]))

        # Non-{0, 1} binary labels {-1, 1} (must be supplied as {0, 1})
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([-1, 1, 1]))

        # Continuous target values
        with self.assertRaises(ValueError):
            clf.fit(X, pd.Series([0.2, 0.8, 0.5]))

        # String target labels
        with self.assertRaises(TypeError):
            clf.fit(X, pd.Series(["no", "yes", "no"]))

    def test_26_rejects_single_class_target(self) -> None:
        """Requirement 26: Verify classifier rejects training data containing only one class."""
        clf = ManualSVM()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})

        # All 0s
        with self.assertRaises(ValueError) as ctx1:
            clf.fit(X, pd.Series([0, 0, 0]))
        self.assertIn("only one class", str(ctx1.exception).lower())

        # All 1s
        with self.assertRaises(ValueError) as ctx2:
            clf.fit(X, pd.Series([1, 1, 1]))
        self.assertIn("only one class", str(ctx2.exception).lower())

    def test_27_rejects_mismatched_sample_counts(self) -> None:
        """Requirement 27: Verify classifier rejects mismatched sample counts between X and y."""
        clf = ManualSVM()
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})  # 3 rows
        y = pd.Series([0, 1])  # 2 rows

        with self.assertRaises(ValueError) as ctx:
            clf.fit(X, y)
        self.assertIn("sample count mismatch", str(ctx.exception).lower())

    def test_28_rejects_mismatched_or_reordered_inference_columns(self) -> None:
        """Requirement 28: Verify prediction rejects missing, extra, or reordered feature columns."""
        X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        y_train = pd.Series([0, 1])
        clf = ManualSVM(max_iter=5).fit(X_train, y_train)

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

    def test_29_rejects_prediction_before_fitting(self) -> None:
        """Requirement 29: Verify prediction and decision function methods reject calls prior to fit()."""
        clf = ManualSVM()
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

        with self.assertRaises(RuntimeError):
            clf.decision_function(X)

        with self.assertRaises(RuntimeError):
            clf.predict(X)

    def test_30_rejects_potability_input_feature(self) -> None:
        """Requirement 30: Verify target column Potability is strictly rejected if supplied in X (leakage prevention)."""
        X_leak = pd.DataFrame({
            "ph": [7.0, 6.5],
            TARGET_COLUMN: [0, 1],  # Leakage
        })
        y = pd.Series([0, 1])

        clf = ManualSVM()
        with self.assertRaises(ValueError) as ctx:
            clf.fit(X_leak, y)
        self.assertIn("target leakage", str(ctx.exception).lower())

    def test_31_rejects_target_index_misalignment(self) -> None:
        """Requirement 31: Verify target Series with mismatched index to X is rejected."""
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=[0, 1])
        y_mismatch = pd.Series([0, 1], index=[10, 20])

        clf = ManualSVM()
        with self.assertRaises(ValueError) as ctx:
            clf.fit(X, y_mismatch)
        self.assertIn("index mismatch", str(ctx.exception).lower())

    def test_32_rejects_invalid_constructor_parameters(self) -> None:
        """Requirement 32: Verify constructor parameter validation for C, kernel, gamma, tol, passes, iter, and eps."""
        # Invalid C
        with self.assertRaises(ValueError):
            ManualSVM(C=-1.0)
        with self.assertRaises(ValueError):
            ManualSVM(C=0.0)
        with self.assertRaises(TypeError):
            ManualSVM(C=True)

        # Invalid kernel
        with self.assertRaises(ValueError):
            ManualSVM(kernel="sigmoid")
        with self.assertRaises(ValueError):
            ManualSVM(kernel="polynomial")
        with self.assertRaises(TypeError):
            ManualSVM(kernel=123)  # type: ignore

        # Invalid gamma
        with self.assertRaises(ValueError):
            ManualSVM(gamma="auto")  # Only 'scale' is valid string
        with self.assertRaises(ValueError):
            ManualSVM(gamma=-0.5)
        with self.assertRaises(TypeError):
            ManualSVM(gamma=False)

        # Invalid max_passes
        with self.assertRaises(ValueError):
            ManualSVM(max_passes=0)
        with self.assertRaises(TypeError):
            ManualSVM(max_passes=1.5)  # type: ignore

        # Invalid max_iter
        with self.assertRaises(ValueError):
            ManualSVM(max_iter=-5)
        with self.assertRaises(TypeError):
            ManualSVM(max_iter=True)

    def test_33_original_inputs_unmodified(self) -> None:
        """Requirement 33: Verify original input DataFrames and target Series remain unmodified in-place."""
        X_df = pd.DataFrame({
            "f1": [1.0, 2.5, 3.8],
            "f2": [4.2, 5.1, 6.9],
        }, index=[10, 20, 30])
        y_series = pd.Series([0, 1, 0], index=[10, 20, 30], name="custom_target")

        X_copy = X_df.copy(deep=True)
        y_copy = y_series.copy(deep=True)

        clf = ManualSVM(kernel="linear", max_iter=10)
        clf.fit(X_df, y_series)

        pd.testing.assert_frame_equal(X_df, X_copy)
        pd.testing.assert_series_equal(y_series, y_copy)

    def test_34_no_probability_estimates_exposed(self) -> None:
        """Requirement 34: Verify SVM does not expose predict_proba or calibrated probability confidence."""
        clf = ManualSVM(kernel="linear", max_iter=20)
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 5.0, 6.0], "f2": [1.0, 2.0, 5.0, 6.0]})
        y_series = pd.Series([0, 0, 1, 1])

        clf.fit(X_df, y_series)

        # Ensure predict_proba is not exposed
        self.assertFalse(hasattr(clf, "predict_proba"))

        # Ensure decision_function outputs uncalibrated margin scores that can exceed [0, 1]
        scores = clf.decision_function(X_df)
        self.assertIsInstance(scores, np.ndarray)
        # Margin scores on separable data typically diverge outside [-1, 1]
        # Verify scores are not bounded between 0 and 1
        has_outside_unit_interval = np.any((scores < 0.0) | (scores > 1.0))
        self.assertTrue(has_outside_unit_interval)

    def test_35_contradictory_duplicate_samples(self) -> None:
        """Requirement 35: Verify contradictory duplicate samples with zero eta satisfy KKT and terminate safely."""
        X_dup = pd.DataFrame({"f1": [1.0, 1.0], "f2": [1.0, 1.0]})
        y_dup = pd.Series([1, 0])
        C_val = 1.0

        clf = ManualSVM(C=C_val, kernel="linear", tol=1e-3, max_passes=5, max_iter=100)

        # 1. Verify kernel pair has approximately zero eta
        X_mat = X_dup.to_numpy(dtype=float)
        K = clf._compute_kernel(X_mat, X_mat)
        eta = 2.0 * K[0, 1] - K[0, 0] - K[1, 1]
        self.assertAlmostEqual(eta, 0.0, places=10)

        # Fit the model
        clf.fit(X_dup, y_dup)

        # 2 & 6. Verify optimizer terminates safely with finite parameters
        self.assertTrue(np.all(np.isfinite(clf.alphas_)))
        self.assertTrue(np.isfinite(clf.bias_))
        self.assertTrue(clf.converged_)

        # 3. Verify alphas remain within [0, C]
        self.assertTrue(np.all(clf.alphas_ >= -clf.alpha_eps))
        self.assertTrue(np.all(clf.alphas_ <= C_val + clf.alpha_eps))
        # For contradictory identical samples, optimal soft-margin alphas are at upper bound C
        self.assertAlmostEqual(clf.alphas_[0], C_val, places=4)
        self.assertAlmostEqual(clf.alphas_[1], C_val, places=4)

        # 4. Verify dual equality constraint: alpha_0 * y_0 + alpha_1 * y_1 = 0
        y_svm = np.array([1.0, -1.0])
        dual_equality = float(np.sum(clf.alphas_ * y_svm))
        self.assertAlmostEqual(dual_equality, 0.0, places=10)

        # 5. Verify final solution satisfies applicable KKT conditions within tolerance
        E = clf._compute_errors(y_svm, K)
        kkt_satisfied = clf._check_kkt_convergence(y_svm, E)
        self.assertTrue(kkt_satisfied)

    def test_36_zero_margin_boundary_decision_rule(self) -> None:
        """Requirement 36: Verify zero-margin boundary behavior (f(x) >= 0 -> 1, f(x) < 0 -> 0)."""
        # Symmetric linear fixture where the optimal hyper-plane passes through x = 0 with bias b = 0
        X = pd.DataFrame({"f1": [-1.0, 1.0]})
        y = pd.Series([0, 1])

        clf = ManualSVM(kernel="linear", C=1.0).fit(X, y)

        # Evaluate at negative, exact zero, and positive decision scores
        X_test = pd.DataFrame({"f1": [-0.5, 0.0, 0.5]})
        scores = clf.decision_function(X_test)
        preds = clf.predict(X_test)

        # Verify exact decision scores
        self.assertAlmostEqual(scores[0], -0.5, places=7)
        self.assertAlmostEqual(scores[1], 0.0, places=7)
        self.assertAlmostEqual(scores[2], 0.5, places=7)

        # Verify predictions strictly follow the decision rule:
        # f(x) < 0.0  -> class 0
        # f(x) == 0.0 -> class 1 (zero margin boundary is assigned to class 1)
        # f(x) > 0.0  -> class 1
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)
        self.assertEqual(preds[2], 1)


if __name__ == "__main__":
    unittest.main()

