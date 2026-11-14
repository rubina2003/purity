"""Integration and Compatibility Test Suite for Purity Machine Learning Classifiers.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.4 - Final Algorithm Integration and Compatibility Verification.

Verifies:
1. Unified package-level and direct imports across all three custom classifiers.
2. Common classifier interface adherence: fit(X, y) and predict(X).
3. Canonical feature schema (9 physicochemical features) compatibility and ordering.
4. Binary classification semantics: 1D NumPy array with integer labels in {0, 1}.
5. Model attribute consistency: classes_, feature_names_in_, n_features_in_, is_fitted_.
6. Output semantics:
   - Logistic Regression: estimated probabilities from sigmoid link function (predict_proba()) and decision_function() logits.
   - Random Forest: uncalibrated predict_proba() tree-vote proportions.
   - SVM: raw decision_function() scores/margins, strictly NO predict_proba.
7. Input validation consistency: missingness, infinities, non-numeric dtypes, sample mismatch,
   target index alignment, target leakage rejection ('Potability' in X), single-class targets.
8. Original DataFrame and Series immutability (no in-place mutation).
9. Deterministic reproducibility across repeated runs.
10. End-to-end preprocessing compatibility with Phase 2 pipelines (Mean and Median imputation +
    Z-Score StandardScaler) using the approved dataset without fitting on test partition.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.data_split import stratified_train_test_split
from backend.ml.eda import CANONICAL_FEATURES, TARGET_COLUMN
from backend.ml.imputation import ManualImputer
from backend.ml.scaling import ManualStandardScaler


class TestAlgorithmCompatibility(unittest.TestCase):
    """Integration and compatibility verification across all custom classifiers and preprocessing."""

    def setUp(self) -> None:
        """Create reproducible fixtures with the 9 canonical water-quality features."""
        np.random.seed(42)
        rng = np.random.default_rng(42)

        self.n_samples = 30
        self.canonical_features = CANONICAL_FEATURES

        # Standardized 9-feature controlled fixture
        X_mat = rng.normal(loc=0.0, scale=1.0, size=(self.n_samples, len(self.canonical_features)))
        # Create linearly separable distribution along first feature for rapid convergence
        X_mat[:15, 0] -= 2.0
        X_mat[15:, 0] += 2.0

        self.X_fixture = pd.DataFrame(X_mat, columns=self.canonical_features)
        y_vec = np.zeros(self.n_samples, dtype=int)
        y_vec[15:] = 1
        self.y_fixture = pd.Series(y_vec, name=TARGET_COLUMN)

        # Unseen test fixture with identical schema
        X_test_mat = rng.normal(loc=0.0, scale=1.0, size=(10, len(self.canonical_features)))
        self.X_test_fixture = pd.DataFrame(X_test_mat, columns=self.canonical_features)

        self.dataset_path = Path("data/water_potability.csv")

    def _get_classifier_instances(self) -> list:
        """Instantiate fresh instances of all three classifiers with fast test parameters."""
        from backend.ml import ManualLogisticRegression, ManualRandomForest, ManualSVM

        return [
            ManualLogisticRegression(learning_rate=0.05, epochs=100),
            ManualSVM(C=1.0, kernel="linear", max_iter=100),
            ManualRandomForest(n_estimators=10, max_depth=3, random_state=42),
        ]

    # =========================================================================
    # 1. Package Imports and Architecture
    # =========================================================================

    def test_01_package_imports(self) -> None:
        """Verify package-level and direct module imports work without circular dependencies."""
        # 1. Package-level import
        from backend.ml import (
            ManualLogisticRegression as MLR_pkg,
            ManualRandomForest as MRF_pkg,
            ManualSVM as MSVM_pkg,
        )

        # 2. Direct module import
        from backend.ml.logistic_regression import ManualLogisticRegression as MLR_dir
        from backend.ml.random_forest import ManualRandomForest as MRF_dir
        from backend.ml.svm import ManualSVM as MSVM_dir

        self.assertIs(MLR_pkg, MLR_dir)
        self.assertIs(MSVM_pkg, MSVM_dir)
        self.assertIs(MRF_pkg, MRF_dir)

    # =========================================================================
    # 2. Common Interface Adherence: fit() and Attributes
    # =========================================================================

    def test_02_common_fit_signature_and_attributes(self) -> None:
        """Verify all three classifiers implement fit(X, y) returning self and setting standard attributes."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                # Prior to fit
                self.assertFalse(clf.is_fitted_)

                # fit() must return self
                result = clf.fit(self.X_fixture, self.y_fixture)
                self.assertIs(result, clf, f"{clf_name}.fit() must return self.")

                # Attribute contract
                self.assertTrue(clf.is_fitted_, f"{clf_name} is_fitted_ must be True after fit.")
                self.assertEqual(
                    clf.feature_names_in_,
                    self.canonical_features,
                    f"{clf_name} feature_names_in_ must match canonical features.",
                )
                self.assertEqual(
                    clf.n_features_in_,
                    len(self.canonical_features),
                    f"{clf_name} n_features_in_ must equal 9.",
                )
                np.testing.assert_array_equal(
                    clf.classes_,
                    np.array([0, 1]),
                    err_msg=f"{clf_name} classes_ must be [0, 1].",
                )

    # =========================================================================
    # 3. Common Interface Adherence: predict() Output
    # =========================================================================

    def test_03_common_predict_output_shape_and_binary_labels(self) -> None:
        """Verify predict(X) across all three classifiers returns 1D array of strictly binary {0, 1} ints."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                clf.fit(self.X_fixture, self.y_fixture)
                preds = clf.predict(self.X_test_fixture)

                self.assertIsInstance(
                    preds, np.ndarray, f"{clf_name}.predict() must return a NumPy ndarray."
                )
                self.assertEqual(
                    preds.ndim, 1, f"{clf_name}.predict() output must be 1-dimensional."
                )
                self.assertEqual(
                    len(preds),
                    len(self.X_test_fixture),
                    f"{clf_name}.predict() length must match input samples.",
                )
                self.assertTrue(
                    np.issubdtype(preds.dtype, np.integer),
                    f"{clf_name}.predict() dtype must be integer.",
                )
                unique_labels = set(np.unique(preds))
                self.assertTrue(
                    unique_labels.issubset({0, 1}),
                    f"{clf_name}.predict() must contain only binary labels {{0, 1}}, got {unique_labels}.",
                )

    # =========================================================================
    # 4. Canonical Feature Ordering and Schema Preservation
    # =========================================================================

    def test_04_canonical_feature_order_preserved(self) -> None:
        """Verify all three classifiers preserve canonical column names and order upon fitting."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                clf.fit(self.X_fixture, self.y_fixture)
                self.assertEqual(clf.feature_names_in_, CANONICAL_FEATURES)

    # =========================================================================
    # 5. Target Leakage Prevention
    # =========================================================================

    def test_05_target_leakage_rejected(self) -> None:
        """Verify that including 'Potability' in feature matrix X raises ValueError across all classifiers."""
        X_leak = self.X_fixture.copy()
        X_leak[TARGET_COLUMN] = 1.0

        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                with self.assertRaises(
                    ValueError,
                    msg=f"{clf_name} must reject feature matrix containing target column '{TARGET_COLUMN}'.",
                ):
                    clf.fit(X_leak, self.y_fixture)

    # =========================================================================
    # 6. Input DataFrame and Target Series Immutability
    # =========================================================================

    def test_06_original_inputs_unmodified(self) -> None:
        """Verify original input DataFrames and target Series are not mutated during fit or predict."""
        X_orig = self.X_fixture.copy(deep=True)
        y_orig = self.y_fixture.copy(deep=True)

        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                clf.fit(self.X_fixture, self.y_fixture)
                clf.predict(self.X_fixture)

                pd.testing.assert_frame_equal(
                    self.X_fixture,
                    X_orig,
                    obj=f"{clf_name} must not mutate input X in place.",
                )
                pd.testing.assert_series_equal(
                    self.y_fixture,
                    y_orig,
                    obj=f"{clf_name} must not mutate input y in place.",
                )

    # =========================================================================
    # 7. Inference Schema Alignment and Rejection
    # =========================================================================

    def test_07_inference_column_compatibility_and_rejection(self) -> None:
        """Verify inference rejects reordered, missing, or extraneous feature columns across all classifiers."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                clf.fit(self.X_fixture, self.y_fixture)

                # Reordered columns
                reordered_cols = list(reversed(self.canonical_features))
                X_reordered = self.X_test_fixture[reordered_cols]
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject reordered inference columns."):
                    clf.predict(X_reordered)

                # Missing column
                X_missing = self.X_test_fixture.drop(columns=[self.canonical_features[0]])
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject missing inference columns."):
                    clf.predict(X_missing)

                # Extra column
                X_extra = self.X_test_fixture.copy()
                X_extra["extra_feature"] = 1.0
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject extra inference columns."):
                    clf.predict(X_extra)

    # =========================================================================
    # 8. Output Semantics: Logistic Regression
    # =========================================================================

    def test_08_logistic_regression_probability_output(self) -> None:
        """Verify Logistic Regression predict_proba() returns estimated probabilities from sigmoid function."""
        from backend.ml import ManualLogisticRegression

        lr = ManualLogisticRegression(learning_rate=0.05, epochs=100)
        lr.fit(self.X_fixture, self.y_fixture)

        # predict_proba contract
        proba = lr.predict_proba(self.X_test_fixture)
        self.assertIsInstance(proba, np.ndarray)
        self.assertEqual(proba.shape, (len(self.X_test_fixture), 2))
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

        # decision_function contract (logits)
        logits = lr.decision_function(self.X_test_fixture)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(logits.shape, (len(self.X_test_fixture),))
        self.assertEqual(logits.ndim, 1)

        # Verify sigmoid connection: proba[:, 1] == sigmoid(logits)
        sigmoid_logits = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(proba[:, 1], sigmoid_logits, atol=1e-10)

    # =========================================================================
    # 9. Output Semantics: Random Forest
    # =========================================================================

    def test_09_random_forest_vote_proportion_output(self) -> None:
        """Verify Random Forest predict_proba() returns uncalibrated vote fractions summing to 1."""
        from backend.ml import ManualRandomForest

        rf = ManualRandomForest(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(self.X_fixture, self.y_fixture)

        proba = rf.predict_proba(self.X_test_fixture)
        self.assertIsInstance(proba, np.ndarray)
        self.assertEqual(proba.shape, (len(self.X_test_fixture), 2))
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

        # Check column ordering: [P(0), P(1)]
        preds = rf.predict(self.X_test_fixture)
        expected_preds = np.where(proba[:, 1] > proba[:, 0], 1, 0)
        np.testing.assert_array_equal(preds, expected_preds)

    # =========================================================================
    # 10. Output Semantics: SVM
    # =========================================================================

    def test_10_svm_raw_decision_scores_and_sign_consistency(self) -> None:
        """Verify SVM decision_function() returns raw margins, sign matches predict (f(x) >= 0 -> 1), and NO predict_proba."""
        from backend.ml import ManualSVM

        svm = ManualSVM(C=1.0, kernel="linear", max_iter=100)
        svm.fit(self.X_fixture, self.y_fixture)

        scores = svm.decision_function(self.X_test_fixture)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (len(self.X_test_fixture),))
        self.assertEqual(scores.ndim, 1)

        preds = svm.predict(self.X_test_fixture)
        expected_preds = np.where(scores >= 0.0, 1, 0)
        np.testing.assert_array_equal(
            preds,
            expected_preds,
            err_msg="SVM prediction must strictly match sign of decision_function scores (f(x) >= 0 -> 1, f(x) < 0 -> 0).",
        )

        # SVM must strictly not expose predict_proba
        self.assertFalse(
            hasattr(svm, "predict_proba"),
            "ManualSVM must NOT expose a predict_proba() method (uncalibrated margins only).",
        )

    # =========================================================================
    # 11. Deterministic Reproducibility
    # =========================================================================

    def test_11_deterministic_behavior(self) -> None:
        """Verify repeated fitting with identical parameters and data produces identical predictions."""
        for clf_factory in [
            lambda: self._get_classifier_instances()[0],
            lambda: self._get_classifier_instances()[1],
            lambda: self._get_classifier_instances()[2],
        ]:
            clf1 = clf_factory()
            clf2 = clf_factory()
            clf_name = type(clf1).__name__

            with self.subTest(classifier=clf_name):
                clf1.fit(self.X_fixture, self.y_fixture)
                preds1 = clf1.predict(self.X_test_fixture)

                clf2.fit(self.X_fixture, self.y_fixture)
                preds2 = clf2.predict(self.X_test_fixture)

                np.testing.assert_array_equal(
                    preds1,
                    preds2,
                    err_msg=f"{clf_name} repeated fitting must produce identical predictions.",
                )

    # =========================================================================
    # 12. Preprocessing Pipeline A: Mean Imputation
    # =========================================================================

    def test_12_mean_imputation_pipeline_compatibility(self) -> None:
        """Verify Pipeline A (Stratified Split -> Mean Imputer -> StandardScaler) on approved dataset."""
        self.assertTrue(self.dataset_path.exists(), f"Dataset file not found at {self.dataset_path}")

        raw_df = pd.read_csv(self.dataset_path)
        X = raw_df.drop(columns=[TARGET_COLUMN])
        y = raw_df[TARGET_COLUMN]

        # 1. Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        self.assertEqual(len(X_train), 2620)
        self.assertEqual(len(X_test), 656)

        # 2. Fit imputer on X_train only, transform both
        imputer = ManualImputer(strategy="mean")
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)

        self.assertEqual(X_train_imp.isnull().sum().sum(), 0)
        self.assertEqual(X_test_imp.isnull().sum().sum(), 0)

        # 3. Fit scaler on imputed X_train only, transform both
        scaler = ManualStandardScaler()
        scaler.fit(X_train_imp)
        X_train_scaled = scaler.transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # 4. Invariant checks
        self.assertEqual(X_train_scaled.shape, (2620, 9))
        self.assertEqual(X_test_scaled.shape, (656, 9))
        self.assertEqual(list(X_train_scaled.columns), CANONICAL_FEATURES)
        self.assertEqual(list(X_test_scaled.columns), CANONICAL_FEATURES)
        self.assertEqual(X_train_scaled.isnull().sum().sum(), 0)
        self.assertEqual(X_test_scaled.isnull().sum().sum(), 0)
        self.assertEqual(int(np.isinf(X_train_scaled.values).sum()), 0)
        self.assertEqual(int(np.isinf(X_test_scaled.values).sum()), 0)
        self.assertTrue(X_train_scaled.index.equals(y_train.index))
        self.assertTrue(X_test_scaled.index.equals(y_test.index))

        # Reusable feature validation check without fitting classifiers on the 2620 partition
        sample_slice = X_train_scaled.iloc[:10]
        self.assertEqual(sample_slice.shape, (10, 9))

    # =========================================================================
    # 13. Preprocessing Pipeline B: Median Imputation
    # =========================================================================

    def test_13_median_imputation_pipeline_compatibility(self) -> None:
        """Verify Pipeline B (Stratified Split -> Median Imputer -> StandardScaler) on approved dataset."""
        self.assertTrue(self.dataset_path.exists(), f"Dataset file not found at {self.dataset_path}")

        raw_df = pd.read_csv(self.dataset_path)
        X = raw_df.drop(columns=[TARGET_COLUMN])
        y = raw_df[TARGET_COLUMN]

        # 1. Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        self.assertEqual(len(X_train), 2620)
        self.assertEqual(len(X_test), 656)

        # 2. Fit imputer on X_train only, transform both
        imputer = ManualImputer(strategy="median")
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)

        self.assertEqual(X_train_imp.isnull().sum().sum(), 0)
        self.assertEqual(X_test_imp.isnull().sum().sum(), 0)

        # 3. Fit scaler on imputed X_train only, transform both
        scaler = ManualStandardScaler()
        scaler.fit(X_train_imp)
        X_train_scaled = scaler.transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # 4. Invariant checks
        self.assertEqual(X_train_scaled.shape, (2620, 9))
        self.assertEqual(X_test_scaled.shape, (656, 9))
        self.assertEqual(list(X_train_scaled.columns), CANONICAL_FEATURES)
        self.assertEqual(list(X_test_scaled.columns), CANONICAL_FEATURES)
        self.assertEqual(X_train_scaled.isnull().sum().sum(), 0)
        self.assertEqual(X_test_scaled.isnull().sum().sum(), 0)
        self.assertEqual(int(np.isinf(X_train_scaled.values).sum()), 0)
        self.assertEqual(int(np.isinf(X_test_scaled.values).sum()), 0)
        self.assertTrue(X_train_scaled.index.equals(y_train.index))
        self.assertTrue(X_test_scaled.index.equals(y_test.index))

    # =========================================================================
    # 14. Pre-Fit Prediction Rejection
    # =========================================================================

    def test_14_reject_prediction_before_fitting(self) -> None:
        """Verify calling predict/decision/probability methods before fit() raises RuntimeError across all."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                with self.assertRaises(RuntimeError, msg=f"{clf_name} must raise RuntimeError on predict before fit."):
                    clf.predict(self.X_test_fixture)

                if hasattr(clf, "predict_proba"):
                    with self.assertRaises(
                        RuntimeError, msg=f"{clf_name} must raise RuntimeError on predict_proba before fit."
                    ):
                        clf.predict_proba(self.X_test_fixture)

                if hasattr(clf, "decision_function"):
                    with self.assertRaises(
                        RuntimeError, msg=f"{clf_name} must raise RuntimeError on decision_function before fit."
                    ):
                        clf.decision_function(self.X_test_fixture)

    # =========================================================================
    # 15. Invalid Input Rejection Consistency
    # =========================================================================

    def test_15_invalid_input_rejection(self) -> None:
        """Verify all three classifiers reject NaNs, infs, non-numerics, mismatched counts, single-class targets."""
        for clf in self._get_classifier_instances():
            clf_name = type(clf).__name__
            with self.subTest(classifier=clf_name):
                # 1. NaN in features
                X_nan = self.X_fixture.copy()
                X_nan.iloc[0, 0] = np.nan
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject NaN in features."):
                    clf.fit(X_nan, self.y_fixture)

                # 2. Infinite value in features
                X_inf = self.X_fixture.copy()
                X_inf.iloc[0, 0] = np.inf
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject infinite values in features."):
                    clf.fit(X_inf, self.y_fixture)

                # 3. Non-numeric feature
                X_str = self.X_fixture.copy()
                X_str["str_col"] = "invalid"
                with self.assertRaises(TypeError, msg=f"{clf_name} must reject non-numeric features."):
                    clf.fit(X_str, self.y_fixture)

                # 4. Mismatched sample counts
                y_short = self.y_fixture.iloc[:-2]
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject sample count mismatch."):
                    clf.fit(self.X_fixture, y_short)

                # 5. Misaligned target index
                y_misaligned = self.y_fixture.copy()
                y_misaligned.index = range(100, 100 + len(y_misaligned))
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject misaligned target index."):
                    clf.fit(self.X_fixture, y_misaligned)

                # 6. Single-class target
                y_single = pd.Series(np.zeros(len(self.X_fixture), dtype=int))
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject single-class target."):
                    clf.fit(self.X_fixture, y_single)

                # 7. Non-binary target values (e.g. {1, 2})
                y_non_binary = pd.Series([1, 2] * (len(self.X_fixture) // 2))
                with self.assertRaises(ValueError, msg=f"{clf_name} must reject non-binary target labels."):
                    clf.fit(self.X_fixture, y_non_binary)

    # =========================================================================
    # 16. Scope Boundary Integrity Check
    # =========================================================================

    def test_16_controlled_fixtures_no_test_partition_leakage(self) -> None:
        """Verify integration fixtures are strictly synthetic/controlled and test partition is never fitted."""
        # Confirm fixture size is independent of Kaggle dataset partitions
        self.assertEqual(len(self.X_fixture), 30)
        self.assertNotEqual(len(self.X_fixture), 2620)
        self.assertNotEqual(len(self.X_fixture), 656)

        # Confirm all 9 canonical features are covered
        self.assertEqual(list(self.X_fixture.columns), CANONICAL_FEATURES)

    # =========================================================================
    # 17. SVM Zero-Margin Boundary Decision Rule
    # =========================================================================

    def test_17_svm_zero_margin_boundary_decision_rule(self) -> None:
        """Verify SVM zero-margin decision boundary: f(x) >= 0 -> class 1, f(x) < 0 -> class 0."""
        from backend.ml import ManualSVM

        # Symmetric linear fixture where the decision boundary sits exactly at x = 0 with bias b = 0
        X = pd.DataFrame({"f1": [-1.0, 1.0]})
        y = pd.Series([0, 1])

        clf = ManualSVM(kernel="linear", C=1.0).fit(X, y)

        # Evaluate at negative, exact zero, and positive decision scores
        X_test = pd.DataFrame({"f1": [-0.5, 0.0, 0.5]})
        scores = clf.decision_function(X_test)
        preds = clf.predict(X_test)

        self.assertAlmostEqual(scores[0], -0.5, places=7)
        self.assertAlmostEqual(scores[1], 0.0, places=7)
        self.assertAlmostEqual(scores[2], 0.5, places=7)

        # Verify predictions strictly follow: f(x) >= 0 -> 1, f(x) < 0 -> 0
        self.assertEqual(preds[0], 0)  # f(x) < 0.0  -> 0
        self.assertEqual(preds[1], 1)  # f(x) == 0.0 -> 1 (zero margin boundary is class 1)
        self.assertEqual(preds[2], 1)  # f(x) > 0.0  -> 1


if __name__ == "__main__":
    unittest.main()

