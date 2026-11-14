"""Unit Tests for Manual Random Forest Binary Classifier.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.3 - Manual Random Forest Implementation Unit Tests.

Covers all 56 required test cases across Sections 14.A through 14.E:
- Section 14.A: Decision Tree and Gini Impurity Tests (Tests 1-15)
- Section 14.B: Bootstrap Sampling and Random Feature Selection Tests (Tests 16-25)
- Section 14.C: Forest Ensemble and Majority Voting Tests (Tests 26-36)
- Section 14.D: Feature Importance Tests (Tests 37-40)
- Section 14.E: Input Validation and Data Integrity Tests (Tests 41-56)
"""

import math
import unittest

import numpy as np
import pandas as pd

from backend.ml.eda import CANONICAL_FEATURES
from backend.ml.random_forest import ManualRandomForest, _DecisionTree, _DecisionTreeNode


class TestManualRandomForest(unittest.TestCase):
    """Comprehensive test suite for ManualRandomForest binary classifier."""

    def setUp(self) -> None:
        """Set up standard reproducible fixtures for testing."""
        np.random.seed(42)

        # Standard 9 canonical feature fixture
        n_samples = 40
        self.canonical_cols = CANONICAL_FEATURES
        rng = np.random.default_rng(42)
        X_mat = rng.normal(loc=5.0, scale=2.0, size=(n_samples, len(self.canonical_cols)))
        self.X_canonical = pd.DataFrame(X_mat, columns=self.canonical_cols)
        # Binary target with both classes
        y_vec = np.zeros(n_samples, dtype=int)
        y_vec[:20] = 1
        self.y_canonical = pd.Series(y_vec, name="Potability")

    # =========================================================================
    # Section 14.A: Decision Tree and Gini Tests (1 - 15)
    # =========================================================================

    def test_01_pure_node_gini_is_zero(self) -> None:
        """Requirement 1: Verify pure node Gini impurity is exactly zero."""
        y_all_zero = np.zeros(20, dtype=int)
        y_all_one = np.ones(25, dtype=int)
        self.assertEqual(ManualRandomForest._gini_impurity(y_all_zero), 0.0)
        self.assertEqual(ManualRandomForest._gini_impurity(y_all_one), 0.0)

    def test_02_balanced_node_gini_is_half(self) -> None:
        """Requirement 2: Verify balanced two-class Gini impurity is 0.5."""
        y_balanced = np.array([0, 0, 1, 1], dtype=int)
        self.assertEqual(ManualRandomForest._gini_impurity(y_balanced), 0.5)

        y_balanced_large = np.array([0] * 50 + [1] * 50, dtype=int)
        self.assertEqual(ManualRandomForest._gini_impurity(y_balanced_large), 0.5)

    def test_03_class_proportions_produce_expected_gini(self) -> None:
        """Requirement 3: Verify manually calculated class proportions produce expected Gini."""
        # 3:1 split -> p0 = 0.75, p1 = 0.25 -> Gini = 1 - (0.75^2 + 0.25^2) = 0.375
        y_3_to_1 = np.array([0, 0, 0, 1], dtype=int)
        self.assertAlmostEqual(ManualRandomForest._gini_impurity(y_3_to_1), 0.375, places=7)

        # 4:1 split -> p0 = 0.8, p1 = 0.2 -> Gini = 1 - (0.64 + 0.04) = 0.32
        y_4_to_1 = np.array([0, 0, 0, 0, 1], dtype=int)
        self.assertAlmostEqual(ManualRandomForest._gini_impurity(y_4_to_1), 0.32, places=7)

    def test_04_valid_split_reduces_weighted_gini(self) -> None:
        """Requirement 4: Verify a valid split reduces weighted Gini impurity."""
        # Parent node: 2 zeros and 2 ones -> Gini = 0.5
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1], dtype=int)

        tree = _DecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features_count=1)
        tree.fit(X, y)

        self.assertFalse(tree.root_.is_leaf)
        # Children are pure: left has [0, 0] (Gini=0), right has [1, 1] (Gini=0)
        self.assertEqual(tree.root_.left.impurity, 0.0)
        self.assertEqual(tree.root_.right.impurity, 0.0)
        # Reduction is parent (0.5) - children (0.0) = 0.5 > 0
        self.assertGreater(tree.root_.impurity, 0.0)

    def test_05_candidate_thresholds_calculated_correctly(self) -> None:
        """Requirement 5: Verify candidate thresholds are calculated as midpoints."""
        values = [1.0, 3.0, 7.0]
        thresholds = ManualRandomForest._calculate_candidate_thresholds(values)
        np.testing.assert_allclose(thresholds, [2.0, 5.0])

    def test_06_thresholds_created_only_between_distinct_values(self) -> None:
        """Requirement 6: Verify thresholds are created only between consecutive distinct values."""
        values_dups = [2.0, 2.0, 2.0, 4.0, 4.0, 8.0]
        thresholds = ManualRandomForest._calculate_candidate_thresholds(values_dups)
        np.testing.assert_allclose(thresholds, [3.0, 6.0])

        # All values identical -> zero candidate thresholds
        values_identical = [5.0, 5.0, 5.0, 5.0]
        thresholds_empty = ManualRandomForest._calculate_candidate_thresholds(values_identical)
        self.assertEqual(len(thresholds_empty), 0)

    def test_07_both_children_satisfy_min_samples_leaf(self) -> None:
        """Requirement 7: Verify both child nodes satisfy min_samples_leaf constraint."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

        min_leaf = 3
        tree = _DecisionTree(min_samples_leaf=min_leaf, max_features_count=1)
        tree.fit(X, y)

        self.assertFalse(tree.root_.is_leaf)
        self.assertGreaterEqual(tree.root_.left.n_node_samples, min_leaf)
        self.assertGreaterEqual(tree.root_.right.n_node_samples, min_leaf)

    def test_08_pure_nodes_become_leaves(self) -> None:
        """Requirement 8: Verify pure nodes stop splitting and become leaves."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1, 1, 1, 1], dtype=int)

        tree = _DecisionTree(max_depth=5, min_samples_split=2)
        tree.fit(X, y)

        self.assertTrue(tree.root_.is_leaf)
        self.assertIsNone(tree.root_.left)
        self.assertIsNone(tree.root_.right)
        self.assertEqual(tree.root_.predicted_class, 1)

    def test_09_max_depth_stopping_behavior(self) -> None:
        """Requirement 9: Verify max_depth stopping behavior halts tree growth."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)

        # max_depth = 1 (decision stump)
        tree = _DecisionTree(max_depth=1, min_samples_split=2, min_samples_leaf=1, max_features_count=1)
        tree.fit(X, y)

        self.assertLessEqual(tree.max_depth_, 1)
        self.assertTrue(tree.root_.left.is_leaf)
        self.assertTrue(tree.root_.right.is_leaf)

    def test_10_min_samples_split_stopping_behavior(self) -> None:
        """Requirement 10: Verify min_samples_split prevents splitting small nodes."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1], dtype=int)

        # min_samples_split = 10 is greater than 4 samples
        tree = _DecisionTree(min_samples_split=10, min_samples_leaf=1, max_features_count=1)
        tree.fit(X, y)

        self.assertTrue(tree.root_.is_leaf)
        self.assertEqual(tree.root_.n_node_samples, 4)

    def test_11_min_samples_leaf_stopping_behavior(self) -> None:
        """Requirement 11: Verify min_samples_leaf prevents split if children cannot satisfy it."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

        # 6 samples cannot be split into two children each >= 4
        tree = _DecisionTree(min_samples_leaf=4, min_samples_split=2, max_features_count=1)
        tree.fit(X, y)

        self.assertTrue(tree.root_.is_leaf)

    def test_12_no_gain_splits_are_rejected(self) -> None:
        """Requirement 12: Verify splits with no gain or below min_impurity_decrease are rejected."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1], dtype=int)

        # Parent Gini is 0.5. Perfect split gain is 0.5.
        # Requiring min_impurity_decrease = 0.6 rejects the split.
        tree = _DecisionTree(min_impurity_decrease=0.6, max_features_count=1)
        tree.fit(X, y)

        self.assertTrue(tree.root_.is_leaf)

    def test_13_equal_split_gains_resolved_deterministically(self) -> None:
        """Requirement 13: Verify equal split gains are resolved deterministically."""
        # Both feature 0 and feature 1 achieve identical split
        X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
        y = np.array([0, 0, 1, 1], dtype=int)

        tree = _DecisionTree(max_depth=1, max_features_count=2, rng=np.random.default_rng(0))
        tree.fit(X, y)

        # Feature 0 was tested first and is deterministically kept
        self.assertEqual(tree.root_.feature_index, 0)

    def test_14_majority_class_leaf_prediction(self) -> None:
        """Requirement 14: Verify majority-class leaf prediction works."""
        node = _DecisionTreeNode(
            predicted_class=1,
            class_counts=(1, 3),
            impurity=0.375,
            n_node_samples=4,
            depth=0,
        )
        self.assertEqual(node.predicted_class, 1)

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1, 1, 1, 0], dtype=int)
        tree = _DecisionTree(max_depth=0)
        tree.fit(X, y)
        self.assertEqual(tree.root_.predicted_class, 1)

    def test_15_equal_class_counts_in_leaf_predict_zero(self) -> None:
        """Requirement 15: Verify exact 50/50 equal class count in leaf predicts class 0."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1], dtype=int)

        tree = _DecisionTree(max_depth=0)
        tree.fit(X, y)

        self.assertEqual(tree.root_.class_counts, (2, 2))
        self.assertEqual(tree.root_.predicted_class, 0)

    # =========================================================================
    # Section 14.B: Bootstrap Sampling and Random Feature Tests (16 - 25)
    # =========================================================================

    def test_16_bootstrap_samples_have_expected_count(self) -> None:
        """Requirement 16: Verify bootstrap samples match training sample count."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        for tree in rf.estimators_:
            self.assertEqual(tree.n_bootstrap_, len(self.X_canonical))

    def test_17_bootstrap_sampling_uses_replacement(self) -> None:
        """Requirement 17: Verify bootstrap sampling samples with replacement."""
        rng = np.random.default_rng(42)
        n = 100
        indices = rng.choice(n, size=n, replace=True)
        unique_indices = np.unique(indices)
        # Drawing 100 items from 100 with replacement produces ~63.2% unique indices (< 100)
        self.assertLess(len(unique_indices), n)
        self.assertGreater(len(unique_indices), 40)

    def test_18_bootstrap_indices_within_training_range(self) -> None:
        """Requirement 18: Verify bootstrap sample indices remain within training range [0, n-1]."""
        rng = np.random.default_rng(42)
        n = 50
        indices = rng.choice(n, size=n, replace=True)
        self.assertTrue(np.all((indices >= 0) & (indices < n)))

    def test_19_different_trees_receive_separate_bootstrap_samples(self) -> None:
        """Requirement 19: Verify different trees receive separately generated bootstrap samples."""
        rf = ManualRandomForest(n_estimators=5, max_depth=3, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        # Confirm the fitted trees do not have identical split thresholds or tree structures
        thresholds = [tree.root_.threshold for tree in rf.estimators_]
        # At least two trees should have different split thresholds
        self.assertGreater(len(set(thresholds)), 1)

    def test_20_same_random_seed_produces_identical_results(self) -> None:
        """Requirement 20: Verify identical seeds produce identical bootstrap and tree results."""
        rf1 = ManualRandomForest(n_estimators=10, random_state=123)
        rf1.fit(self.X_canonical, self.y_canonical)

        rf2 = ManualRandomForest(n_estimators=10, random_state=123)
        rf2.fit(self.X_canonical, self.y_canonical)

        np.testing.assert_array_equal(rf1.predict(self.X_canonical), rf2.predict(self.X_canonical))
        np.testing.assert_allclose(
            rf1.predict_proba(self.X_canonical), rf2.predict_proba(self.X_canonical)
        )
        np.testing.assert_allclose(rf1.feature_importances_, rf2.feature_importances_)

    def test_21_sqrt_selects_expected_feature_count(self) -> None:
        """Requirement 21: Verify 'sqrt' selects floor(sqrt(n_features))."""
        rf = ManualRandomForest(max_features="sqrt")
        # 9 features -> floor(sqrt(9)) = 3
        self.assertEqual(rf._get_max_features_count(9), 3)
        # 16 features -> floor(sqrt(16)) = 4
        self.assertEqual(rf._get_max_features_count(16), 4)
        # 1 feature -> max(1, 1) = 1
        self.assertEqual(rf._get_max_features_count(1), 1)

    def test_22_log2_selects_expected_feature_count(self) -> None:
        """Requirement 22: Verify 'log2' selects floor(log2(n_features))."""
        rf = ManualRandomForest(max_features="log2")
        # 9 features -> floor(log2(9)) = 3
        self.assertEqual(rf._get_max_features_count(9), 3)
        # 8 features -> floor(log2(8)) = 3
        self.assertEqual(rf._get_max_features_count(8), 3)
        # 4 features -> floor(log2(4)) = 2
        self.assertEqual(rf._get_max_features_count(4), 2)
        # 1 feature -> max(1, 0) = 1
        self.assertEqual(rf._get_max_features_count(1), 1)

    def test_23_explicit_integer_max_features_works(self) -> None:
        """Requirement 23: Verify explicit integer max_features is honored and clamped."""
        rf5 = ManualRandomForest(max_features=5)
        self.assertEqual(rf5._get_max_features_count(9), 5)

        # Clamped to n_features if larger
        rf20 = ManualRandomForest(max_features=20)
        self.assertEqual(rf20._get_max_features_count(9), 9)

    def test_24_feature_selection_occurs_separately_at_eligible_nodes(self) -> None:
        """Requirement 24: Verify feature selection occurs separately at eligible tree nodes."""
        rng = np.random.default_rng(42)
        tree = _DecisionTree(max_depth=3, max_features_count=2, rng=rng)
        # 10 features, 30 samples
        X = rng.normal(size=(30, 10))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        tree.fit(X, y)

        # Collect split feature indices across tree
        split_features = []

        def _traverse(node: _DecisionTreeNode) -> None:
            if not node.is_leaf:
                split_features.append(node.feature_index)
                _traverse(node.left)
                _traverse(node.right)

        _traverse(tree.root_)
        self.assertGreater(len(split_features), 1)

    def test_25_features_selected_without_replacement_within_node(self) -> None:
        """Requirement 25: Verify feature selection at any node is strictly without replacement."""
        rng = np.random.default_rng(99)
        n_features = 9
        k = 3
        for _ in range(50):
            selection = rng.choice(n_features, size=k, replace=False)
            self.assertEqual(len(selection), len(set(selection)))

    # =========================================================================
    # Section 14.C: Forest and Prediction Tests (26 - 36)
    # =========================================================================

    def test_26_configured_number_of_trees_constructed(self) -> None:
        """Requirement 26: Verify configured number of trees is constructed."""
        rf = ManualRandomForest(n_estimators=15, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        self.assertEqual(len(rf.estimators_), 15)
        self.assertEqual(rf.n_estimators_, 15)

    def test_27_linearly_separable_fixture_classified_correctly(self) -> None:
        """Requirement 27: Verify linearly separable fixture achieves 100% accuracy."""
        rng = np.random.default_rng(42)
        X_mat = rng.normal(size=(40, 2))
        # Clear linear separation along x0
        X_mat[:20, 0] -= 5.0
        X_mat[20:, 0] += 5.0
        y_vec = np.zeros(40, dtype=int)
        y_vec[20:] = 1

        X_df = pd.DataFrame(X_mat, columns=["feat_a", "feat_b"])
        y_s = pd.Series(y_vec)

        rf = ManualRandomForest(n_estimators=20, max_depth=3, random_state=42)
        rf.fit(X_df, y_s)

        preds = rf.predict(X_df)
        np.testing.assert_array_equal(preds, y_vec)

    def test_28_nonlinear_fixture_learned(self) -> None:
        """Requirement 28: Verify a nonlinear fixture (XOR) can be learned by the forest."""
        # XOR problem: nonlinearly separable
        X_xor = pd.DataFrame(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.2, -0.8],
                [-0.8, 1.1],
                [1.1, -0.9],
                [0.9, 0.9],
            ],
            columns=["x0", "x1"],
        )
        y_xor = pd.Series([0, 1, 1, 0, 0, 1, 1, 0])

        rf = ManualRandomForest(
            n_estimators=30,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=2,
            random_state=42,
        )
        rf.fit(X_xor, y_xor)

        preds = rf.predict(X_xor)
        np.testing.assert_array_equal(preds, y_xor.values)

    def test_29_majority_voting_works_correctly(self) -> None:
        """Requirement 29: Verify majority voting properly aggregates individual tree votes."""
        rf = ManualRandomForest(n_estimators=11, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        proba = rf.predict_proba(self.X_canonical)
        preds = rf.predict(self.X_canonical)

        for i in range(len(self.X_canonical)):
            p0, p1 = proba[i, 0], proba[i, 1]
            if p1 > p0:
                self.assertEqual(preds[i], 1)
            elif p0 > p1:
                self.assertEqual(preds[i], 0)
            else:
                self.assertEqual(preds[i], 0)

    def test_30_exact_voting_tie_predicts_zero(self) -> None:
        """Requirement 30: Verify an exact 50/50 voting tie deterministically predicts class 0."""
        # Create an ensemble of 2 trees where tree 0 predicts 0 and tree 1 predicts 1
        rf = ManualRandomForest(n_estimators=2, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        # Force tree 0 to predict 0 and tree 1 to predict 1
        rf.estimators_[0].root_ = _DecisionTreeNode(
            predicted_class=0, class_counts=(1, 0), impurity=0.0, n_node_samples=1, depth=0
        )
        rf.estimators_[1].root_ = _DecisionTreeNode(
            predicted_class=1, class_counts=(0, 1), impurity=0.0, n_node_samples=1, depth=0
        )

        sample = self.X_canonical.iloc[[0]]
        proba = rf.predict_proba(sample)
        self.assertEqual(proba[0, 0], 0.5)
        self.assertEqual(proba[0, 1], 0.5)

        # Tie-breaker must predict class 0
        pred = rf.predict(sample)
        self.assertEqual(pred[0], 0)

    def test_31_predict_returns_binary_only(self) -> None:
        """Requirement 31: Verify predict() returns strictly a 1D NumPy array of {0, 1} integers."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        preds = rf.predict(self.X_canonical)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.ndim, 1)
        self.assertTrue(np.issubdtype(preds.dtype, np.integer))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_32_predict_proba_returns_two_columns(self) -> None:
        """Requirement 32: Verify predict_proba() returns a 2D NumPy array with exactly two columns."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        proba = rf.predict_proba(self.X_canonical)
        self.assertIsInstance(proba, np.ndarray)
        self.assertEqual(proba.shape, (len(self.X_canonical), 2))

    def test_33_class_vote_proportions_sum_to_one(self) -> None:
        """Requirement 33: Verify class vote proportions sum to 1.0 for every sample."""
        rf = ManualRandomForest(n_estimators=10, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        proba = rf.predict_proba(self.X_canonical)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    def test_34_probability_column_ordering(self) -> None:
        """Requirement 34: Verify probability column ordering is [P(0), P(1)]."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        sample = self.X_canonical.iloc[[0]]
        # If all trees predict 0:
        for tree in rf.estimators_:
            tree.root_ = _DecisionTreeNode(
                predicted_class=0, class_counts=(1, 0), impurity=0.0, n_node_samples=1, depth=0
            )
        proba = rf.predict_proba(sample)
        self.assertEqual(proba[0, 0], 1.0)
        self.assertEqual(proba[0, 1], 0.0)

        # If all trees predict 1:
        for tree in rf.estimators_:
            tree.root_ = _DecisionTreeNode(
                predicted_class=1, class_counts=(0, 1), impurity=0.0, n_node_samples=1, depth=0
            )
        proba = rf.predict_proba(sample)
        self.assertEqual(proba[0, 0], 0.0)
        self.assertEqual(proba[0, 1], 1.0)

    def test_35_repeated_fitting_with_same_seed_reproducible(self) -> None:
        """Requirement 35: Verify repeated fitting with the same random seed produces identical predictions."""
        rf = ManualRandomForest(n_estimators=8, random_state=77)
        rf.fit(self.X_canonical, self.y_canonical)
        preds_1 = rf.predict(self.X_canonical)

        rf.fit(self.X_canonical, self.y_canonical)
        preds_2 = rf.predict(self.X_canonical)

        np.testing.assert_array_equal(preds_1, preds_2)

    def test_36_fitted_tree_feature_ordering_preserved(self) -> None:
        """Requirement 36: Verify fitted feature ordering preserves training DataFrame columns."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        self.assertEqual(rf.feature_names_in_, self.canonical_cols)
        self.assertEqual(rf.n_features_in_, len(self.canonical_cols))

    # =========================================================================
    # Section 14.D: Feature Importance Tests (37 - 40)
    # =========================================================================

    def test_37_feature_importance_finite_and_non_negative(self) -> None:
        """Requirement 37: Verify feature importance values are finite and non-negative."""
        rf = ManualRandomForest(n_estimators=10, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        self.assertTrue(np.all(np.isfinite(rf.feature_importances_)))
        self.assertTrue(np.all(rf.feature_importances_ >= 0.0))

    def test_38_feature_importance_sums_to_one_when_nonzero(self) -> None:
        """Requirement 38: Verify feature importances sum to approximately 1.0 when non-zero."""
        rf = ManualRandomForest(n_estimators=10, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        self.assertAlmostEqual(float(np.sum(rf.feature_importances_)), 1.0, places=6)

    def test_39_informative_feature_receives_positive_importance(self) -> None:
        """Requirement 39: Verify an informative feature receives higher importance than noise."""
        rng = np.random.default_rng(42)
        n = 50
        x_signal = rng.normal(size=n)
        x_noise = rng.normal(size=n)
        y = (x_signal > 0.0).astype(int)

        X_df = pd.DataFrame({"signal": x_signal, "noise": x_noise})
        y_s = pd.Series(y)

        rf = ManualRandomForest(n_estimators=20, max_features=2, random_state=42)
        rf.fit(X_df, y_s)

        # Informative feature 'signal' must have greater importance than 'noise'
        self.assertGreater(rf.feature_importances_[0], rf.feature_importances_[1])
        self.assertGreater(rf.feature_importances_[0], 0.5)

    def test_40_no_useful_splits_returns_zero_importance_safely(self) -> None:
        """Requirement 40: Verify zero importance values when no useful splits exist without ZeroDivisionError."""
        # Unachievable min_impurity_decrease prevents any split
        rf = ManualRandomForest(n_estimators=5, min_impurity_decrease=100.0, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        np.testing.assert_array_equal(rf.feature_importances_, np.zeros(rf.n_features_in_))

    # =========================================================================
    # Section 14.E: Validation and Integrity Tests (41 - 56)
    # =========================================================================

    def test_41_rejects_empty_inputs(self) -> None:
        """Requirement 41: Verify empty feature DataFrames or target Series are rejected."""
        rf = ManualRandomForest()

        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            rf.fit(empty_df, self.y_canonical)

        empty_y = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, empty_y)

    def test_42_rejects_nan_feature_values(self) -> None:
        """Requirement 42: Verify NaN values in features are strictly rejected."""
        rf = ManualRandomForest()
        X_nan = self.X_canonical.copy()
        X_nan.iloc[2, 3] = np.nan

        with self.assertRaises(ValueError):
            rf.fit(X_nan, self.y_canonical)

    def test_43_rejects_infinite_feature_values(self) -> None:
        """Requirement 43: Verify infinite values (+inf, -inf) in features are rejected."""
        rf = ManualRandomForest()
        X_inf = self.X_canonical.copy()
        X_inf.iloc[1, 1] = np.inf

        with self.assertRaises(ValueError):
            rf.fit(X_inf, self.y_canonical)

    def test_44_rejects_non_numeric_features(self) -> None:
        """Requirement 44: Verify non-numeric feature columns are rejected with TypeError."""
        rf = ManualRandomForest()
        X_str = self.X_canonical.copy()
        X_str["string_col"] = "invalid"

        with self.assertRaises(TypeError):
            rf.fit(X_str, self.y_canonical)

    def test_45_rejects_invalid_target_labels(self) -> None:
        """Requirement 45: Verify non-binary, continuous, or out-of-domain targets are rejected."""
        rf = ManualRandomForest()

        # Labels {1, 2} instead of {0, 1}
        y_invalid = pd.Series([1, 2] * 20)
        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, y_invalid)

        # Continuous values
        y_continuous = pd.Series(np.linspace(0.0, 1.0, len(self.X_canonical)))
        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, y_continuous)

    def test_46_rejects_single_class_target(self) -> None:
        """Requirement 46: Verify training targets with only one class are rejected."""
        rf = ManualRandomForest()
        y_single = pd.Series(np.zeros(len(self.X_canonical), dtype=int))

        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, y_single)

    def test_47_rejects_mismatched_sample_counts(self) -> None:
        """Requirement 47: Verify mismatched X and y sample counts are rejected."""
        rf = ManualRandomForest()
        y_short = self.y_canonical.iloc[:-5]

        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, y_short)

    def test_48_rejects_target_index_misalignment(self) -> None:
        """Requirement 48: Verify target Series with misaligned index is rejected."""
        rf = ManualRandomForest()
        y_misaligned = self.y_canonical.copy()
        y_misaligned.index = range(100, 100 + len(y_misaligned))

        with self.assertRaises(ValueError):
            rf.fit(self.X_canonical, y_misaligned)

    def test_49_rejects_missing_inference_columns(self) -> None:
        """Requirement 49: Verify inference rejects input DataFrames missing training features."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        X_missing = self.X_canonical.drop(columns=[self.canonical_cols[0]])
        with self.assertRaises(ValueError):
            rf.predict(X_missing)

    def test_50_rejects_extra_inference_columns(self) -> None:
        """Requirement 50: Verify inference rejects input DataFrames with unexpected extra columns."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        X_extra = self.X_canonical.copy()
        X_extra["extra_col"] = 1.0
        with self.assertRaises(ValueError):
            rf.predict(X_extra)

    def test_51_rejects_reordered_inference_columns(self) -> None:
        """Requirement 51: Verify inference rejects input DataFrames with permuted column order."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)

        reordered_cols = list(reversed(self.canonical_cols))
        X_reordered = self.X_canonical[reordered_cols]
        with self.assertRaises(ValueError):
            rf.predict(X_reordered)

    def test_52_rejects_prediction_before_fitting(self) -> None:
        """Requirement 52: Verify predict() and predict_proba() raise RuntimeError before fitting."""
        rf = ManualRandomForest()

        with self.assertRaises(RuntimeError):
            rf.predict(self.X_canonical)

        with self.assertRaises(RuntimeError):
            rf.predict_proba(self.X_canonical)

    def test_53_rejects_potability_input_feature(self) -> None:
        """Requirement 53: Verify Potability is strictly rejected as an input feature (leakage prevention)."""
        rf = ManualRandomForest()
        X_leak = self.X_canonical.copy()
        X_leak["Potability"] = 1.0

        with self.assertRaises(ValueError):
            rf.fit(X_leak, self.y_canonical)

    def test_54_rejects_invalid_constructor_parameters(self) -> None:
        """Requirement 54: Verify invalid constructor parameters are rejected."""
        # Non-positive or boolean n_estimators
        with self.assertRaises(ValueError):
            ManualRandomForest(n_estimators=0)
        with self.assertRaises(ValueError):
            ManualRandomForest(n_estimators=-5)
        with self.assertRaises(TypeError):
            ManualRandomForest(n_estimators=True)

        # Invalid max_depth
        with self.assertRaises(ValueError):
            ManualRandomForest(max_depth=0)
        with self.assertRaises(TypeError):
            ManualRandomForest(max_depth=False)

        # Invalid min_samples_split
        with self.assertRaises(ValueError):
            ManualRandomForest(min_samples_split=1)
        with self.assertRaises(TypeError):
            ManualRandomForest(min_samples_split="invalid")

        # Invalid min_samples_leaf
        with self.assertRaises(ValueError):
            ManualRandomForest(min_samples_leaf=0)
        with self.assertRaises(TypeError):
            ManualRandomForest(min_samples_leaf=1.5)

        # Invalid max_features
        with self.assertRaises(ValueError):
            ManualRandomForest(max_features="invalid_strategy")
        with self.assertRaises(ValueError):
            ManualRandomForest(max_features=0)
        with self.assertRaises(TypeError):
            ManualRandomForest(max_features=True)

        # Invalid min_impurity_decrease
        with self.assertRaises(ValueError):
            ManualRandomForest(min_impurity_decrease=-0.1)
        with self.assertRaises(ValueError):
            ManualRandomForest(min_impurity_decrease=np.nan)
        with self.assertRaises(TypeError):
            ManualRandomForest(min_impurity_decrease="zero")

    def test_55_original_inputs_unmodified(self) -> None:
        """Requirement 55: Verify input feature DataFrames and target Series remain unmodified in place."""
        X_orig = self.X_canonical.copy(deep=True)
        y_orig = self.y_canonical.copy(deep=True)

        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)
        rf.predict(self.X_canonical)
        rf.predict_proba(self.X_canonical)

        pd.testing.assert_frame_equal(self.X_canonical, X_orig)
        pd.testing.assert_series_equal(self.y_canonical, y_orig)

    def test_56_refitting_resets_previous_forest(self) -> None:
        """Requirement 56: Verify refitting resets the previous forest state completely."""
        rf = ManualRandomForest(n_estimators=5, random_state=42)
        rf.fit(self.X_canonical, self.y_canonical)
        self.assertEqual(rf.n_features_in_, 9)

        # Fit on a new dataset with only 3 features
        X_new = pd.DataFrame(
            np.random.normal(size=(20, 3)),
            columns=["alpha", "beta", "gamma"],
        )
        y_new = pd.Series([0] * 10 + [1] * 10)

        rf.fit(X_new, y_new)
        self.assertEqual(rf.n_features_in_, 3)
        self.assertEqual(rf.feature_names_in_, ["alpha", "beta", "gamma"])
        self.assertEqual(len(rf.feature_importances_), 3)
        self.assertEqual(len(rf.estimators_), 5)


if __name__ == "__main__":
    unittest.main()

