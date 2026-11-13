"""Automated Unit Tests for Manual Stratified Train-Test Splitting.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

Requirements:
- Tests all 11 required scenarios from Phase 2.2 specification.
- Uses Python's built-in unittest module only (no pytest, no scikit-learn).
- Uses the approved Water Potability Dataset for integration testing.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.data_split import (
    get_split_summary,
    load_and_split_data,
    stratified_train_test_split,
)
from backend.ml.eda import (
    CANONICAL_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)


class TestStratifiedTrainTestSplit(unittest.TestCase):
    """Comprehensive test suite for custom stratified splitting implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the approved water potability dataset once for the test suite."""
        cls.df = load_dataset()
        cls.X = cls.df[CANONICAL_FEATURES]
        cls.y = cls.df[TARGET_COLUMN]

    def test_01_correct_split_size(self) -> None:
        """Test 1: Verify correct sample counts in training, testing, and total subsets."""
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        self.assertEqual(len(X_train), 2620, "Training set feature count must equal 2,620.")
        self.assertEqual(len(y_train), 2620, "Training set target count must equal 2,620.")
        self.assertEqual(len(X_test), 656, "Testing set feature count must equal 656.")
        self.assertEqual(len(y_test), 656, "Testing set target count must equal 656.")
        self.assertEqual(
            len(X_train) + len(X_test), 3276, "Total samples must equal 3,276."
        )

    def test_02_class_proportion_preservation(self) -> None:
        """Test 2: Verify exact class counts and proportion preservation in subsets."""
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        # Training counts
        train_counts = y_train.value_counts().to_dict()
        self.assertEqual(train_counts.get(0), 1598, "Train Class 0 must contain 1,598 samples.")
        self.assertEqual(train_counts.get(1), 1022, "Train Class 1 must contain 1,022 samples.")

        # Testing counts
        test_counts = y_test.value_counts().to_dict()
        self.assertEqual(test_counts.get(0), 400, "Test Class 0 must contain 400 samples.")
        self.assertEqual(test_counts.get(1), 256, "Test Class 1 must contain 256 samples.")

        # Proportions check (within 0.05% of original dataset)
        orig_class0_pct = (self.y == 0).mean() * 100.0
        train_class0_pct = (y_train == 0).mean() * 100.0
        test_class0_pct = (y_test == 0).mean() * 100.0

        self.assertAlmostEqual(train_class0_pct, orig_class0_pct, delta=0.05)
        self.assertAlmostEqual(test_class0_pct, orig_class0_pct, delta=0.05)

    def test_03_no_data_overlap(self) -> None:
        """Test 3: Verify that training and testing subsets have no overlapping row indices."""
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        intersection = train_indices.intersection(test_indices)
        self.assertEqual(
            len(intersection), 0, f"Detected {len(intersection)} overlapping indices between train and test."
        )

    def test_04_complete_record_coverage(self) -> None:
        """Test 4: Verify that the union of train and test indices covers every original record exactly once."""
        X_train, X_test, _, _ = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        combined_indices = set(X_train.index).union(set(X_test.index))
        original_indices = set(self.X.index)
        self.assertEqual(combined_indices, original_indices, "Union of subsets must match original indices.")
        self.assertEqual(len(combined_indices), len(self.X), "No records may be lost or duplicated.")

    def test_05_reproducibility(self) -> None:
        """Test 5: Verify that identical seeds produce identical train and test splits."""
        X_train_1, X_test_1, y_train_1, y_test_1 = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        X_train_2, X_test_2, y_train_2, y_test_2 = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )

        self.assertTrue(X_train_1.index.equals(X_train_2.index), "Training indices must match across runs.")
        self.assertTrue(X_test_1.index.equals(X_test_2.index), "Testing indices must match across runs.")
        self.assertTrue(y_train_1.equals(y_train_2), "Training targets must match across runs.")
        self.assertTrue(y_test_1.equals(y_test_2), "Testing targets must match across runs.")

    def test_06_different_seed(self) -> None:
        """Test 6: Verify that a different seed produces a different partition with identical class counts."""
        X_train_42, X_test_42, y_train_42, _ = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        X_train_99, X_test_99, y_train_99, y_test_99 = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=99
        )

        # Indices must differ between seeds
        self.assertFalse(
            X_train_42.index.equals(X_train_99.index),
            "Splits with different seeds must produce different index orderings.",
        )

        # But sample counts and class allocations must remain identical
        self.assertEqual(len(X_train_99), 2620)
        self.assertEqual(len(X_test_99), 656)
        self.assertEqual(int((y_train_99 == 0).sum()), 1598)
        self.assertEqual(int((y_train_99 == 1).sum()), 1022)
        self.assertEqual(int((y_test_99 == 0).sum()), 400)
        self.assertEqual(int((y_test_99 == 1).sum()), 256)

    def test_07_feature_target_alignment(self) -> None:
        """Test 7: Verify feature and target indices and values remain perfectly aligned with source data."""
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        # Check index equality between X and y
        self.assertTrue(X_train.index.equals(y_train.index), "X_train and y_train indices must be identical.")
        self.assertTrue(X_test.index.equals(y_test.index), "X_test and y_test indices must be identical.")

        # Check a sample of rows against original DataFrame
        sample_indices = list(X_train.index[:10]) + list(X_test.index[:10])
        for idx in sample_indices:
            expected_target = self.df.loc[idx, TARGET_COLUMN]
            actual_target = y_train.loc[idx] if idx in y_train.index else y_test.loc[idx]
            self.assertEqual(actual_target, expected_target, f"Target mismatch at original index {idx}.")

    def test_08_feature_order_preservation(self) -> None:
        """Test 8: Verify that canonical feature ordering is preserved in both subsets."""
        X_train, X_test, _, _ = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        self.assertEqual(
            list(X_train.columns),
            CANONICAL_FEATURES,
            "X_train column order must exactly match canonical feature order.",
        )
        self.assertEqual(
            list(X_test.columns),
            CANONICAL_FEATURES,
            "X_test column order must exactly match canonical feature order.",
        )

    def test_09_input_preservation(self) -> None:
        """Test 9: Verify that original X and y are not modified by the splitting function."""
        X_copy = self.X.copy(deep=True)
        y_copy = self.y.copy(deep=True)

        _ = stratified_train_test_split(self.X, self.y, test_size=0.20, random_state=42)

        pd.testing.assert_frame_equal(self.X, X_copy, check_exact=True)
        pd.testing.assert_series_equal(self.y, y_copy, check_exact=True)

    def test_10_missing_feature_values_preserved(self) -> None:
        """Test 10: Verify that missing feature values are untouched and conserved across subsets."""
        X_train, X_test, _, _ = stratified_train_test_split(
            self.X, self.y, test_size=0.20, random_state=42
        )
        for feature in ["ph", "Sulfate", "Trihalomethanes"]:
            orig_nulls = int(self.X[feature].isnull().sum())
            split_nulls = int(X_train[feature].isnull().sum()) + int(X_test[feature].isnull().sum())
            self.assertEqual(
                orig_nulls,
                split_nulls,
                f"Missing value conservation failed for feature '{feature}'. Expected {orig_nulls}, got {split_nulls}.",
            )

        self.assertEqual(int(X_train["ph"].isnull().sum()) + int(X_test["ph"].isnull().sum()), 491)
        self.assertEqual(int(X_train["Sulfate"].isnull().sum()) + int(X_test["Sulfate"].isnull().sum()), 781)
        self.assertEqual(int(X_train["Trihalomethanes"].isnull().sum()) + int(X_test["Trihalomethanes"].isnull().sum()), 162)

    def test_11_invalid_arguments_and_error_handling(self) -> None:
        """Test 11: Verify proper exceptions for invalid split parameters, dimensions, and data structures."""
        # 1. Invalid test_size boundaries
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, self.y, test_size=0.0)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, self.y, test_size=1.0)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, self.y, test_size=-0.2)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, self.y, test_size=1.5)

        # 2. Invalid test_size types
        with self.assertRaises(TypeError):
            stratified_train_test_split(self.X, self.y, test_size="0.2")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            stratified_train_test_split(self.X, self.y, test_size=True)  # type: ignore[arg-type]

        # 3. Length mismatch between X and y
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X.iloc[:-5], self.y, test_size=0.20)

        # 4. Index mismatch between X and y
        y_mismatched_index = self.y.copy()
        y_mismatched_index.index = range(100, 100 + len(self.y))
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, y_mismatched_index, test_size=0.20)

        # 5. Missing values in target y
        y_with_nan = self.y.copy()
        y_with_nan.iloc[5] = np.nan
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, y_with_nan, test_size=0.20)

        # 6. Single class target
        y_single_class = pd.Series([0] * len(self.X), index=self.X.index)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, y_single_class, test_size=0.20)

        # 7. Class with insufficient samples (< 2 samples)
        y_insufficient = pd.Series([0] * (len(self.X) - 1) + [1], index=self.X.index)
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, y_insufficient, test_size=0.20)

        # 8. Empty DataFrame
        empty_X = pd.DataFrame(columns=CANONICAL_FEATURES)
        empty_y = pd.Series([], dtype=int)
        with self.assertRaises(ValueError):
            stratified_train_test_split(empty_X, empty_y, test_size=0.20)

        # 9. Invalid random_state types / values
        with self.assertRaises(TypeError):
            stratified_train_test_split(self.X, self.y, random_state="invalid_seed")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.X, self.y, random_state=-10)

    def test_12_split_summary_and_helper(self) -> None:
        """Test 12: Verify get_split_summary and load_and_split_data helper functions."""
        X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.20, random_state=42)
        summary = get_split_summary(X_train, X_test, y_train, y_test, random_state=42, test_size=0.20)

        self.assertEqual(summary["total_samples"], 3276)
        self.assertEqual(summary["train_samples"], 2620)
        self.assertEqual(summary["test_samples"], 656)
        self.assertEqual(summary["overlap_count"], 0)
        self.assertTrue(summary["complete_coverage"])
        self.assertTrue(summary["feature_target_aligned"])
        self.assertEqual(summary["train_class_counts"][0], 1598)
        self.assertEqual(summary["train_class_counts"][1], 1022)
        self.assertEqual(summary["test_class_counts"][0], 400)
        self.assertEqual(summary["test_class_counts"][1], 256)


if __name__ == "__main__":
    unittest.main()

