"""Automated Unit Tests for Manual Missing-Value Imputation Module.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

Requirements:
- Tests all 17+ required scenarios from Phase 2.3 specification.
- Uses Python's built-in unittest module only (no pytest, no scikit-learn).
- Includes strict data-leakage prevention validation.
- Tests both synthetic isolated cases and the actual approved dataset.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.data_split import load_and_split_data
from backend.ml.eda import CANONICAL_FEATURES
from backend.ml.imputation import (
    ManualImputer,
    compare_split_imputation,
    evaluate_imputation_strategies,
)


class TestManualImputer(unittest.TestCase):
    """Unit test suite for custom ManualImputer and evaluation utilities."""

    def test_01_correct_mean_calculation(self) -> None:
        """Test 1: Verify correct sample mean calculation ignoring NaNs."""
        df = pd.DataFrame({"A": [10.0, 20.0, np.nan, 30.0]})
        imputer = ManualImputer(strategy="mean")
        imputer.fit(df)
        expected_mean = (10.0 + 20.0 + 30.0) / 3.0  # 20.0
        self.assertAlmostEqual(imputer.statistics_["A"], expected_mean)

    def test_02_correct_median_calculation(self) -> None:
        """Test 2: Verify correct sample median calculation ignoring NaNs."""
        df = pd.DataFrame({"A": [5.0, 1.0, np.nan, 9.0, 3.0]})
        imputer = ManualImputer(strategy="median")
        imputer.fit(df)
        # Sorted observed: [1.0, 3.0, 5.0, 9.0] -> mid is 4 items -> (3.0 + 5.0) / 2 = 4.0
        self.assertEqual(imputer.statistics_["A"], 4.0)

    def test_03_even_and_odd_median_handling(self) -> None:
        """Test 3: Verify median calculation for both odd and even counts of observed values."""
        # Odd: 3 values [10, 20, 30] -> median 20
        df_odd = pd.DataFrame({"X": [30.0, np.nan, 10.0, 20.0]})
        imputer_odd = ManualImputer(strategy="median").fit(df_odd)
        self.assertEqual(imputer_odd.statistics_["X"], 20.0)

        # Even: 4 values [10, 20, 30, 40] -> median 25
        df_even = pd.DataFrame({"X": [40.0, 10.0, np.nan, 20.0, 30.0]})
        imputer_even = ManualImputer(strategy="median").fit(df_even)
        self.assertEqual(imputer_even.statistics_["X"], 25.0)

    def test_04_correct_treatment_of_nan_values(self) -> None:
        """Test 4: Verify NaNs are ignored during statistic computation and filled during transform."""
        df = pd.DataFrame({"feat": [np.nan, 2.0, 4.0, np.nan, 6.0]})
        imputer = ManualImputer(strategy="mean").fit(df)
        transformed = imputer.transform(df)

        self.assertEqual(transformed["feat"].isnull().sum(), 0)
        self.assertEqual(transformed.loc[0, "feat"], 4.0)
        self.assertEqual(transformed.loc[3, "feat"], 4.0)
        self.assertEqual(transformed.loc[1, "feat"], 2.0)
        self.assertEqual(transformed.loc[2, "feat"], 4.0)
        self.assertEqual(transformed.loc[4, "feat"], 6.0)

    def test_05_training_only_fitted_statistics(self) -> None:
        """Test 5: Verify that fitted statistics depend exclusively on training data."""
        X_train = pd.DataFrame({"val": [10.0, 20.0, np.nan]})
        X_test = pd.DataFrame({"val": [1000.0, np.nan]})

        imputer = ManualImputer(strategy="mean").fit(X_train)
        self.assertEqual(imputer.statistics_["val"], 15.0)

        # Transform test data; statistics_ must remain unchanged
        _ = imputer.transform(X_test)
        self.assertEqual(imputer.statistics_["val"], 15.0)

    def test_06_testing_data_transformed_with_training_statistics(self) -> None:
        """Test 6: Verify testing data NaNs are replaced using training statistics."""
        X_train = pd.DataFrame({"val": [2.0, 4.0]})
        X_test = pd.DataFrame({"val": [np.nan, 10.0]})

        imputer = ManualImputer(strategy="mean").fit(X_train)
        X_test_imputed = imputer.transform(X_test)

        self.assertEqual(X_test_imputed.loc[0, "val"], 3.0)  # Mean of train: (2+4)/2 = 3.0
        self.assertEqual(X_test_imputed.loc[1, "val"], 10.0)

    def test_07_original_data_preservation(self) -> None:
        """Test 7: Verify that the input DataFrame is not mutated during fit or transform."""
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        df_copy = df.copy(deep=True)

        imputer = ManualImputer(strategy="median")
        imputer.fit(df)
        _ = imputer.transform(df)

        pd.testing.assert_frame_equal(df, df_copy, check_exact=True)

    def test_08_index_preservation(self) -> None:
        """Test 8: Verify that custom non-sequential row indices are preserved during transform."""
        custom_indices = [102, 505, 909]
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0]}, index=custom_indices)

        imputer = ManualImputer(strategy="mean").fit(df)
        df_imputed = imputer.transform(df)

        self.assertEqual(list(df_imputed.index), custom_indices)

    def test_09_feature_order_preservation(self) -> None:
        """Test 9: Verify that column ordering is strictly maintained."""
        cols = ["c", "a", "b"]
        df = pd.DataFrame({"c": [1.0, 2.0], "a": [np.nan, 3.0], "b": [4.0, 5.0]}, columns=cols)

        imputer = ManualImputer(strategy="median").fit(df)
        df_imputed = imputer.transform(df)

        self.assertEqual(list(df_imputed.columns), cols)

    def test_10_no_missing_values_remaining(self) -> None:
        """Test 10: Verify zero NaNs remain after transform."""
        df = pd.DataFrame({
            "A": [np.nan, 1.0, np.nan],
            "B": [2.0, np.nan, 4.0],
        })
        imputer = ManualImputer(strategy="mean")
        df_imputed = imputer.fit_transform(df)

        self.assertEqual(df_imputed.isnull().sum().sum(), 0)

    def test_11_unsupported_strategy_error(self) -> None:
        """Test 11: Verify ValueError is raised for unsupported strategy names."""
        with self.assertRaises(ValueError):
            ManualImputer(strategy="mode")
        with self.assertRaises(ValueError):
            ManualImputer(strategy="constant")
        with self.assertRaises(TypeError):
            ManualImputer(strategy=123)  # type: ignore[arg-type]

    def test_12_all_missing_feature_error(self) -> None:
        """Test 12: Verify ValueError is raised if a feature in training data is entirely NaN."""
        df_all_nan = pd.DataFrame({"A": [np.nan, np.nan, np.nan]})
        imputer = ManualImputer(strategy="mean")
        with self.assertRaises(ValueError):
            imputer.fit(df_all_nan)

    def test_13_transform_before_fit_error(self) -> None:
        """Test 13: Verify RuntimeError is raised if transform is invoked before fit."""
        df = pd.DataFrame({"A": [1.0, 2.0]})
        imputer = ManualImputer(strategy="mean")
        with self.assertRaises(RuntimeError):
            imputer.transform(df)

    def test_14_infinite_value_rejection(self) -> None:
        """Test 14: Verify ValueError is raised if infinite values are present in input."""
        df_inf = pd.DataFrame({"A": [1.0, np.inf, 3.0]})
        imputer = ManualImputer(strategy="mean")
        with self.assertRaises(ValueError):
            imputer.fit(df_inf)

        df_valid = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        imputer.fit(df_valid)
        with self.assertRaises(ValueError):
            imputer.transform(df_inf)

    def test_15_feature_target_separation(self) -> None:
        """Test 15: Verify imputer works strictly on features, independent of target labels."""
        X = pd.DataFrame({"feat1": [1.0, np.nan, 3.0]})
        y = pd.Series([0, 1, 0])

        imputer = ManualImputer(strategy="median")
        imputer.fit(X)

        # Confirm target is completely unreferenced by imputer
        self.assertNotIn("Potability", imputer.statistics_)
        self.assertEqual(len(imputer.statistics_), 1)

    def test_16_masked_value_validation_reproducibility(self) -> None:
        """Test 16: Verify masked-value validation produces deterministic results with fixed seed."""
        df_train = pd.DataFrame({
            "ph": [7.0, 7.2, 6.8, 7.5, 6.5, 7.1, 7.3, 6.9, 7.4, 7.0] * 10,
            "Sulfate": [300.0, 320.0, 310.0, 330.0, 305.0, 315.0, 325.0, 308.0, 322.0, 318.0] * 10,
            "Trihalomethanes": [60.0, 65.0, 62.0, 68.0, 58.0, 63.0, 66.0, 61.0, 64.0, 67.0] * 10,
        })
        res1 = evaluate_imputation_strategies(df_train, random_state=42)
        res2 = evaluate_imputation_strategies(df_train, random_state=42)

        pd.testing.assert_frame_equal(res1, res2)

    def test_17_equal_masking_positions_for_strategies(self) -> None:
        """Test 17: Verify Mean and Median evaluate against the exact same masked positions."""
        df_train = pd.DataFrame({
            "ph": [7.0, 7.2, 6.8, 7.5, 6.5, 7.1, 7.3, 6.9, 7.4, 7.0] * 10,
            "Sulfate": [300.0, 320.0, 310.0, 330.0, 305.0, 315.0, 325.0, 308.0, 322.0, 318.0] * 10,
            "Trihalomethanes": [60.0, 65.0, 62.0, 68.0, 58.0, 63.0, 66.0, 61.0, 64.0, 67.0] * 10,
        })
        res = evaluate_imputation_strategies(df_train, random_state=42)
        for feat in ["ph", "Sulfate", "Trihalomethanes"]:
            feat_rows = res[res["Feature"] == feat]
            self.assertEqual(len(feat_rows), 2)
            # Masked counts must be identical between Mean and Median
            self.assertEqual(feat_rows.iloc[0]["Masked_Count"], feat_rows.iloc[1]["Masked_Count"])

    def test_18_data_leakage_prevention(self) -> None:
        """Test 18: Specific leakage prevention test ensuring test values never affect imputation."""
        # Fit median on training values [1, 3, NaN]
        X_train = pd.DataFrame({"A": [1.0, 3.0, np.nan]})
        # Test values [100, NaN]
        X_test = pd.DataFrame({"A": [100.0, np.nan]})

        imputer = ManualImputer(strategy="median")
        imputer.fit(X_train)

        # Expected training median of [1.0, 3.0] is 2.0
        self.assertEqual(imputer.statistics_["A"], 2.0)

        X_test_imputed = imputer.transform(X_test)
        # Missing value in X_test must be replaced with 2.0, NOT affected by 100.0
        self.assertEqual(X_test_imputed.loc[1, "A"], 2.0)
        self.assertEqual(X_test_imputed.loc[0, "A"], 100.0)

    def test_19_full_workflow_on_approved_dataset(self) -> None:
        """Test 19: Full integration test on approved Kaggle Water Potability dataset."""
        X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.20, random_state=42)

        results = compare_split_imputation(X_train, X_test)
        v = results["verification"]

        self.assertEqual(v["train_samples"], 2620)
        self.assertEqual(v["test_samples"], 656)
        self.assertEqual(v["train_nulls_before"], 1135)
        self.assertEqual(v["test_nulls_before"], 299)
        self.assertEqual(v["mean_train_nulls_after"], 0)
        self.assertEqual(v["mean_test_nulls_after"], 0)
        self.assertEqual(v["median_train_nulls_after"], 0)
        self.assertEqual(v["median_test_nulls_after"], 0)
        self.assertTrue(v["indices_preserved"])
        self.assertTrue(v["columns_preserved"])
        self.assertTrue(v["observed_values_preserved"])

        # Check that target series are completely untouched
        self.assertEqual(len(y_train), 2620)
        self.assertEqual(len(y_test), 656)


if __name__ == "__main__":
    unittest.main()

