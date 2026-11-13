"""Automated Unit Tests for Manual Feature Scaling (Z-Score Standardization).

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

Requirements:
- Tests all required scenarios from Phase 2.4 specification.
- Uses Python's built-in unittest module only (no pytest, no scikit-learn).
- Includes strict data-leakage prevention validation and zero-variance tests.
- Tests both synthetic isolated cases and the actual approved dataset with both imputation strategies.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.data_split import load_and_split_data
from backend.ml.eda import CANONICAL_FEATURES
from backend.ml.imputation import ManualImputer
from backend.ml.scaling import ManualStandardScaler, run_scaling_pipeline


class TestManualStandardScaler(unittest.TestCase):
    """Unit test suite for custom ManualStandardScaler and standardization pipeline."""

    def test_01_correct_mean_calculation(self) -> None:
        """Test 1: Verify correct sample mean calculation across features."""
        df = pd.DataFrame({
            "A": [10.0, 20.0, 30.0],
            "B": [1.0, 2.0, 6.0],
        })
        scaler = ManualStandardScaler().fit(df)
        self.assertAlmostEqual(scaler.mean_["A"], 20.0)
        self.assertAlmostEqual(scaler.mean_["B"], 3.0)

    def test_02_correct_population_std_calculation(self) -> None:
        """Test 2: Verify population standard deviation (ddof=0) calculation."""
        # For values [10, 20, 30]:
        # mean = 20
        # variance (ddof=0) = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = 200 / 3 = 66.6666...
        # std (ddof=0) = sqrt(200 / 3) = 8.1649658...
        df = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
        scaler = ManualStandardScaler().fit(df)

        expected_var = 200.0 / 3.0
        expected_std = np.sqrt(expected_var)

        self.assertAlmostEqual(scaler.var_["A"], expected_var)
        self.assertAlmostEqual(scaler.scale_["A"], expected_std)

    def test_03_correct_z_score_calculation(self) -> None:
        """Test 3: Verify accurate Z-score transformation: z = (x - mu) / sigma."""
        df = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
        scaler = ManualStandardScaler().fit(df)
        df_scaled = scaler.transform(df)

        mu = 20.0
        sigma = np.sqrt(200.0 / 3.0)

        expected_z = [(10.0 - mu) / sigma, (20.0 - mu) / sigma, (30.0 - mu) / sigma]

        np.testing.assert_allclose(df_scaled["A"].to_numpy(), expected_z, rtol=1e-10)

    def test_04_fit_transform_behavior(self) -> None:
        """Test 4: Verify fit_transform produces identical results to fit followed by transform."""
        df = pd.DataFrame({
            "A": [2.0, 4.0, 8.0, 16.0],
            "B": [100.0, 200.0, 300.0, 400.0],
        })
        scaler1 = ManualStandardScaler()
        res_fit_transform = scaler1.fit_transform(df)

        scaler2 = ManualStandardScaler()
        scaler2.fit(df)
        res_transform = scaler2.transform(df)

        pd.testing.assert_frame_equal(res_fit_transform, res_transform)
        self.assertEqual(scaler1.mean_, scaler2.mean_)
        self.assertEqual(scaler1.scale_, scaler2.scale_)

    def test_05_transform_separate_test_data(self) -> None:
        """Test 5: Verify transform applies training mu and sigma to unseen test data."""
        X_train = pd.DataFrame({"A": [10.0, 20.0, 30.0]})  # mu=20, sigma=8.1649658
        X_test = pd.DataFrame({"A": [5.0, 20.0, 50.0]})

        scaler = ManualStandardScaler().fit(X_train)
        X_test_scaled = scaler.transform(X_test)

        mu_train = scaler.mean_["A"]
        sigma_train = scaler.scale_["A"]

        expected_test_z = [(5.0 - mu_train) / sigma_train, 0.0, (50.0 - mu_train) / sigma_train]

        np.testing.assert_allclose(X_test_scaled["A"].to_numpy(), expected_test_z, rtol=1e-10)

    def test_06_feature_names_order_and_index_preservation(self) -> None:
        """Test 6: Verify feature names, column order, and custom indices are strictly preserved."""
        custom_idx = [105, 205, 305]
        cols = ["col_z", "col_a", "col_m"]
        df = pd.DataFrame({
            "col_z": [1.0, 2.0, 3.0],
            "col_a": [10.0, 20.0, 30.0],
            "col_m": [100.0, 200.0, 300.0],
        }, index=custom_idx, columns=cols)

        scaler = ManualStandardScaler().fit(df)
        df_scaled = scaler.transform(df)

        self.assertEqual(list(df_scaled.index), custom_idx)
        self.assertEqual(list(df_scaled.columns), cols)
        self.assertEqual(scaler.feature_names_in_, cols)

    def test_07_zero_variance_feature_handling(self) -> None:
        """Test 7: Verify safe zero-variance feature handling without division by zero."""
        df = pd.DataFrame({
            "constant": [5.0, 5.0, 5.0, 5.0],
            "normal": [1.0, 2.0, 3.0, 4.0],
        })
        scaler = ManualStandardScaler().fit(df)

        self.assertEqual(scaler.scale_["constant"], 0.0)
        self.assertEqual(scaler.var_["constant"], 0.0)

        # Transform must not raise ZeroDivisionError and must map constant feature to 0.0
        df_scaled = scaler.transform(df)
        np.testing.assert_array_equal(df_scaled["constant"].to_numpy(), np.zeros(4))

        # Check separate test data with constant feature
        df_test = pd.DataFrame({
            "constant": [5.0, 99.0],
            "normal": [2.0, 5.0],
        })
        df_test_scaled = scaler.transform(df_test)
        np.testing.assert_array_equal(df_test_scaled["constant"].to_numpy(), np.zeros(2))

    def test_08_missing_value_rejection(self) -> None:
        """Test 8: Verify that missing values (NaN) are strictly rejected with ValueError."""
        df_nan = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        scaler = ManualStandardScaler()

        with self.assertRaises(ValueError) as ctx_fit:
            scaler.fit(df_nan)
        self.assertIn("imputation", str(ctx_fit.exception).lower())

        df_clean = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        scaler.fit(df_clean)

        with self.assertRaises(ValueError) as ctx_trans:
            scaler.transform(df_nan)
        self.assertIn("imputation", str(ctx_trans.exception).lower())

    def test_09_infinite_value_rejection(self) -> None:
        """Test 9: Verify that infinite values (+inf, -inf) are strictly rejected with ValueError."""
        df_pos_inf = pd.DataFrame({"A": [1.0, np.inf, 3.0]})
        df_neg_inf = pd.DataFrame({"A": [1.0, -np.inf, 3.0]})
        scaler = ManualStandardScaler()

        with self.assertRaises(ValueError):
            scaler.fit(df_pos_inf)
        with self.assertRaises(ValueError):
            scaler.fit(df_neg_inf)

        df_clean = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        scaler.fit(df_clean)

        with self.assertRaises(ValueError):
            scaler.transform(df_pos_inf)
        with self.assertRaises(ValueError):
            scaler.transform(df_neg_inf)

    def test_10_non_numeric_input_rejection(self) -> None:
        """Test 10: Verify non-numeric column types raise TypeError."""
        df_str = pd.DataFrame({"A": [1.0, "two", 3.0]})
        scaler = ManualStandardScaler()

        with self.assertRaises(TypeError):
            scaler.fit(df_str)

    def test_11_mismatched_feature_names_or_order(self) -> None:
        """Test 11: Verify ValueError is raised when transform features mismatch fit features."""
        df_train = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        scaler = ManualStandardScaler().fit(df_train)

        # Mismatched order
        df_test_reordered = pd.DataFrame({"B": [3.0, 4.0], "A": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            scaler.transform(df_test_reordered)

        # Missing feature
        df_test_missing = pd.DataFrame({"A": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            scaler.transform(df_test_missing)

        # Extra feature
        df_test_extra = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})
        with self.assertRaises(ValueError):
            scaler.transform(df_test_extra)

    def test_12_empty_input_handling(self) -> None:
        """Test 12: Verify ValueError is raised for empty DataFrames."""
        empty_df = pd.DataFrame()
        scaler = ManualStandardScaler()

        with self.assertRaises(ValueError):
            scaler.fit(empty_df)

    def test_13_input_dataframe_unmodified(self) -> None:
        """Test 13: Verify input DataFrames are not mutated in-place by fit or transform."""
        df = pd.DataFrame({"A": [1.0, 5.0, 10.0], "B": [20.0, 40.0, 80.0]})
        df_copy = df.copy(deep=True)

        scaler = ManualStandardScaler()
        scaler.fit(df)
        _ = scaler.transform(df)

        pd.testing.assert_frame_equal(df, df_copy, check_exact=True)

    def test_14_transform_before_fit_error(self) -> None:
        """Test 14: Verify RuntimeError is raised when transform is called before fit."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        scaler = ManualStandardScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform(df)

    def test_15_refitting_behavior(self) -> None:
        """Test 15: Verify refitting updates scaler parameters cleanly."""
        df1 = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        scaler = ManualStandardScaler().fit(df1)
        self.assertAlmostEqual(scaler.mean_["A"], 2.0)

        df2 = pd.DataFrame({"A": [10.0, 20.0, 30.0]})
        scaler.fit(df2)
        self.assertAlmostEqual(scaler.mean_["A"], 20.0)

    def test_16_inverse_transform(self) -> None:
        """Test 16: Verify inverse_transform recovers original data accurately."""
        df = pd.DataFrame({
            "A": [1.23, 4.56, 7.89],
            "B": [100.5, 200.75, 300.25],
            "C": [5.0, 5.0, 5.0],  # zero-variance
        })
        scaler = ManualStandardScaler().fit(df)
        df_scaled = scaler.transform(df)
        df_recovered = scaler.inverse_transform(df_scaled)

        pd.testing.assert_frame_equal(df, df_recovered)

    def test_17_leakage_prevention(self) -> None:
        """Test 17: Strict leakage prevention test demonstrating independence from test data."""
        X_train = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})
        X_test_normal = pd.DataFrame({"feat": [15.0, 25.0]})
        X_test_extreme = pd.DataFrame({"feat": [1_000_000.0, 5_000_000.0]})

        scaler = ManualStandardScaler().fit(X_train)
        original_mean = scaler.mean_["feat"]
        original_std = scaler.scale_["feat"]

        # 1. Transform normal test set
        _ = scaler.transform(X_test_normal)
        self.assertEqual(scaler.mean_["feat"], original_mean)
        self.assertEqual(scaler.scale_["feat"], original_std)

        # 2. Transform extreme test set
        res_extreme = scaler.transform(X_test_extreme)
        # Scaler parameters must remain strictly unaffected by extreme test values
        self.assertEqual(scaler.mean_["feat"], original_mean)
        self.assertEqual(scaler.scale_["feat"], original_std)

        # Verify test data is transformed using training statistics
        expected_extreme_z0 = (1_000_000.0 - original_mean) / original_std
        self.assertAlmostEqual(res_extreme.loc[0, "feat"], expected_extreme_z0)

    def test_18_compatibility_with_mean_imputation(self) -> None:
        """Test 18: Full pipeline test with Mean Imputation on actual approved dataset."""
        results = run_scaling_pipeline(imputation_strategy="mean", test_size=0.20, random_state=42)
        v = results["verification"]

        self.assertEqual(v["n_train"], 2620)
        self.assertEqual(v["n_test"], 656)
        self.assertEqual(v["train_nulls"], 0)
        self.assertEqual(v["test_nulls"], 0)
        self.assertEqual(v["train_infs"], 0)
        self.assertEqual(v["test_infs"], 0)
        self.assertTrue(v["indices_preserved"])
        self.assertTrue(v["columns_preserved"])
        self.assertTrue(v["train_means_near_zero"])
        self.assertTrue(v["train_stds_near_one"])

    def test_19_compatibility_with_median_imputation(self) -> None:
        """Test 19: Full pipeline test with Median Imputation on actual approved dataset."""
        results = run_scaling_pipeline(imputation_strategy="median", test_size=0.20, random_state=42)
        v = results["verification"]

        self.assertEqual(v["n_train"], 2620)
        self.assertEqual(v["n_test"], 656)
        self.assertEqual(v["train_nulls"], 0)
        self.assertEqual(v["test_nulls"], 0)
        self.assertEqual(v["train_infs"], 0)
        self.assertEqual(v["test_infs"], 0)
        self.assertTrue(v["indices_preserved"])
        self.assertTrue(v["columns_preserved"])
        self.assertTrue(v["train_means_near_zero"])
        self.assertTrue(v["train_stds_near_one"])


if __name__ == "__main__":
    unittest.main()

