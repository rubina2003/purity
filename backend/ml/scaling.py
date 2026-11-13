"""Manual Feature Scaling Module (Z-Score Standardization) for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere.
- Pure custom Z-score standardization implemented using NumPy and Pandas.
- Formula: z = (x - mu) / sigma
- Uses population standard deviation (ddof=0) consistently.
- Prevents data leakage by fitting exclusively on training partitions (X_train).
- Applies fitted training statistics (mu, sigma) to testing partitions (X_test) and future inference.
- Rejects missing values (NaN) rather than silently filling them: imputation must precede scaling.
- Safely handles zero-variance features by mapping them to 0.0 without division by zero.
- Preserves original DataFrame row indices, column names, and canonical feature ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.data_split import load_and_split_data
from backend.ml.eda import CANONICAL_FEATURES
from backend.ml.imputation import ManualImputer


class ManualStandardScaler:
    """Custom StandardScaler implementing Z-score standardization without Scikit-learn.

    Standardizes features by removing the mean and scaling to unit variance:
        z = (x - mu) / sigma

    Attributes:
        mean_: Dictionary mapping feature column names to fitted training sample means.
        scale_: Dictionary mapping feature column names to fitted training population standard deviations (ddof=0).
        var_: Dictionary mapping feature column names to fitted training population variances (ddof=0).
        feature_names_in_: List of feature names seen during fit().
        n_features_in_: Number of features seen during fit().
        n_samples_seen_: Number of training samples processed during fit().
        is_fitted_: Boolean indicating whether the scaler has been fitted.
    """

    def __init__(self, copy: bool = True) -> None:
        """Initialize the manual standard scaler.

        Args:
            copy: Whether to perform transformations on deep copies of DataFrames.
        """
        self.copy: bool = copy
        self.mean_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}
        self.var_: Dict[str, float] = {}
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0
        self.n_samples_seen_: int = 0
        self.is_fitted_: bool = False

    def _validate_input_frame(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Validate input DataFrame structure, types, absence of NaNs, and absence of infs.

        Args:
            X: Input DataFrame of features.
            is_fit: Boolean indicating whether validation is for fit() or transform().

        Returns:
            Validated DataFrame.

        Raises:
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If X is empty, has duplicate columns, missing values, or infinite values.
            RuntimeError: If transform() is invoked before fit().
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        if X.empty:
            raise ValueError("Input DataFrame is empty (contains 0 records).")

        # Check for duplicate column names
        if len(X.columns) != len(set(X.columns)):
            duplicates = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(f"Input DataFrame contains duplicate column names: {duplicates}")

        # Check for non-numeric data types
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise TypeError(f"Column '{col}' must have numeric dtype, got '{X[col].dtype}'.")

        # Enforce strict imputation-before-scaling rule: reject missing values
        if X.isnull().any().any():
            null_summary = {col: int(cnt) for col, cnt in X.isnull().sum().items() if cnt > 0}
            raise ValueError(
                f"Input contains missing (NaN) values: {null_summary}. "
                f"Missing-value imputation must be performed before feature scaling."
            )

        # Check for infinite values (+inf or -inf)
        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("Input DataFrame contains infinite values (+inf or -inf).")

        if is_fit:
            # Check canonical ordering if the canonical features are present
            if set(CANONICAL_FEATURES).issubset(set(X.columns)):
                actual_canonical = [c for c in X.columns if c in CANONICAL_FEATURES]
                if actual_canonical != CANONICAL_FEATURES:
                    raise ValueError(
                        f"Feature columns do not follow canonical ordering.\n"
                        f"Expected: {CANONICAL_FEATURES}\n"
                        f"Actual:   {actual_canonical}"
                    )
        else:
            # For transform: ensure scaler is already fitted
            if not self.is_fitted_:
                raise RuntimeError("ManualStandardScaler must be fitted before calling transform().")

            # Ensure columns and ordering match exactly what was seen during fit()
            actual_cols = list(X.columns)
            if actual_cols != self.feature_names_in_:
                raise ValueError(
                    f"Feature mismatch between fit and transform.\n"
                    f"Expected: {self.feature_names_in_}\n"
                    f"Actual:   {actual_cols}"
                )

        return X

    def fit(self, X: pd.DataFrame) -> ManualStandardScaler:
        """Fit the scaler by computing population mean and standard deviation for each feature.

        Uses ddof=0 for population standard deviation and variance consistently.
        If called on an already fitted scaler, previous parameters are overwritten with new calculations.

        Args:
            X: Training feature DataFrame (X_train).

        Returns:
            Fitted scaler instance (self).
        """
        X_validated = self._validate_input_frame(X, is_fit=True)

        self.mean_ = {}
        self.scale_ = {}
        self.var_ = {}
        self.feature_names_in_ = list(X_validated.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_samples_seen_ = len(X_validated)

        for col in self.feature_names_in_:
            col_vals = X_validated[col].to_numpy(dtype=float)

            mean_val = float(np.mean(col_vals))
            var_val = float(np.var(col_vals, ddof=0))
            scale_val = float(np.std(col_vals, ddof=0))

            self.mean_[col] = mean_val
            self.var_[col] = var_val
            self.scale_[col] = scale_val

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize X by centering with stored training means and scaling with stored standard deviations.

        Standardization formula:
            z = (x - mu_train) / sigma_train

        Zero-variance handling:
            If a feature has zero variance in the training data (sigma_train == 0.0 or sigma_train < 1e-12),
            it is safely mapped to 0.0 to prevent division by zero.

        Does NOT recalculate or adapt statistics based on X.
        Preserves original row indices, column names, and column ordering.

        Args:
            X: Feature DataFrame to standardize (X_train, X_test, or future inference samples).

        Returns:
            Standardized DataFrame with float64 values.
        """
        X_validated = self._validate_input_frame(X, is_fit=False)

        # Work on a defensive deep copy
        X_scaled = X_validated.copy(deep=True).astype(float)

        for col in self.feature_names_in_:
            scale = self.scale_[col]
            mean = self.mean_[col]

            # Safe zero-variance handling: map constant features to zeros
            if scale == 0.0 or scale < 1e-12:
                X_scaled[col] = 0.0
            else:
                X_scaled[col] = (X_scaled[col].to_numpy(dtype=float) - mean) / scale

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler using X and transform X in a single step.

        Args:
            X: Training feature DataFrame (X_train).

        Returns:
            Standardized DataFrame.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Revert standardized features back to original measurement scales.

        Formula:
            x = z * sigma_train + mu_train

        Args:
            X: Standardized DataFrame.

        Returns:
            Reconstructed DataFrame on original measurement scales.
        """
        X_validated = self._validate_input_frame(X, is_fit=False)
        X_orig = X_validated.copy(deep=True).astype(float)

        for col in self.feature_names_in_:
            scale = self.scale_[col]
            mean = self.mean_[col]

            if scale == 0.0 or scale < 1e-12:
                X_orig[col] = mean
            else:
                X_orig[col] = (X_orig[col].to_numpy(dtype=float) * scale) + mean

        return X_orig


# -----------------------------------------------------------------------------
# Preprocessing Pipeline Demonstration & Evaluation
# -----------------------------------------------------------------------------

def run_scaling_pipeline(
    imputation_strategy: str = "median",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end split, imputation, and standardization pipeline in-memory.

    Args:
        imputation_strategy: Imputation strategy ('mean' or 'median'). Defaults to 'median'.
        test_size: Test partition proportion. Defaults to 0.20.
        random_state: Seed for reproducible stratified split. Defaults to 42.

    Returns:
        Dictionary containing processed DataFrames, fitted transformers, and verification metrics.
    """
    # 1. Obtain stratified split
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=test_size, random_state=random_state)

    # 2. Impute missing values (fit exclusively on X_train)
    imputer = ManualImputer(strategy=imputation_strategy)
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 3. Fit scaler exclusively on imputed X_train
    scaler = ManualStandardScaler()
    scaler.fit(X_train_imp)

    # 4. Transform both training and testing partitions using fitted training statistics
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # 5. Compute verification statistics
    train_means_after = X_train_scaled.mean(axis=0)
    train_stds_after = X_train_scaled.std(axis=0, ddof=0)
    test_means_after = X_test_scaled.mean(axis=0)
    test_stds_after = X_test_scaled.std(axis=0, ddof=0)

    verification = {
        "strategy": imputation_strategy,
        "n_train": len(X_train_scaled),
        "n_test": len(X_test_scaled),
        "train_nulls": int(X_train_scaled.isnull().sum().sum()),
        "test_nulls": int(X_test_scaled.isnull().sum().sum()),
        "train_infs": int(np.isinf(X_train_scaled.to_numpy()).sum()),
        "test_infs": int(np.isinf(X_test_scaled.to_numpy()).sum()),
        "indices_preserved": (
            X_train.index.equals(X_train_scaled.index)
            and X_test.index.equals(X_test_scaled.index)
            and y_train.index.equals(X_train_scaled.index)
            and y_test.index.equals(X_test_scaled.index)
        ),
        "columns_preserved": (
            list(X_train_scaled.columns) == CANONICAL_FEATURES
            and list(X_test_scaled.columns) == CANONICAL_FEATURES
        ),
        "train_means_near_zero": bool(np.allclose(train_means_after, 0.0, atol=1e-12)),
        "train_stds_near_one": bool(np.allclose(train_stds_after, 1.0, atol=1e-12)),
        "max_train_mean_abs": float(np.max(np.abs(train_means_after))),
        "max_train_std_diff": float(np.max(np.abs(train_stds_after - 1.0))),
    }

    return {
        "imputer": imputer,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "train_means_after": train_means_after,
        "train_stds_after": train_stds_after,
        "test_means_after": test_means_after,
        "test_stds_after": test_stds_after,
        "verification": verification,
    }


def print_scaling_demonstration(results_mean: Dict[str, Any], results_median: Dict[str, Any]) -> None:
    """Print an ASCII demonstration report comparing standardization across both imputation strategies."""
    print("=" * 86)
    print(" PURITY - PHASE 2.4: MANUAL FEATURE SCALING (Z-SCORE STANDARDIZATION)")
    print(" Academic Context: BCA 8th Sem, Tribhuvan University - Project III")
    print(" Formula: z = (x - mu_train) / sigma_train  |  Convention: Population Std (ddof=0)")
    print("=" * 86)

    for res in (results_mean, results_median):
        strat = res["verification"]["strategy"]
        scaler: ManualStandardScaler = res["scaler"]
        print(f"\n[{strat.upper()} IMPUTATION + Z-SCORE STANDARDIZATION PIPELINE]")
        print("-" * 86)
        print(
            f"{'Feature':<18} | {'Fitted Mean (mu)':<16} | {'Fitted Std (sigma)':<18} | "
            f"{'Train Mean [z]':<14} | {'Train Std [z]':<13}"
        )
        print("-" * 86)

        for feat in CANONICAL_FEATURES:
            mu = scaler.mean_[feat]
            sig = scaler.scale_[feat]
            tr_m = res["train_means_after"][feat]
            tr_s = res["train_stds_after"][feat]
            print(f"{feat:<18} | {mu:<16.4f} | {sig:<18.4f} | {tr_m:<14.2e} | {tr_s:<13.6f}")

        print("-" * 86)
        print(f"{'Feature':<18} | {'Test Mean [z]':<16} | {'Test Std [z]':<18} | {'Status':<28}")
        print("-" * 86)

        for feat in CANONICAL_FEATURES:
            te_m = res["test_means_after"][feat]
            te_s = res["test_stds_after"][feat]
            print(f"{feat:<18} | {te_m:<16.4f} | {te_s:<18.4f} | Scaled via training mu/sigma")

        v = res["verification"]
        print("-" * 86)
        print(f"Sample Count:                 Train = {v['n_train']:,} | Test = {v['n_test']:,}")
        print(f"Missing Values (NaN):         Train = {v['train_nulls']} | Test = {v['test_nulls']}")
        print(f"Infinite Values (±inf):       Train = {v['train_infs']} | Test = {v['test_infs']}")
        print(f"Training Mean Near Zero:      {'VERIFIED (max |mu| < 1e-12)' if v['train_means_near_zero'] else 'FAILED'}")
        print(f"Training Std Near One:        {'VERIFIED (max |sigma-1| < 1e-12)' if v['train_stds_near_one'] else 'FAILED'}")
        print(f"Indices & Order Preserved:    {'VERIFIED' if v['indices_preserved'] and v['columns_preserved'] else 'FAILED'}")

    print("=" * 86)


def main() -> int:
    """CLI execution entrypoint for testing and verification."""
    print("[1/3] Running scaling pipeline with Mean Imputation...")
    results_mean = run_scaling_pipeline(imputation_strategy="mean", test_size=0.20, random_state=42)

    print("[2/3] Running scaling pipeline with Median Imputation...")
    results_median = run_scaling_pipeline(imputation_strategy="median", test_size=0.20, random_state=42)

    print("[3/3] Generating standardization report...\n")
    print_scaling_demonstration(results_mean, results_median)

    # Verification assertions
    for res in (results_mean, results_median):
        v = res["verification"]
        assert v["train_nulls"] == 0, "Missing values found in standardized training data."
        assert v["test_nulls"] == 0, "Missing values found in standardized testing data."
        assert v["train_infs"] == 0, "Infinite values found in standardized training data."
        assert v["test_infs"] == 0, "Infinite values found in standardized testing data."
        assert v["train_means_near_zero"] is True, "Standardized training means are not approximately zero."
        assert v["train_stds_near_one"] is True, "Standardized training standard deviations are not approximately one."
        assert v["indices_preserved"] is True, "Original row indices were not preserved."
        assert v["columns_preserved"] is True, "Canonical feature ordering was altered."

    print("\n[SUCCESS] Manual feature scaling verified against all criteria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

