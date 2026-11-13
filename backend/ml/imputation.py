"""Manual Missing-Value Imputation Module for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere.
- Pure custom imputation implemented using Python, NumPy, and Pandas.
- Implements both Mean and Median imputation strategies.
- Prevents data leakage by fitting exclusively on training partitions (X_train).
- Applies fitted statistics to testing partitions (X_test) and future inference.
- Preserves original DataFrame row indices, canonical feature order, and observed values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.data_split import load_and_split_data
from backend.ml.eda import CANONICAL_FEATURES


class ManualImputer:
    """Custom missing-value imputer supporting Mean and Median strategies without Scikit-learn.

    Attributes:
        strategy: Imputation strategy, either 'mean' or 'median'.
        statistics_: Dictionary mapping feature column names to fitted statistics.
        feature_names_in_: List of feature names seen during fit().
        is_fitted_: Boolean indicating whether the imputer has been fitted.
    """

    SUPPORTED_STRATEGIES = ("mean", "median")

    def __init__(self, strategy: str = "median") -> None:
        """Initialize the imputer with the chosen strategy.

        Args:
            strategy: Imputation strategy ('mean' or 'median'). Defaults to 'median'.

        Raises:
            ValueError: If strategy is not 'mean' or 'median'.
        """
        if not isinstance(strategy, str):
            raise TypeError(f"strategy must be a string, got {type(strategy).__name__}.")

        normalized_strategy = strategy.strip().lower()
        if normalized_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy '{strategy}'. Supported strategies: {self.SUPPORTED_STRATEGIES}."
            )

        self.strategy: str = normalized_strategy
        self.statistics_: Dict[str, float] = {}
        self.feature_names_in_: List[str] = []
        self.is_fitted_: bool = False

    def _validate_input_frame(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Validate input DataFrame schema, types, infinite values, and non-emptiness.

        Args:
            X: Input features DataFrame.
            is_fit: Boolean indicating if validation is for fit() or transform().

        Returns:
            Validated DataFrame.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            ValueError: If schema, column ordering, or values are invalid.
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

        # Check for infinite values
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

            # Ensure no feature is completely missing (all NaN) in training data
            for col in X.columns:
                if X[col].isnull().all():
                    raise ValueError(
                        f"Column '{col}' contains no observed training values (all NaN). "
                        f"Cannot compute {self.strategy} statistic."
                    )
        else:
            # For transform: ensure imputer is already fitted
            if not self.is_fitted_:
                raise RuntimeError("ManualImputer must be fitted before calling transform().")

            # Ensure columns and ordering match exactly what was seen during fit()
            actual_cols = list(X.columns)
            if actual_cols != self.feature_names_in_:
                raise ValueError(
                    f"Feature mismatch between fit and transform.\n"
                    f"Expected: {self.feature_names_in_}\n"
                    f"Actual:   {actual_cols}"
                )

        return X

    @staticmethod
    def _compute_mean(values: np.ndarray) -> float:
        """Calculate manual sample mean over observed (non-NaN) values.

        Formula: mu = (1 / n) * sum(x_i)
        """
        clean_vals = values[~np.isnan(values)]
        n = len(clean_vals)
        if n == 0:
            raise ValueError("Cannot compute mean of empty array.")
        return float(np.sum(clean_vals) / n)

    @staticmethod
    def _compute_median(values: np.ndarray) -> float:
        """Calculate manual sample median over observed (non-NaN) values.

        Steps:
        1. Filter out NaNs.
        2. Sort values.
        3. If odd: middle element.
        4. If even: average of two middle elements.
        """
        clean_vals = values[~np.isnan(values)]
        n = len(clean_vals)
        if n == 0:
            raise ValueError("Cannot compute median of empty array.")

        sorted_vals = np.sort(clean_vals)
        mid = n // 2
        if n % 2 == 1:
            median_val = sorted_vals[mid]
        else:
            median_val = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return float(median_val)

    def fit(self, X: pd.DataFrame) -> ManualImputer:
        """Fit the imputer by computing training-only statistics for each feature.

        Args:
            X: Training feature DataFrame (X_train).

        Returns:
            Fitted imputer instance (self).
        """
        X_validated = self._validate_input_frame(X, is_fit=True)

        self.statistics_ = {}
        self.feature_names_in_ = list(X_validated.columns)

        for col in self.feature_names_in_:
            col_array = X_validated[col].to_numpy(dtype=float)
            if self.strategy == "mean":
                stat = self._compute_mean(col_array)
            elif self.strategy == "median":
                stat = self._compute_median(col_array)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.statistics_[col] = stat

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values in X using the stored training statistics.

        Does NOT recompute or alter statistics based on X.
        Preserves original row indices, column ordering, and observed values.

        Args:
            X: Feature DataFrame to transform (X_train, X_test, or new samples).

        Returns:
            New DataFrame with missing values replaced by fitted statistics.
        """
        X_validated = self._validate_input_frame(X, is_fit=False)

        # Work on a defensive copy to prevent mutating the original input
        X_imputed = X_validated.copy(deep=True)

        for col in self.feature_names_in_:
            fill_value = self.statistics_[col]
            X_imputed[col] = X_imputed[col].fillna(fill_value)

        return X_imputed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit to X, then transform X.

        Args:
            X: Training feature DataFrame (X_train).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X).transform(X)


# -----------------------------------------------------------------------------
# Task 4: Diagnostic Masked-Value Comparison (Training Data Only)
# -----------------------------------------------------------------------------

def evaluate_imputation_strategies(
    X_train: pd.DataFrame,
    features: Optional[List[str]] = None,
    mask_fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate Mean vs. Median imputation using masked-value validation on training data.

    Protocol:
    1. Restricted strictly to training data (zero data leakage from test partition).
    2. For each feature with natural missing values, extract observed values.
    3. Randomly select mask_fraction (~10%) of observed values using fixed seed.
    4. Compute mean and median using remaining (1 - mask_fraction) values.
    5. Evaluate both statistics against actual masked values using MAE and RMSE.
    6. Identical masked positions are used for both strategies.

    Args:
        X_train: Training feature DataFrame.
        features: Features to evaluate. Defaults to ['ph', 'Sulfate', 'Trihalomethanes'].
        mask_fraction: Fraction of observed values to mask for diagnostic (default 0.10).
        random_state: Seed for reproducible masking selection. Defaults to 42.

    Returns:
        DataFrame containing evaluation results with columns:
        ['Feature', 'Strategy', 'Masked_Count', 'MAE', 'RMSE']
    """
    if features is None:
        features = ["ph", "Sulfate", "Trihalomethanes"]

    rng = np.random.default_rng(random_state)
    records: List[Dict[str, Any]] = []

    for feat in features:
        if feat not in X_train.columns:
            raise ValueError(f"Feature '{feat}' not found in X_train columns.")

        # Extract only observed (non-NaN) values from training subset
        observed = X_train[feat].dropna().to_numpy(dtype=float)
        n_obs = len(observed)
        n_mask = int(np.round(n_obs * mask_fraction))
        if n_mask < 1:
            continue

        # Shuffle positions and partition into masked and remaining
        shuffled_indices = rng.permutation(n_obs)
        mask_indices = shuffled_indices[:n_mask]
        remaining_indices = shuffled_indices[n_mask:]

        actual_masked = observed[mask_indices]
        remaining_vals = observed[remaining_indices]

        # Calculate Mean and Median from the remaining observed values
        mean_estimate = float(np.mean(remaining_vals))
        median_estimate = float(np.median(remaining_vals))

        # Calculate Mean Absolute Error (MAE)
        mae_mean = float(np.mean(np.abs(actual_masked - mean_estimate)))
        mae_median = float(np.mean(np.abs(actual_masked - median_estimate)))

        # Calculate Root Mean Squared Error (RMSE)
        rmse_mean = float(np.sqrt(np.mean((actual_masked - mean_estimate) ** 2)))
        rmse_median = float(np.sqrt(np.mean((actual_masked - median_estimate) ** 2)))

        records.append({
            "Feature": feat,
            "Strategy": "Mean",
            "Masked_Count": n_mask,
            "MAE": round(mae_mean, 4),
            "RMSE": round(rmse_mean, 4),
        })
        records.append({
            "Feature": feat,
            "Strategy": "Median",
            "Masked_Count": n_mask,
            "MAE": round(mae_median, 4),
            "RMSE": round(rmse_median, 4),
        })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Task 5: Actual Split Imputation Runner & Verification
# -----------------------------------------------------------------------------

def compare_split_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, Any]:
    """Fit both Mean and Median imputers on X_train and transform both subsets.

    Returns in-memory imputed subsets without writing to disk or modifying inputs.

    Args:
        X_train: Training features DataFrame.
        X_test: Testing features DataFrame.

    Returns:
        Dictionary containing fitted imputers, transformed DataFrames, and integrity metrics.
    """
    # 1. Mean Imputation
    imputer_mean = ManualImputer(strategy="mean")
    imputer_mean.fit(X_train)
    X_train_mean = imputer_mean.transform(X_train)
    X_test_mean = imputer_mean.transform(X_test)

    # 2. Median Imputation
    imputer_median = ManualImputer(strategy="median")
    imputer_median.fit(X_train)
    X_train_median = imputer_median.transform(X_train)
    X_test_median = imputer_median.transform(X_test)

    # Integrity verification
    verification = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_nulls_before": int(X_train.isnull().sum().sum()),
        "test_nulls_before": int(X_test.isnull().sum().sum()),
        "mean_train_nulls_after": int(X_train_mean.isnull().sum().sum()),
        "mean_test_nulls_after": int(X_test_mean.isnull().sum().sum()),
        "median_train_nulls_after": int(X_train_median.isnull().sum().sum()),
        "median_test_nulls_after": int(X_test_median.isnull().sum().sum()),
        "indices_preserved": (
            X_train.index.equals(X_train_mean.index)
            and X_train.index.equals(X_train_median.index)
            and X_test.index.equals(X_test_mean.index)
            and X_test.index.equals(X_test_median.index)
        ),
        "columns_preserved": (
            list(X_train_mean.columns) == CANONICAL_FEATURES
            and list(X_test_mean.columns) == CANONICAL_FEATURES
            and list(X_train_median.columns) == CANONICAL_FEATURES
            and list(X_test_median.columns) == CANONICAL_FEATURES
        ),
        "observed_values_preserved": True,
    }

    # Verify that previously observed values were not altered
    for col in CANONICAL_FEATURES:
        train_obs_mask = ~X_train[col].isnull()
        test_obs_mask = ~X_test[col].isnull()

        if not np.allclose(X_train.loc[train_obs_mask, col], X_train_mean.loc[train_obs_mask, col]):
            verification["observed_values_preserved"] = False
        if not np.allclose(X_train.loc[train_obs_mask, col], X_train_median.loc[train_obs_mask, col]):
            verification["observed_values_preserved"] = False
        if not np.allclose(X_test.loc[test_obs_mask, col], X_test_mean.loc[test_obs_mask, col]):
            verification["observed_values_preserved"] = False
        if not np.allclose(X_test.loc[test_obs_mask, col], X_test_median.loc[test_obs_mask, col]):
            verification["observed_values_preserved"] = False

    return {
        "imputer_mean": imputer_mean,
        "imputer_median": imputer_median,
        "X_train_mean": X_train_mean,
        "X_test_mean": X_test_mean,
        "X_train_median": X_train_median,
        "X_test_median": X_test_median,
        "verification": verification,
    }


def print_imputation_summary(
    imputer_mean: ManualImputer,
    imputer_median: ManualImputer,
    eval_df: pd.DataFrame,
    verification: Dict[str, Any],
) -> None:
    """Print an informative ASCII summary of imputation statistics and comparison."""
    print("=" * 78)
    print(" PURITY - PHASE 2.3: MANUAL MISSING-VALUE IMPUTATION SUMMARY")
    print(" Academic Context: BCA 8th Sem, Tribhuvan University - Project III")
    print("=" * 78)
    print(f"Training Samples: {verification['train_samples']:,} | Testing Samples: {verification['test_samples']:,}")
    print(f"Missing Values Before Imputation: Train = {verification['train_nulls_before']:,}, Test = {verification['test_nulls_before']:,}")
    print(f"Missing Values After Mean Imputation: Train = {verification['mean_train_nulls_after']}, Test = {verification['mean_test_nulls_after']}")
    print(f"Missing Values After Median Imputation: Train = {verification['median_train_nulls_after']}, Test = {verification['median_test_nulls_after']}")
    print("-" * 78)
    print("FITTED TRAINING-ONLY IMPUTATION STATISTICS (Zero Data Leakage):")
    print(f"{'Feature':<20} | {'Training Mean':<20} | {'Training Median':<20}")
    print("-" * 78)

    for feat in CANONICAL_FEATURES:
        m_val = imputer_mean.statistics_[feat]
        med_val = imputer_median.statistics_[feat]
        marker = " (Imputed)" if feat in ("ph", "Sulfate", "Trihalomethanes") else " (Complete)"
        print(f"{feat + marker:<20} | {m_val:<20.4f} | {med_val:<20.4f}")

    print("-" * 78)
    print("DIAGNOSTIC MASKED-VALUE VALIDATION (10% Random Mask on Training Data):")
    print(f"{'Feature':<18} | {'Strategy':<10} | {'Masked N':<10} | {'MAE':<12} | {'RMSE':<12}")
    print("-" * 78)
    for _, row in eval_df.iterrows():
        print(
            f"{row['Feature']:<18} | {row['Strategy']:<10} | {row['Masked_Count']:<10} | "
            f"{row['MAE']:<12.4f} | {row['RMSE']:<12.4f}"
        )

    print("-" * 78)
    print(f"Row Indices Preserved:        {'VERIFIED' if verification['indices_preserved'] else 'FAILED'}")
    print(f"Canonical Feature Order:      {'VERIFIED' if verification['columns_preserved'] else 'FAILED'}")
    print(f"Observed Values Preserved:    {'VERIFIED' if verification['observed_values_preserved'] else 'FAILED'}")
    print("=" * 78)


def main() -> int:
    """CLI execution entrypoint for testing and verification."""
    print("[1/4] Loading approved stratified train-test split via backend.ml.data_split...")
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.20, random_state=42)

    print("[2/4] Running diagnostic masked-value comparison on training partition...")
    eval_df = evaluate_imputation_strategies(X_train, mask_fraction=0.10, random_state=42)

    print("[3/4] Fitting Mean and Median imputers on X_train and transforming subsets...")
    split_results = compare_split_imputation(X_train, X_test)

    print("[4/4] Outputting imputation verification report...\n")
    print_imputation_summary(
        imputer_mean=split_results["imputer_mean"],
        imputer_median=split_results["imputer_median"],
        eval_df=eval_df,
        verification=split_results["verification"],
    )

    # Verification assertions
    v = split_results["verification"]
    assert v["mean_train_nulls_after"] == 0, "Missing values remain in mean-imputed training data."
    assert v["mean_test_nulls_after"] == 0, "Missing values remain in mean-imputed testing data."
    assert v["median_train_nulls_after"] == 0, "Missing values remain in median-imputed training data."
    assert v["median_test_nulls_after"] == 0, "Missing values remain in median-imputed testing data."
    assert v["indices_preserved"] is True, "Original row indices were not preserved."
    assert v["columns_preserved"] is True, "Canonical feature ordering was altered."
    assert v["observed_values_preserved"] is True, "Observed feature values were modified."

    print("\n[SUCCESS] Custom missing-value imputation verified against all criteria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

