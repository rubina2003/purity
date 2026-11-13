"""Manual Stratified Train-Test Splitting Module for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere.
- Pure custom stratified splitting implemented using NumPy random number generation.
- Preserves exact canonical feature order and original DataFrame row indices.
- Preserves missing feature values without modification (for Phase 2.3 imputation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.eda import (
    CANONICAL_FEATURES,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    load_dataset,
)


def _validate_split_inputs(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, List[Any]],
    test_size: float,
    random_state: Union[int, np.random.Generator, None],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate input data structures, indices, target labels, and split configuration.

    Args:
        X: Feature matrix DataFrame.
        y: Target series or 1D array.
        test_size: Proportion of dataset allocated to test partition.
        random_state: Seed or NumPy Generator for reproducible shuffling.

    Returns:
        Tuple of validated (X, y) as (pd.DataFrame, pd.Series).

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If input dimensions, indices, target values, or test_size are invalid.
    """
    # 1. Type validation for X
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}.")

    if X.empty:
        raise ValueError("Feature matrix X cannot be empty (contains 0 records).")

    # 2. Type and structure validation for y
    if isinstance(y, (np.ndarray, list)):
        y_series = pd.Series(y, index=X.index, name=TARGET_COLUMN)
    elif isinstance(y, pd.Series):
        y_series = y
    else:
        raise TypeError(f"y must be a pandas Series or 1D array, got {type(y).__name__}.")

    # 3. Dimensionality and Index Alignment
    if len(X) != len(y_series):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows, but y has {len(y_series)} records."
        )

    if not X.index.equals(y_series.index):
        raise ValueError(
            "Index mismatch: X and y must have identical row indices and ordering."
        )

    # 4. Canonical Feature Order Check (when columns match canonical set)
    if set(CANONICAL_FEATURES).issubset(set(X.columns)):
        actual_canonical = [c for c in X.columns if c in CANONICAL_FEATURES]
        if actual_canonical != CANONICAL_FEATURES:
            raise ValueError(
                f"Feature columns do not follow canonical ordering.\n"
                f"Expected: {CANONICAL_FEATURES}\n"
                f"Actual:   {actual_canonical}"
            )

    # 5. Target Integrity (Target must not contain NaNs)
    if y_series.isnull().any():
        null_count = int(y_series.isnull().sum())
        raise ValueError(f"Target y contains {null_count} missing (NaN) values.")

    # 6. Target Labels Validation
    unique_classes = np.unique(y_series)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Target y must contain at least 2 distinct classes, found {len(unique_classes)}: {unique_classes}."
        )

    # 7. Test Size Validation
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size must be a float between 0 and 1, got {type(test_size).__name__}.")

    test_size_float = float(test_size)
    if not (0.0 < test_size_float < 1.0):
        raise ValueError(f"test_size must be strictly between 0 and 1 (exclusive), got {test_size_float}.")

    # 8. Random State Validation
    if random_state is not None:
        if not isinstance(random_state, (int, np.integer, np.random.Generator)):
            raise TypeError(
                f"random_state must be an int, None, or numpy.random.Generator, got {type(random_state).__name__}."
            )
        if isinstance(random_state, (int, np.integer)) and random_state < 0:
            raise ValueError(f"random_state integer must be non-negative, got {random_state}.")

    # 9. Class Sample Capacity Check
    for cls in unique_classes:
        count_cls = int(np.sum(y_series == cls))
        if count_cls < 2:
            raise ValueError(
                f"Class '{cls}' has only {count_cls} sample(s). Each class must contain at least 2 "
                f"samples to allow stratified partitioning into training and testing sets."
            )

    return X, y_series


def stratified_train_test_split(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, List[Any]],
    test_size: float = 0.20,
    random_state: Union[int, np.random.Generator, None] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a custom, reproducible stratified train-test split without Scikit-learn.

    The algorithm:
    1. Validates inputs, indices, canonical feature order, and target labels.
    2. Identifies unique target classes in y.
    3. For each class:
       a. Extracts row positional indices belonging to that class.
       b. Shuffles class positions using NumPy's seeded default_rng generator.
       c. Calculates class-specific testing count using standard rounding:
          n_test = int(round(n_class * test_size)), bounded to [1, n_class - 1].
       d. Allocates the first n_test positions to test, remainder to train.
    4. Combines class-specific training positions and testing positions.
    5. Shuffles combined training and testing positions using the same generator.
    6. Constructs final subsets via .iloc[], preserving original DataFrame indices,
       canonical column ordering, and any un-imputed missing feature values.

    Args:
        X: Feature matrix DataFrame.
        y: Target series or 1D array.
        test_size: Proportion of the dataset to allocate to test. Defaults to 0.20.
        random_state: Seed or NumPy Generator for reproducibility. Defaults to 42.

    Returns:
        Tuple containing (X_train, X_test, y_train, y_test).
    """
    X_valid, y_valid = _validate_split_inputs(X, y, test_size, random_state)

    # Initialize NumPy random number generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # 1. Identify unique target classes (sorted for deterministic iteration)
    unique_classes = np.sort(np.unique(y_valid))

    train_positions_list: List[int] = []
    test_positions_list: List[int] = []

    y_numpy = y_valid.to_numpy()

    # 2. Stratified per-class partitioning
    for cls in unique_classes:
        # Extract row positional indices for the current class
        cls_positions = np.where(y_numpy == cls)[0]
        n_cls = len(cls_positions)

        # Shuffle positions within this class
        shuffled_cls_positions = rng.permutation(cls_positions)

        # Calculate testing samples for this class using standard mathematical rounding
        # For Potability dataset with test_size=0.20:
        #   Class 0: 1,998 * 0.20 = 399.6 -> 400 test, 1,598 train
        #   Class 1: 1,278 * 0.20 = 255.6 -> 256 test, 1,022 train
        n_test = int(np.round(n_cls * float(test_size)))
        # Boundary protection: guarantee at least 1 sample in both train and test
        n_test = max(1, min(n_cls - 1, n_test))

        test_positions_list.extend(shuffled_cls_positions[:n_test])
        train_positions_list.extend(shuffled_cls_positions[n_test:])

    # 3. Shuffle combined positions so subsets are not ordered by class
    final_train_positions = rng.permutation(train_positions_list)
    final_test_positions = rng.permutation(test_positions_list)

    # 4. Construct final training and testing subsets while preserving original row indices
    X_train = X_valid.iloc[final_train_positions].copy()
    X_test = X_valid.iloc[final_test_positions].copy()
    y_train = y_valid.iloc[final_train_positions].copy()
    y_test = y_valid.iloc[final_test_positions].copy()

    return X_train, X_test, y_train, y_test


def load_and_split_data(
    filepath: Union[Path, str, None] = None,
    test_size: float = 0.20,
    random_state: Union[int, np.random.Generator, None] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Helper function to load the verified dataset and perform stratified splitting.

    Args:
        filepath: Optional path to CSV file. Defaults to DEFAULT_DATA_PATH.
        test_size: Proportion of test partition. Defaults to 0.20.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = load_dataset(filepath or DEFAULT_DATA_PATH)
    X = df[CANONICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return stratified_train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_split_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: Any = 42,
    test_size: float = 0.20,
) -> Dict[str, Any]:
    """Calculate comprehensive metrics evaluating stratified split integrity and proportions.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training targets.
        y_test: Testing targets.
        random_state: Seed utilized for the split.
        test_size: Configured test size proportion.

    Returns:
        Dictionary of summary statistics.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_total = n_train + n_test

    # Class counts
    y_total_series = pd.concat([y_train, y_test])
    classes = sorted(y_total_series.unique())

    orig_counts = {int(c): int((y_total_series == c).sum()) for c in classes}
    train_counts = {int(c): int((y_train == c).sum()) for c in classes}
    test_counts = {int(c): int((y_test == c).sum()) for c in classes}

    orig_pcts = {int(c): round((orig_counts[c] / n_total) * 100.0, 2) for c in classes}
    train_pcts = {int(c): round((train_counts[c] / n_train) * 100.0, 2) for c in classes}
    test_pcts = {int(c): round((test_counts[c] / n_test) * 100.0, 2) for c in classes}

    # Index integrity
    train_idx_set = set(X_train.index)
    test_idx_set = set(X_test.index)
    overlap_count = len(train_idx_set & test_idx_set)
    unique_count = len(train_idx_set | test_idx_set)

    # Feature-target alignment
    train_aligned = X_train.index.equals(y_train.index)
    test_aligned = X_test.index.equals(y_test.index)

    return {
        "total_samples": n_total,
        "train_samples": n_train,
        "test_samples": n_test,
        "actual_train_pct": round((n_train / n_total) * 100.0, 2),
        "actual_test_pct": round((n_test / n_total) * 100.0, 2),
        "configured_test_size": test_size,
        "random_state": random_state,
        "train_test_ratio": round(n_train / n_test, 3) if n_test > 0 else float("inf"),
        "original_class_counts": orig_counts,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "original_class_pcts": orig_pcts,
        "train_class_pcts": train_pcts,
        "test_class_pcts": test_pcts,
        "overlap_count": overlap_count,
        "complete_coverage": unique_count == n_total,
        "feature_target_aligned": train_aligned and test_aligned,
    }


def print_split_summary(summary: Dict[str, Any]) -> None:
    """Print an informative ASCII summary of the stratified split results."""
    print("=" * 72)
    print(" PURITY - PHASE 2.2: STRATIFIED TRAIN-TEST SPLIT SUMMARY")
    print(" Academic Context: BCA 8th Sem, Tribhuvan University - Project III")
    print("=" * 72)
    print(f"Total Dataset Size:    {summary['total_samples']:,} samples")
    print(f"Training Subset:       {summary['train_samples']:,} samples ({summary['actual_train_pct']:.2f}%)")
    print(f"Testing Subset:        {summary['test_samples']:,} samples ({summary['actual_test_pct']:.2f}%)")
    print(f"Configured Test Size:  {summary['configured_test_size']:.2f} (80:20 target)")
    print(f"Random Seed:           {summary['random_state']}")
    print(f"Train/Test Ratio:      {summary['train_test_ratio']:.3f} : 1")
    print("-" * 72)
    print(f"{'Class':<20} | {'Original':<15} | {'Training':<15} | {'Testing':<15}")
    print("-" * 72)

    for cls in summary["original_class_counts"]:
        name = "0 (Non-potable)" if cls == 0 else "1 (Potable)"
        orig = f"{summary['original_class_counts'][cls]:,} ({summary['original_class_pcts'][cls]:.2f}%)"
        train = f"{summary['train_class_counts'][cls]:,} ({summary['train_class_pcts'][cls]:.2f}%)"
        test = f"{summary['test_class_counts'][cls]:,} ({summary['test_class_pcts'][cls]:.2f}%)"
        print(f"{name:<20} | {orig:<15} | {train:<15} | {test:<15}")

    print("-" * 72)
    print(f"Index Disjointness (Overlap): {summary['overlap_count']} rows (No data leakage)")
    print(f"Record Coverage (100%):       {'VERIFIED' if summary['complete_coverage'] else 'FAILED'}")
    print(f"Feature-Target Alignment:     {'VERIFIED' if summary['feature_target_aligned'] else 'FAILED'}")
    print("=" * 72)


def main() -> int:
    """CLI execution entrypoint for testing and verification."""
    print("[1/3] Loading approved dataset via backend.ml.eda...")
    df = load_dataset()
    X = df[CANONICAL_FEATURES]
    y = df[TARGET_COLUMN]

    print("[2/3] Executing manual stratified train-test split (80:20, seed=42)...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("[3/3] Generating stratification verification summary...\n")
    summary = get_split_summary(X_train, X_test, y_train, y_test, random_state=42, test_size=0.20)
    print_split_summary(summary)

    # Verification checks
    assert summary["train_samples"] == 2620, f"Expected 2620 train samples, got {summary['train_samples']}"
    assert summary["test_samples"] == 656, f"Expected 656 test samples, got {summary['test_samples']}"
    assert summary["train_class_counts"][0] == 1598, f"Expected 1598 non-potable in train, got {summary['train_class_counts'][0]}"
    assert summary["train_class_counts"][1] == 1022, f"Expected 1022 potable in train, got {summary['train_class_counts'][1]}"
    assert summary["test_class_counts"][0] == 400, f"Expected 400 non-potable in test, got {summary['test_class_counts'][0]}"
    assert summary["test_class_counts"][1] == 256, f"Expected 256 potable in test, got {summary['test_class_counts'][1]}"
    assert summary["overlap_count"] == 0, f"Index overlap detected: {summary['overlap_count']}"
    assert summary["complete_coverage"] is True, "Complete record coverage check failed."
    assert summary["feature_target_aligned"] is True, "Feature-target index alignment failed."

    print("\n[SUCCESS] Stratified split integrity verified against all criteria.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

