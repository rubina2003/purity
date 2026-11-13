"""Exploratory Data Analysis (EDA) Module for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- No Scikit-learn (sklearn) is imported or used anywhere.
- No Seaborn is imported or used.
- Permitted libraries: Python Standard Library, NumPy, Pandas, Matplotlib.
- Pure dataset exploration only: no preprocessing, no imputation, no scaling,
  and no model training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
# Use non-interactive Agg backend for headless / script execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Canonical Project Configuration
# -----------------------------------------------------------------------------

CANONICAL_FEATURES: List[str] = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

TARGET_COLUMN: str = "Potability"

EXPECTED_COLUMNS: List[str] = CANONICAL_FEATURES + [TARGET_COLUMN]

# Project paths resolved relative to this source file
MODULE_PATH = Path(__file__).resolve()
PROJECT_ROOT = MODULE_PATH.parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "water_potability.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"


# -----------------------------------------------------------------------------
# Task 2: Dataset Loading and Validation
# -----------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset structure, schema, column order, data types, and target values.

    Args:
        df: Loaded pandas DataFrame.

    Raises:
        ValueError: If validation checks fail.
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise ValueError("Dataset is empty (contains 0 rows).")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")

    # Check presence of all expected columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    unexpected_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if unexpected_cols:
        raise ValueError(f"Dataset contains unexpected extra columns: {unexpected_cols}")

    # Validate column ordering
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Dataset column order does not match canonical order.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    # Validate feature data types (must be numeric float or int)
    for col in CANONICAL_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature column '{col}' is not numeric (type: {df[col].dtype}).")

    # Validate target column data type and values
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise ValueError(f"Target column '{TARGET_COLUMN}' is not numeric.")

    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    valid_targets = {0, 1}
    if not unique_targets.issubset(valid_targets):
        raise ValueError(
            f"Target column contains invalid labels: {unique_targets - valid_targets}. "
            f"Expected only {valid_targets}."
        )


def load_dataset(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load the water potability dataset with comprehensive validation.

    Path is resolved relative to the project root unless an explicit path is provided.
    The original dataset file is treated as read-only.

    Args:
        filepath: Optional path to CSV file. Defaults to DEFAULT_DATA_PATH.

    Returns:
        Validated pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        RuntimeError: If dataset loading or validation fails.
    """
    target_path = Path(filepath) if filepath else DEFAULT_DATA_PATH

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {target_path.resolve()}\n"
            f"Please verify that 'data/water_potability.csv' exists in the project root."
        )

    try:
        df = pd.read_csv(target_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV dataset from {target_path}: {exc}") from exc

    validate_dataset(df)
    return df


# -----------------------------------------------------------------------------
# Task 3: Dataset Overview
# -----------------------------------------------------------------------------

def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level overview metrics of the dataset."""
    total_rows, total_cols = df.shape
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_str = f"{memory_bytes / 1024:.2f} KB"

    numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numerical_cols = [c for c in df.columns if c not in numerical_cols]

    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": list(df.columns),
        "data_types": dtypes_dict,
        "memory_usage_bytes": memory_bytes,
        "memory_usage_str": memory_str,
        "numerical_cols_count": len(numerical_cols),
        "non_numerical_cols_count": len(non_numerical_cols),
    }


# -----------------------------------------------------------------------------
# Task 4: Missing-Value Analysis
# -----------------------------------------------------------------------------

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform detailed missing-value analysis across all columns."""
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows) * 100.0

    column_missing: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        cnt = int(missing_counts[col])
        pct = float(missing_pcts[col])
        column_missing[col] = {
            "missing_count": cnt,
            "missing_percentage": round(pct, 2),
            "has_missing": cnt > 0,
        }

    total_missing_values = int(missing_counts.sum())
    complete_records = int(df.dropna().shape[0])
    incomplete_records = int(df.isnull().any(axis=1).sum())
    complete_pct = round((complete_records / total_rows) * 100.0, 2)
    incomplete_pct = round((incomplete_records / total_rows) * 100.0, 2)

    return {
        "column_missing": column_missing,
        "total_missing_values": total_missing_values,
        "complete_records": complete_records,
        "complete_records_percentage": complete_pct,
        "incomplete_records": incomplete_records,
        "incomplete_records_percentage": incomplete_pct,
    }


# -----------------------------------------------------------------------------
# Task 5: Target-Class Distribution Analysis
# -----------------------------------------------------------------------------

def analyze_target_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the binary Potability target variable distribution and class balance."""
    target_series = df[TARGET_COLUMN]
    total_samples = len(target_series)

    counts = target_series.value_counts().to_dict()
    non_potable_count = int(counts.get(0, 0))
    potable_count = int(counts.get(1, 0))

    non_potable_pct = round((non_potable_count / total_samples) * 100.0, 2)
    potable_pct = round((potable_count / total_samples) * 100.0, 2)

    # Ratio of majority (0) to minority (1)
    imbalance_ratio = (
        round(non_potable_count / potable_count, 3) if potable_count > 0 else float("inf")
    )
    difference_pct = round(abs(non_potable_pct - potable_pct), 2)

    return {
        "total_samples": total_samples,
        "non_potable_count": non_potable_count,
        "non_potable_pct": non_potable_pct,
        "potable_count": potable_count,
        "potable_pct": potable_pct,
        "majority_class": 0,
        "minority_class": 1,
        "imbalance_ratio": imbalance_ratio,
        "difference_pct": difference_pct,
    }


# -----------------------------------------------------------------------------
# Task 6: Individual Feature Statistics & Distributions
# -----------------------------------------------------------------------------

def analyze_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute parametric and non-parametric descriptive statistics for all 9 features."""
    stats: Dict[str, Dict[str, float]] = {}

    for feature in CANONICAL_FEATURES:
        series = df[feature].dropna()
        count = int(series.count())
        mean_val = float(series.mean())
        std_val = float(series.std())
        var_val = float(series.var())
        min_val = float(series.min())
        q25_val = float(series.quantile(0.25))
        median_val = float(series.median())
        q75_val = float(series.quantile(0.75))
        max_val = float(series.max())
        skew_val = float(series.skew())

        stats[feature] = {
            "count": count,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "variance": round(var_val, 4),
            "min": round(min_val, 4),
            "q25": round(q25_val, 4),
            "median": round(median_val, 4),
            "q75": round(q75_val, 4),
            "max": round(max_val, 4),
            "skewness": round(skew_val, 4),
        }

    return stats


# -----------------------------------------------------------------------------
# Task 7: Outlier Exploration (Manual IQR Method)
# -----------------------------------------------------------------------------

def analyze_outliers(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Perform outlier detection using the standard Interquartile Range (IQR) method.

    IQR = Q3 - Q1
    Lower Bound = Q1 - 1.5 * IQR
    Upper Bound = Q3 + 1.5 * IQR
    """
    outlier_results: Dict[str, Dict[str, Any]] = {}

    for feature in CANONICAL_FEATURES:
        series = df[feature].dropna()
        n_obs = len(series)

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        low_outliers = int((series < lower_bound).sum())
        high_outliers = int((series > upper_bound).sum())
        total_outliers = low_outliers + high_outliers
        outlier_pct = round((total_outliers / n_obs) * 100.0, 2)

        outlier_results[feature] = {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "low_outliers_count": low_outliers,
            "high_outliers_count": high_outliers,
            "total_outliers_count": total_outliers,
            "outlier_percentage": outlier_pct,
            "min_val": round(float(series.min()), 4),
            "max_val": round(float(series.max()), 4),
            "contains_zero_minimum": bool(series.min() == 0.0),
        }

    return outlier_results


# -----------------------------------------------------------------------------
# Task 8: Feature Correlation Analysis
# -----------------------------------------------------------------------------

def analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Pearson correlation matrix across features and with target."""
    feature_df = df[CANONICAL_FEATURES]
    feature_corr = feature_df.corr(method="pearson")

    # Correlations of features with target
    target_corr = df[CANONICAL_FEATURES + [TARGET_COLUMN]].corr(method="pearson")[TARGET_COLUMN]
    feature_to_target = {
        feat: round(float(target_corr[feat]), 4) for feat in CANONICAL_FEATURES
    }

    # Find strongest and weakest pairwise correlations among features (excluding diagonal)
    pairs = []
    n = len(CANONICAL_FEATURES)
    for i in range(n):
        for j in range(i + 1, n):
            f1 = CANONICAL_FEATURES[i]
            f2 = CANONICAL_FEATURES[j]
            val = float(feature_corr.loc[f1, f2])
            pairs.append((f1, f2, val))

    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    matrix_dict = {
        row: {col: round(float(feature_corr.loc[row, col]), 4) for col in CANONICAL_FEATURES}
        for row in CANONICAL_FEATURES
    }

    return {
        "correlation_matrix": matrix_dict,
        "feature_target_correlation": feature_to_target,
        "strongest_pairs": [
            {"feat1": p[0], "feat2": p[1], "corr": round(p[2], 4), "abs_corr": round(abs(p[2]), 4)}
            for p in pairs_sorted[:5]
        ],
        "weakest_pairs": [
            {"feat1": p[0], "feat2": p[1], "corr": round(p[2], 4), "abs_corr": round(abs(p[2]), 4)}
            for p in pairs_sorted[-5:]
        ],
    }


# -----------------------------------------------------------------------------
# Task 9: Feature Distribution by Potability Class
# -----------------------------------------------------------------------------

def analyze_features_by_potability(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute grouped statistics for each feature conditioned on Potability class."""
    by_class: Dict[str, Dict[str, Any]] = {}

    for feature in CANONICAL_FEATURES:
        s0 = df.loc[df[TARGET_COLUMN] == 0, feature].dropna()
        s1 = df.loc[df[TARGET_COLUMN] == 1, feature].dropna()

        by_class[feature] = {
            "non_potable": {
                "count": int(s0.count()),
                "mean": round(float(s0.mean()), 4),
                "std": round(float(s0.std()), 4),
                "median": round(float(s0.median()), 4),
                "min": round(float(s0.min()), 4),
                "max": round(float(s0.max()), 4),
            },
            "potable": {
                "count": int(s1.count()),
                "mean": round(float(s1.mean()), 4),
                "std": round(float(s1.std()), 4),
                "median": round(float(s1.median()), 4),
                "min": round(float(s1.min()), 4),
                "max": round(float(s1.max()), 4),
            },
            "mean_difference": round(float(s1.mean() - s0.mean()), 4),
            "median_difference": round(float(s1.median() - s0.median()), 4),
        }

    return by_class


# -----------------------------------------------------------------------------
# Visualizations (Matplotlib only, strictly no Seaborn or Scikit-learn)
# -----------------------------------------------------------------------------

def generate_visualizations(
    df: pd.DataFrame,
    missing_data: Dict[str, Any],
    target_data: Dict[str, Any],
    stats_data: Dict[str, Dict[str, float]],
    outlier_data: Dict[str, Dict[str, Any]],
    corr_data: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate and save all 6 required exploratory visualization charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: Dict[str, Path] = {}

    # Setup styling defaults
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # 1. Missing Values Bar Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    col_missing = missing_data["column_missing"]
    feature_names = CANONICAL_FEATURES
    counts = [col_missing[feat]["missing_count"] for feat in feature_names]
    pcts = [col_missing[feat]["missing_percentage"] for feat in feature_names]

    colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in counts]
    bars = ax.bar(feature_names, counts, color=colors, edgecolor="#2c3e50", width=0.6, alpha=0.9)

    ax.set_title("Missing Values Count and Percentage per Feature", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Physicochemical Features", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Missing Records (NaN)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=35, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)

    for bar, cnt, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        label_text = f"{cnt}\n({pct:.1f}%)" if cnt > 0 else "0\n(0.0%)"
        ax.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if cnt > 0 else "normal",
            color="#c0392b" if cnt > 0 else "#27ae60",
        )

    plt.tight_layout()
    missing_path = output_dir / "missing_values.png"
    fig.savefig(missing_path, dpi=300)
    plt.close(fig)
    generated_charts["missing_values"] = missing_path

    # -------------------------------------------------------------------------
    # 2. Target Distribution Chart (Bar + Donut breakdown)
    # -------------------------------------------------------------------------
    fig, (ax_bar, ax_donut) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    classes = ["Non-Potable (0)", "Potable (1)"]
    counts = [target_data["non_potable_count"], target_data["potable_count"]]
    pcts = [target_data["non_potable_pct"], target_data["potable_pct"]]
    palette = ["#3498db", "#2ecc71"]

    # Bar plot
    bars = ax_bar.bar(classes, counts, color=palette, edgecolor="#2c3e50", width=0.5, alpha=0.9)
    ax_bar.set_title("Potability Class Sample Counts", fontsize=12, fontweight="bold", pad=12)
    ax_bar.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, max(counts) * 1.18)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax_bar.annotate(
            f"{cnt:,} samples\n({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Donut plot
    wedges, texts, autotexts = ax_donut.pie(
        counts,
        labels=classes,
        autopct="%1.2f%%",
        startangle=90,
        colors=palette,
        wedgeprops=dict(width=0.4, edgecolor="#2c3e50", linewidth=1.2),
        textprops=dict(fontsize=10, fontweight="bold"),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax_donut.set_title("Potability Class Ratio", fontsize=12, fontweight="bold", pad=12)

    fig.suptitle("Target Variable (Potability) Distribution and Class Balance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    target_path = output_dir / "target_distribution.png"
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_charts["target_distribution"] = target_path

    # -------------------------------------------------------------------------
    # 3. Individual Feature Distributions (3x3 Subplot Histograms)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes_flat = axes.flatten()

    for idx, feature in enumerate(CANONICAL_FEATURES):
        ax = axes_flat[idx]
        data_clean = df[feature].dropna()
        mean_v = stats_data[feature]["mean"]
        median_v = stats_data[feature]["median"]
        skew_v = stats_data[feature]["skewness"]

        ax.hist(
            data_clean,
            bins=35,
            color="#34495e",
            edgecolor="#ffffff",
            alpha=0.75,
            density=False,
        )
        ax.axvline(mean_v, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.2f}")
        ax.axvline(median_v, color="#27ae60", linestyle=":", linewidth=1.5, label=f"Median: {median_v:.2f}")

        ax.set_title(f"{feature} (skew: {skew_v:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Feature Distributions with Mean and Median (Original Scales)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    feat_dist_path = output_dir / "feature_distributions.png"
    fig.savefig(feat_dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_charts["feature_distributions"] = feat_dist_path

    # -------------------------------------------------------------------------
    # 4. Feature Boxplots (3x3 Subplot Grid for Individual Feature Scaling)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes_flat = axes.flatten()

    for idx, feature in enumerate(CANONICAL_FEATURES):
        ax = axes_flat[idx]
        data_clean = df[feature].dropna()
        out_info = outlier_data[feature]

        bp = ax.boxplot(
            data_clean,
            tick_labels=[feature],
            patch_artist=True,
            showmeans=True,
            meanline=True,
            flierprops=dict(marker="o", markerfacecolor="#e74c3c", markeredgecolor="#c0392b", markersize=4, alpha=0.6),
            boxprops=dict(facecolor="#5dade2", edgecolor="#1b4f72", linewidth=1.2),
            medianprops=dict(color="#b03a2e", linewidth=1.8),
            meanprops=dict(color="#196f3d", linewidth=1.5, linestyle="--"),
            whiskerprops=dict(color="#1b4f72", linewidth=1.2),
            capprops=dict(color="#1b4f72", linewidth=1.2),
        )

        ax.set_title(
            f"{feature}\n[Outliers: {out_info['total_outliers_count']} ({out_info['outlier_percentage']}%) | IQR: {out_info['iqr']:.1f}]",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Feature Boxplots and IQR Outlier Detection (Individual Scales)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    box_path = output_dir / "feature_boxplots.png"
    fig.savefig(box_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_charts["feature_boxplots"] = box_path

    # -------------------------------------------------------------------------
    # 5. Pearson Correlation Heatmap (Matplotlib only)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=300)
    corr_matrix_df = pd.DataFrame(corr_data["correlation_matrix"])
    matrix_vals = corr_matrix_df.values

    cax = ax.imshow(matrix_vals, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient (r)", fontsize=11, fontweight="bold")

    n_feats = len(CANONICAL_FEATURES)
    ax.set_xticks(range(n_feats))
    ax.set_yticks(range(n_feats))
    ax.set_xticklabels(CANONICAL_FEATURES, rotation=40, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(CANONICAL_FEATURES, fontsize=10, fontweight="bold")

    # Annotate values inside cells
    for i in range(n_feats):
        for j in range(n_feats):
            val = matrix_vals[i, j]
            # Use white text for dark colored cells, black for lighter cells
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title("Pearson Correlation Heatmap (9 Physicochemical Features)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_path = output_dir / "correlation_heatmap.png"
    fig.savefig(corr_path, dpi=300)
    plt.close(fig)
    generated_charts["correlation_heatmap"] = corr_path

    # -------------------------------------------------------------------------
    # 6. Feature Distribution by Potability (Task 9)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes_flat = axes.flatten()

    for idx, feature in enumerate(CANONICAL_FEATURES):
        ax = axes_flat[idx]
        s0 = df.loc[df[TARGET_COLUMN] == 0, feature].dropna()
        s1 = df.loc[df[TARGET_COLUMN] == 1, feature].dropna()

        ax.hist(s0, bins=30, alpha=0.55, color="#3498db", label="Non-Potable (0)", density=True)
        ax.hist(s1, bins=30, alpha=0.55, color="#e67e22", label="Potable (1)", density=True)

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Feature Distribution Comparison by Potability Class (Normalized Density)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    by_pot_path = output_dir / "feature_distribution_by_potability.png"
    fig.savefig(by_pot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_charts["feature_distribution_by_potability"] = by_pot_path

    return generated_charts


# -----------------------------------------------------------------------------
# Task 10: Markdown EDA Report Generation
# -----------------------------------------------------------------------------

def generate_eda_report(
    overview: Dict[str, Any],
    missing_data: Dict[str, Any],
    target_data: Dict[str, Any],
    stats_data: Dict[str, Dict[str, float]],
    outlier_data: Dict[str, Dict[str, Any]],
    corr_data: Dict[str, Any],
    by_potability: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Generate the structured markdown summary report with all calculated EDA findings."""
    report_path = output_dir / "eda_summary.md"

    # Build descriptive statistics markdown table
    stats_table_rows = []
    for feat in CANONICAL_FEATURES:
        s = stats_data[feat]
        stats_table_rows.append(
            f"| `{feat}` | {s['count']:,} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} "
            f"| {s['q25']:.3f} | {s['median']:.3f} | {s['q75']:.3f} | {s['max']:.3f} | {s['variance']:.3f} |"
        )
    stats_table = "\n".join(stats_table_rows)

    # Build missing values markdown table
    missing_table_rows = []
    col_missing = missing_data["column_missing"]
    for col in EXPECTED_COLUMNS:
        info = col_missing[col]
        missing_table_rows.append(
            f"| `{col}` | {info['missing_count']:,} | {info['missing_percentage']:.2f}% | "
            f"{'Yes (Requires Attention)' if info['has_missing'] else 'No (Complete)'} |"
        )
    missing_table = "\n".join(missing_table_rows)

    # Build outlier markdown table
    outlier_table_rows = []
    for feat in CANONICAL_FEATURES:
        o = outlier_data[feat]
        outlier_table_rows.append(
            f"| `{feat}` | {o['q1']:.2f} | {o['q3']:.2f} | {o['iqr']:.2f} | "
            f"[{o['lower_bound']:.2f}, {o['upper_bound']:.2f}] | {o['low_outliers_count']} | "
            f"{o['high_outliers_count']} | {o['total_outliers_count']} | {o['outlier_percentage']:.2f}% |"
        )
    outlier_table = "\n".join(outlier_table_rows)

    # Build feature by potability comparison table
    by_pot_rows = []
    for feat in CANONICAL_FEATURES:
        p_info = by_potability[feat]
        np_mean = p_info["non_potable"]["mean"]
        p_mean = p_info["potable"]["mean"]
        diff_mean = p_info["mean_difference"]
        np_med = p_info["non_potable"]["median"]
        p_med = p_info["potable"]["median"]
        by_pot_rows.append(
            f"| `{feat}` | {np_mean:.3f} | {p_mean:.3f} | {diff_mean:+.3f} | {np_med:.3f} | {p_med:.3f} |"
        )
    by_pot_table = "\n".join(by_pot_rows)

    # Build feature-target correlation rows
    ft_corr_rows = []
    for feat in CANONICAL_FEATURES:
        r_val = corr_data["feature_target_correlation"][feat]
        ft_corr_rows.append(f"| `{feat}` | {r_val:+.4f} |")
    ft_corr_table = "\n".join(ft_corr_rows)

    content = f"""# Phase 2.1: Dataset Exploration and Exploratory Data Analysis (EDA) Report

**Project:** Purity – Water Quality Classification and Potability Prediction System  
**Academic Context:** Tribhuvan University, BCA 8th Semester, Project III  
**Dataset:** Kaggle Water Potability Dataset (`data/water_potability.csv`)  
**Execution Environment:** Python 3.13.9, NumPy, Pandas, Matplotlib (Strictly No Scikit-Learn)

---

## 1. Dataset Overview

The dataset was validated and loaded through `backend/ml/eda.py` without modifying the original source file.

- **Total Water Samples (Rows):** {overview['total_rows']:,}
- **Total Columns:** {overview['total_columns']} (9 Physicochemical Feature Columns + 1 Binary Target Column)
- **Feature Schema Order:** `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- **Target Column:** `Potability` (Binary: `0` = Non-potable, `1` = Potable)
- **Numerical Columns:** {overview['numerical_cols_count']} / {overview['total_columns']}
- **Non-Numerical Columns:** {overview['non_numerical_cols_count']}
- **Memory Footprint:** {overview['memory_usage_str']} ({overview['memory_usage_bytes']:,} bytes)
- **Column Data Types:** All 9 physicochemical parameters are 64-bit floating point numbers (`float64`), and the target variable is 64-bit integer (`int64`).

---

## 2. Descriptive Statistics

All descriptive metrics were calculated using Pandas and NumPy directly from the unscaled raw data.

| Feature | Count | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max | Variance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{stats_table}

---

## 3. Missing-Value Analysis

![Missing Values Count and Percentage](missing_values.png)

### Summary of Missing Values

- **Total Missing Entries:** {missing_data['total_missing_values']:,} across {overview['total_rows']:,} observations.
- **Complete Observations (no missing values):** {missing_data['complete_records']:,} ({missing_data['complete_records_percentage']:.2f}% of dataset).
- **Incomplete Observations (at least one missing value):** {missing_data['incomplete_records']:,} ({missing_data['incomplete_records_percentage']:.2f}% of dataset).

### Column Breakdown

| Column | Missing Count | Missing (%) | Status |
| :--- | :--- | :--- | :--- |
{missing_table}

### Key Observations
1. **Sulfate** exhibits the highest missingness: **781 missing records** (23.84%).
2. **ph** exhibits significant missingness: **491 missing records** (14.99%).
3. **Trihalomethanes** has moderate missingness: **162 missing records** (4.95%).
4. The remaining six features (`Hardness`, `Solids`, `Chloramines`, `Conductivity`, `Organic_carbon`, `Turbidity`) and the target variable `Potability` have **zero missing values**.
5. Dropping incomplete records outright would discard 38.61% of the data, which would degrade sample diversity. Imputation strategies must be evaluated carefully during Phase 2.2.

---

## 4. Target-Class Distribution Analysis

![Target Distribution](target_distribution.png)

### Class Distribution Metrics

- **Class 0 (Non-Potable):** {target_data['non_potable_count']:,} samples ({target_data['non_potable_pct']:.2f}%)
- **Class 1 (Potable):** {target_data['potable_count']:,} samples ({target_data['potable_pct']:.2f}%)
- **Total Validated Samples:** {target_data['total_samples']:,}
- **Class Ratio (Majority : Minority):** {target_data['imbalance_ratio']:.3f} : 1
- **Absolute Percentage Difference:** {target_data['difference_pct']:.2f}%

### Class Imbalance Implications
- **Majority Class:** Non-potable (`0`) comprises ~61% of records.
- **Minority Class:** Potable (`1`) comprises ~39% of records.
- **Impact on Evaluation:** A naive classifier that predicts non-potable for every sample would attain approximately 61% accuracy without learning water quality characteristics. Therefore, overall accuracy alone is an insufficient evaluation metric. Precision, Recall, F1-Score, and ROC-AUC must be monitored, and stratified partitioning must be employed during data splitting to maintain identical class ratios in training and testing subsets.

---

## 5. Individual Feature Distribution Findings

![Feature Distributions](feature_distributions.png)

### Observations
1. **Symmetry and Skewness Profile:** Seven of the nine features (`ph`, `Hardness`, `Chloramines`, `Sulfate`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`) exhibit relatively low absolute skewness values below 0.10. While low skewness suggests relatively symmetrical empirical distributions around their central values, skewness alone does not establish that a feature follows a normal or Gaussian distribution. Distribution shapes should also be examined through the generated histograms. Normality cannot be claimed without formal normality analysis.
2. **Positively Skewed Features:**
   - `Solids` shows positive skewness of approximately +0.62, with an extended upper tail reaching 61,227.20 ppm.
   - `Conductivity` shows positive skewness of approximately +0.26, with an upper tail extending toward 753.34 μS/cm.
3. **Mean vs. Median Proximity:** Across all nine features, the sample mean and median values remain closely positioned, reflecting the relative symmetry observed in the majority of variables.

---

## 6. Outlier Exploration (IQR Method)

![Feature Boxplots](feature_boxplots.png)

Outliers were identified using the non-parametric Interquartile Range method ($IQR = Q_3 - Q_1$, bounds $[Q_1 - 1.5 \\times IQR, Q_3 + 1.5 \\times IQR]$).

| Feature | Q1 | Q3 | IQR | Valid Range Bounds | Low Outliers | High Outliers | Total Outliers | Outlier (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{outlier_table}

### Key Observations
1. **Statistical Nature of IQR Detection:** The Interquartile Range method identifies observations falling outside the $1.5 \\times IQR$ boundaries as statistically unusual. An IQR outlier is not automatically a data-recording error, nor is it automatically confirmed to be a genuine natural water variation. The IQR rule serves strictly as an analytical screening mechanism.
2. **Outlier Counts Across Features:**
   - `Hardness` exhibits the highest count of statistical outliers (83 samples, 2.53%), distributed across both lower (44) and upper (39) boundaries.
   - `Chloramines` exhibits 61 statistical outliers (1.86%), with 31 below the lower bound and 30 above the upper bound.
   - `Solids` exhibits 47 statistical outliers (1.43%), all situated along the upper boundary (above 44,831.87 ppm), consistent with its positive skewness.
   - Overall statistical outlier prevalence ranges between 0.34% (`Conductivity`) and 2.53% (`Hardness`) across the features.
3. **Extreme Boundary Observations (pH Values):** The dataset records pH values ranging from a minimum of **0.000** to a maximum of **14.000**. In natural water systems, pH 0 (extreme acidity) and pH 14 (extreme alkalinity) represent extreme boundary observations. These observations require further investigation using water-quality domain knowledge and source documentation. They should not be automatically removed, replaced, or classified as corrupted or invalid at this stage.
4. **Outlier Treatment Policy:** All original observations remain strictly unchanged at this stage. No outlier treatment (such as Winsorization, robust scaling, trimming, or sample deletion) is presented as a predetermined strategy. Outlier treatment decisions will be considered only after domain investigation and, where justified, comparative sensitivity analysis during preprocessing.

---

## 7. Feature Correlation Analysis

![Correlation Heatmap](correlation_heatmap.png)

### Pairwise Feature Correlation Summary
The Pearson correlation coefficients among all 9 physicochemical parameters are remarkably low:
- Maximum positive correlation: `ph` and `Hardness` ($r = +0.082$)
- Maximum negative correlation: `Sulfate` and `Hardness` ($r = -0.107$)
- All pairwise feature correlations fall strictly within the narrow interval of **[-0.11, +0.08]**.

### Correlation with Potability Target

| Feature | Pearson $r$ with Potability |
| :--- | :--- |
{ft_corr_table}

### Interpretation
1. **Weak Pairwise Linear Relationships:** Pearson correlation coefficients among the nine physicochemical parameters fall strictly within the narrow range of **[-0.107, +0.082]**. These values indicate weak pairwise linear associations, demonstrating an absence of linear multicollinearity among input features.
2. **Weak Feature-Target Linear Correlations:** Individual Pearson correlations between input features and the binary `Potability` target are also weak ($|r| < 0.035$ for all nine parameters).
3. **Analytical Limits of Linear Correlation:** Weak individual linear correlations do not prove that features lack predictive value. Pearson's $r$ evaluates only bivariate linear relationships; it does not capture every possible non-linear relationship or complex combination of features.
4. **Premature Algorithm Assumptions Avoided:** It is premature to conclude that non-linear feature interaction modeling is strictly required or that linear models will be ineffective based solely on correlation analysis. The actual predictive usefulness of the features and the relative performance of the approved algorithms—Logistic Regression, Support Vector Machine (SVM), and Random Forest—must be determined through subsequent training and evaluation. No algorithm is favored or dismissed based solely on these correlation findings.

---

## 8. Feature Distribution by Potability Class

![Feature Distribution by Potability](feature_distribution_by_potability.png)

| Feature | Non-Potable Mean | Potable Mean | Difference (Potable - Non-Potable) | Non-Potable Median | Potable Median |
| :--- | :--- | :--- | :--- | :--- | :--- |
{by_pot_table}

### Observations
1. **Overlapping Densities:** Visualized conditioned on the target class ($Potability = 0$ vs. $Potability = 1$), the marginal density distributions for potable and non-potable samples show substantial overlap across all nine individual features.
2. **Mean & Median Proximity:** The differences in sample means and medians between potable and non-potable subsets are minimal across the dataset.
3. **Implications for Evaluation:** While individual univariate distributions do not show distinct separation between classes, the predictive value of multidimensional feature combinations must be evaluated through model training across the approved classification algorithms.

---

## 9. Preprocessing Recommendations for Subsequent Phases

Based on the empirical findings of Phase 2.1, the following considerations are recommended for Phase 2.2 (Data Preprocessing):

1. **Missing-Value Imputation Candidates:**
   - Discarding incomplete records outright would discard 38.61% of data (1,265 samples), which would substantially reduce sample size.
   - Mean and median imputation are both potential candidate approaches for features with missing values (`Sulfate`, `ph`, `Trihalomethanes`).
   - Median imputation is less sensitive to extreme boundary values, whereas mean imputation may also be suitable depending on training-data distributions.
   - The final imputation approach should be chosen after comparing both methods where appropriate.
   - Imputation statistics must be computed strictly on training data to prevent data leakage, and the same fitted values must be applied to testing data and future user inputs.
   - Imputation is not implemented during this task.

2. **Feature Scaling Considerations:**
   - The nine physicochemical features operate on substantially different numerical scales:
     - `ph`: 0.000 to 14.000
     - `Solids`: 320.943 to 61,227.196 ppm
     - `Conductivity`: 181.484 to 753.343 μS/cm
     - `Turbidity`: 1.450 to 6.739 NTU
   - Feature scaling is particularly important for distance- and gradient-sensitive algorithms such as Support Vector Machines (SVM) and Logistic Regression.
   - Z-score standardization may be considered as the initial scaling approach.
   - Scaling parameters (mean and standard deviation) must be fitted using training data only and reapplied to testing partitions and inference inputs.
   - The scaling decision should be evaluated appropriately for each selected algorithm.
   - Scaling is not implemented during this task.

3. **Stratified Train-Test Splitting:**
   - The dataset exhibits a class distribution of approximately 60.99% non-potable (`0`) and 39.01% potable (`1`).
   - Manual stratified splitting should be implemented (without Scikit-learn) to maintain consistent class proportions across training, validation, and test partitions.
   - A fixed random seed must be utilized to guarantee reproducibility.
   - Splitting is not implemented during this task.

4. **Outlier Investigation and Handling:**
   - No observations are removed or modified during this EDA phase.
   - IQR boundaries serve as analytical screening indicators rather than definitive filters.
   - Potentially unusual observations (such as boundary pH readings of 0 and 14) will be investigated using domain context.
   - Any future outlier treatment must be empirically justified and evaluated through sensitivity analysis rather than assumed beforehand.
   - Outlier treatment is not implemented during this task.

---

## 10. Conclusion

Phase 2.1 establishes a verified baseline understanding of the Water Potability Dataset while strictly adhering to the no-Scikit-learn constraint and preserving the original dataset without modifications. The analysis highlights key data characteristics to address in preprocessing: substantial missingness across three features, wide variance in measurement scales, moderate class imbalance (~61% : 39%), and weak pairwise linear correlations. All verified numerical results and analytical indicators remain intact, providing an objective foundation for exploring preprocessing alternatives in Phase 2.2.
"""

    report_path.write_text(content, encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------------
# Main Orchestration Function
# -----------------------------------------------------------------------------

def run_eda(
    data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Execute the full Exploratory Data Analysis workflow.

    Args:
        data_path: Path to water_potability.csv (optional).
        output_dir: Path to directory where reports and charts are saved (optional).

    Returns:
        Dictionary containing all analysis statistics and paths of generated files.
    """
    resolved_data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    resolved_output_dir = Path(output_dir) if output_dir else DEFAULT_REPORTS_DIR

    print("=================================================================")
    print(" PURITY - PHASE 2.1: DATASET EXPLORATION AND EDA")
    print(" Academic Context: BCA 8th Sem, Tribhuvan University - Project III")
    print("=================================================================")
    print(f"Dataset path: {resolved_data_path}")
    print(f"Report output dir: {resolved_output_dir}")

    # Step 1: Load and Validate Dataset
    print("\n[1/6] Loading and validating dataset...")
    df = load_dataset(resolved_data_path)
    print(f"      Success: Loaded {len(df):,} rows and {len(df.columns)} columns.")

    # Step 2: Overview & Missing Values
    print("\n[2/6] Calculating dataset overview and missing values...")
    overview = get_dataset_overview(df)
    missing_data = analyze_missing_values(df)
    print(f"      Complete records: {missing_data['complete_records']:,} ({missing_data['complete_records_percentage']}%)")
    print(f"      Incomplete records: {missing_data['incomplete_records']:,} ({missing_data['incomplete_records_percentage']}%)")

    # Step 3: Target Distribution & Class Imbalance
    print("\n[3/6] Analyzing target distribution (Potability)...")
    target_data = analyze_target_distribution(df)
    print(f"      Non-Potable (0): {target_data['non_potable_count']:,} ({target_data['non_potable_pct']}%)")
    print(f"      Potable (1):     {target_data['potable_count']:,} ({target_data['potable_pct']}%)")
    print(f"      Imbalance ratio: {target_data['imbalance_ratio']}:1")

    # Step 4: Feature Statistics, Outliers & Correlations
    print("\n[4/6] Computing descriptive stats, IQR outliers, and correlations...")
    stats_data = analyze_feature_statistics(df)
    outlier_data = analyze_outliers(df)
    corr_data = analyze_correlations(df)
    by_potability = analyze_features_by_potability(df)

    # Step 5: Generate Visualizations
    print("\n[5/6] Generating visualizations using Matplotlib (no Seaborn/Sklearn)...")
    generated_charts = generate_visualizations(
        df=df,
        missing_data=missing_data,
        target_data=target_data,
        stats_data=stats_data,
        outlier_data=outlier_data,
        corr_data=corr_data,
        output_dir=resolved_output_dir,
    )
    for chart_key, chart_path in generated_charts.items():
        print(f"      Saved: {chart_path.name} ({chart_path.stat().st_size:,} bytes)")

    # Step 6: Generate Markdown Summary Report
    print("\n[6/6] Writing comprehensive Markdown EDA report...")
    report_file = generate_eda_report(
        overview=overview,
        missing_data=missing_data,
        target_data=target_data,
        stats_data=stats_data,
        outlier_data=outlier_data,
        corr_data=corr_data,
        by_potability=by_potability,
        output_dir=resolved_output_dir,
    )
    print(f"      Saved: {report_file.name} ({report_file.stat().st_size:,} bytes)")

    print("\n=================================================================")
    print(" EDA WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=================================================================")

    return {
        "overview": overview,
        "missing_data": missing_data,
        "target_data": target_data,
        "stats_data": stats_data,
        "outlier_data": outlier_data,
        "corr_data": corr_data,
        "by_potability": by_potability,
        "generated_charts": generated_charts,
        "report_file": report_file,
    }


if __name__ == "__main__":
    try:
        run_eda()
        sys.exit(0)
    except Exception as error:
        print(f"\n[ERROR] EDA execution failed: {error}", file=sys.stderr)
        sys.exit(1)
