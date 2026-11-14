"""Manual Random Forest Binary Classification Algorithm for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.3 - Manual Random Forest Implementation from Scratch.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere in the codebase.
- Pure custom decision trees, Gini impurity, bootstrap sampling, random feature
  subspace selection, and ensemble majority voting implemented using NumPy and Pandas.
- Fully deterministic random number generation through NumPy's default_rng(random_state).
- Pure custom Gini impurity splitting criterion with midpoint candidate thresholds.
- Impurity-based feature importance calculated from weighted Gini reduction across splits.
- Compatible with DataFrames produced by Phase 2 data splitting, imputation, and scaling.
- Strict input validation: column ordering, types, missingness, and target leakage prevention.
- Uncalibrated vote proportions exposed via predict_proba() strictly for transparency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.eda import CANONICAL_FEATURES, TARGET_COLUMN


class _DecisionTreeNode:
    """Internal binary decision tree node for ManualRandomForest.

    Represents either a decision split node with left/right children or a terminal
    leaf node with a predicted binary class and observed class distribution.

    Attributes:
        predicted_class: Majority binary class label {0, 1} assigned to this node.
        class_counts: Tuple of observed class counts (count_class_0, count_class_1).
        impurity: Node Gini impurity before split in [0.0, 0.5].
        n_node_samples: Number of training samples reaching this node.
        depth: Depth level of this node in the tree (root is depth 0).
        feature_index: Feature column index used for binary split (None if leaf).
        threshold: Real-valued midpoint threshold for binary split (None if leaf).
        left: Left child subtree (feature <= threshold) (None if leaf).
        right: Right child subtree (feature > threshold) (None if leaf).
    """

    def __init__(
        self,
        predicted_class: int,
        class_counts: Tuple[int, int],
        impurity: float,
        n_node_samples: int,
        depth: int,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional[_DecisionTreeNode] = None,
        right: Optional[_DecisionTreeNode] = None,
    ) -> None:
        self.predicted_class: int = int(predicted_class)
        self.class_counts: Tuple[int, int] = class_counts
        self.impurity: float = float(impurity)
        self.n_node_samples: int = int(n_node_samples)
        self.depth: int = int(depth)
        self.feature_index: Optional[int] = feature_index
        self.threshold: Optional[float] = threshold
        self.left: Optional[_DecisionTreeNode] = left
        self.right: Optional[_DecisionTreeNode] = right

    @property
    def is_leaf(self) -> bool:
        """Return True if this node is a leaf (has no split feature or children)."""
        return self.left is None and self.right is None


class _DecisionTree:
    """Internal binary decision tree classifier built via recursive Gini impurity splitting.

    Parameters:
        max_depth: Maximum allowable depth of the decision tree. None allows unlimited depth.
        min_samples_split: Minimum number of samples required to attempt a binary split.
        min_samples_leaf: Minimum number of samples required in each resulting child leaf.
        max_features_count: Number of candidate features evaluated at each split node.
        min_impurity_decrease: Minimum required reduction in weighted Gini impurity.
        rng: Deterministic NumPy Generator instance for random feature subspace selection.

    Attributes:
        root_: Root _DecisionTreeNode of the fitted tree.
        n_nodes_: Total number of nodes (split nodes + leaves) in the fitted tree.
        max_depth_: Actual maximum depth reached by any leaf in the fitted tree.
        n_bootstrap_: Number of samples in the bootstrap training dataset.
        n_features_: Number of input features observed during fitting.
        feature_importances_: 1D array of unnormalized accumulated Gini importance per feature.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features_count: int = 1,
        min_impurity_decrease: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_features_count: int = max_features_count
        self.min_impurity_decrease: float = min_impurity_decrease
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self.root_: Optional[_DecisionTreeNode] = None
        self.n_nodes_: int = 0
        self.max_depth_: int = 0
        self.n_bootstrap_: int = 0
        self.n_features_: int = 0
        self.feature_importances_: np.ndarray = np.array([], dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _DecisionTree:
        """Fit decision tree on 2D NumPy array X and 1D binary target array y.

        Args:
            X: 2D NumPy array of shape (n_samples, n_features).
            y: 1D NumPy array of binary targets {0, 1} of shape (n_samples,).

        Returns:
            Fitted _DecisionTree instance (self).
        """
        self.n_bootstrap_, self.n_features_ = X.shape
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self.n_nodes_ = 0
        self.max_depth_ = 0

        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X_sub: np.ndarray, y_sub: np.ndarray, depth: int) -> _DecisionTreeNode:
        """Recursively construct a binary decision subtree or leaf node.

        Stopping Conditions for Leaf Creation:
        1. Pure node: all samples have the same class label (impurity <= 1e-12).
        2. Depth limit: current depth >= max_depth (when max_depth is configured).
        3. Sample count limit: node sample count < min_samples_split.
        4. Leaf capacity limit: node sample count < 2 * min_samples_leaf.
        5. No eligible split: candidate thresholds cannot satisfy min_samples_leaf.
        6. Impurity decrease threshold: best impurity decrease <= min_impurity_decrease.

        Leaf Prediction Rule:
        Predicts majority class. If counts are equal (50/50 tie), deterministically predicts 0.

        Args:
            X_sub: Feature subset matrix for samples in current node.
            y_sub: Target subset vector for samples in current node.
            depth: Current tree depth level.

        Returns:
            Constructed _DecisionTreeNode.
        """
        n_samples = len(y_sub)
        n1 = int(np.count_nonzero(y_sub == 1))
        n0 = n_samples - n1

        # Majority class with deterministic tie-breaking (predict 0 on exact tie)
        predicted_class = 1 if n1 > n0 else 0

        # Node Gini impurity: 1 - (p0^2 + p1^2)
        p0 = n0 / n_samples if n_samples > 0 else 0.0
        p1 = n1 / n_samples if n_samples > 0 else 0.0
        impurity = float(1.0 - (p0 * p0 + p1 * p1))

        self.max_depth_ = max(self.max_depth_, depth)

        # Evaluate base stopping conditions
        if (
            impurity <= 1e-12
            or (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
        ):
            self.n_nodes_ += 1
            return _DecisionTreeNode(
                predicted_class=predicted_class,
                class_counts=(n0, n1),
                impurity=impurity,
                n_node_samples=n_samples,
                depth=depth,
            )

        # Random feature subspace selection without replacement
        feat_subset_size = min(self.max_features_count, self.n_features_)
        selected_features = self.rng.choice(self.n_features_, size=feat_subset_size, replace=False)
        selected_features = np.sort(selected_features)

        best_gain = 0.0
        best_feat_idx: Optional[int] = None
        best_threshold: Optional[float] = None

        # Search for optimal split among selected candidate features
        for feat_idx in selected_features:
            x_col = X_sub[:, feat_idx]
            sort_idx = np.argsort(x_col)
            x_sorted = x_col[sort_idx]
            y_sorted = y_sub[sort_idx]

            split_mask = x_sorted[:-1] != x_sorted[1:]
            if not np.any(split_mask):
                continue

            split_indices = np.where(split_mask)[0]
            cumsum_y = np.cumsum(y_sorted)
            total_y1 = int(cumsum_y[-1])

            for i in split_indices:
                n_L = i + 1
                n_R = n_samples - n_L

                if n_L < self.min_samples_leaf or n_R < self.min_samples_leaf:
                    continue

                threshold = float((x_sorted[i] + x_sorted[i + 1]) / 2.0)

                n_L1 = int(cumsum_y[i])
                n_L0 = n_L - n_L1
                n_R1 = total_y1 - n_L1
                n_R0 = n_R - n_R1

                p_L0 = n_L0 / n_L
                p_L1 = n_L1 / n_L
                gini_L = 1.0 - (p_L0 * p_L0 + p_L1 * p_L1)

                p_R0 = n_R0 / n_R
                p_R1 = n_R1 / n_R
                gini_R = 1.0 - (p_R0 * p_R0 + p_R1 * p_R1)

                child_gini = (n_L / n_samples) * gini_L + (n_R / n_samples) * gini_R
                gain = impurity - child_gini

                # Check minimum impurity decrease and strict superiority over best_gain
                # Strict > with 1e-12 keeps the first seen in case of near-ties (lower feature index & threshold)
                if gain > self.min_impurity_decrease and gain > (best_gain + 1e-12):
                    best_gain = gain
                    best_feat_idx = int(feat_idx)
                    best_threshold = threshold

        # If no split achieved required impurity decrease or satisfied leaf constraints, create leaf
        if best_feat_idx is None or best_threshold is None or best_gain <= 0.0:
            self.n_nodes_ += 1
            return _DecisionTreeNode(
                predicted_class=predicted_class,
                class_counts=(n0, n1),
                impurity=impurity,
                n_node_samples=n_samples,
                depth=depth,
            )

        # Split data partition
        left_mask = X_sub[:, best_feat_idx] <= best_threshold
        right_mask = ~left_mask

        # Defensive check: ensure partition is non-degenerate
        if np.all(left_mask) or np.all(right_mask):
            self.n_nodes_ += 1
            return _DecisionTreeNode(
                predicted_class=predicted_class,
                class_counts=(n0, n1),
                impurity=impurity,
                n_node_samples=n_samples,
                depth=depth,
            )

        # Accumulate weighted Gini impurity reduction: (n_node / n_bootstrap) * delta_gini
        node_importance = (n_samples / self.n_bootstrap_) * best_gain
        self.feature_importances_[best_feat_idx] += node_importance

        self.n_nodes_ += 1
        left_child = self._build_tree(X_sub[left_mask], y_sub[left_mask], depth + 1)
        right_child = self._build_tree(X_sub[right_mask], y_sub[right_mask], depth + 1)

        return _DecisionTreeNode(
            predicted_class=predicted_class,
            class_counts=(n0, n1),
            impurity=impurity,
            n_node_samples=n_samples,
            depth=depth,
            feature_index=best_feat_idx,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _predict_row(self, x: np.ndarray) -> int:
        """Traverse tree from root to leaf to predict binary class for a single feature vector."""
        node = self.root_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classes {0, 1} for each sample in 2D NumPy array X.

        Args:
            X: 2D NumPy array of shape (n_samples, n_features).

        Returns:
            1D NumPy array of integers {0, 1} of shape (n_samples,).
        """
        return np.array([self._predict_row(row) for row in X], dtype=int)


class ManualRandomForest:
    """Manual Random Forest Binary Classifier implemented from scratch using NumPy.

    Constructs an ensemble of decision trees trained on separate bootstrap samples of the
    training data, with randomized feature subspace selection at each split node.
    Final binary classification predictions are obtained via majority voting across trees.
    Uncalibrated vote proportions are exposed via predict_proba() strictly as diagnostic output.

    CRITICAL STATISTICAL NOTICE:
    predict_proba() returns uncalibrated forest tree-vote proportions, which represent
    the fraction of constituent decision trees voting for each class. They are NOT calibrated
    Bayesian posterior probabilities and MUST NOT be interpreted as physical or clinical
    guarantees of water safety or potability.

    Attributes:
        n_estimators: Number of decision trees in the ensemble (default: 100).
        max_depth: Maximum allowable depth for each decision tree (default: None).
        min_samples_split: Minimum samples required to split an internal node (default: 2).
        min_samples_leaf: Minimum samples required in each child leaf node (default: 1).
        max_features: Candidate feature count configuration ('sqrt', 'log2', or int).
        random_state: Integer seed for reproducible pseudo-random number generation.
        min_impurity_decrease: Minimum required reduction in weighted Gini impurity.
        estimators_: List of fitted _DecisionTree instances in the ensemble.
        classes_: Binary class labels array np.array([0, 1]).
        feature_names_in_: List of input feature names observed during fit().
        n_features_in_: Total count of input features observed during fit().
        n_estimators_: Actual count of fitted decision trees.
        feature_importances_: Normalized forest-level feature importance array summing to 1.0.
        is_fitted_: Boolean flag indicating whether the ensemble has been fitted.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int] = "sqrt",
        random_state: Optional[int] = 42,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """Initialize ManualRandomForest with hyperparameters and deterministic seed.

        Args:
            n_estimators: Number of trees in the forest. Must be positive integer (> 0).
            max_depth: Maximum depth of individual trees. Positive integer or None.
            min_samples_split: Minimum samples to split a node. Integer >= 2.
            min_samples_leaf: Minimum samples in a leaf. Integer >= 1.
            max_features: Number of features per split: 'sqrt', 'log2', or positive int.
            random_state: Deterministic random seed integer or None.
            min_impurity_decrease: Minimum required Gini decrease (finite float >= 0.0).

        Raises:
            TypeError: If hyperparameters have invalid types (e.g. booleans).
            ValueError: If hyperparameters violate mathematical constraints.
        """
        self.n_estimators: int = n_estimators
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_features: Union[str, int] = max_features
        self.random_state: Optional[int] = random_state
        self.min_impurity_decrease: float = min_impurity_decrease

        self._validate_constructor_params()

        # Model state attributes initialized prior to fit()
        self.estimators_: List[_DecisionTree] = []
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0
        self.n_estimators_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def _validate_constructor_params(self) -> None:
        """Validate constructor parameters for correct types and numeric bounds.

        Raises:
            TypeError: If parameters are of invalid type or booleans.
            ValueError: If parameters violate mathematical boundary constraints.
        """
        # Validate n_estimators
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)):
            raise TypeError(
                f"n_estimators must be a positive integer, got {type(self.n_estimators).__name__}."
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be a positive integer (> 0), got {self.n_estimators}."
            )

        # Validate max_depth
        if self.max_depth is not None:
            if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, (int, np.integer)):
                raise TypeError(
                    f"max_depth must be an integer or None, got {type(self.max_depth).__name__}."
                )
            if self.max_depth <= 0:
                raise ValueError(
                    f"max_depth must be a positive integer (> 0) when provided, got {self.max_depth}."
                )

        # Validate min_samples_split
        if isinstance(self.min_samples_split, bool) or not isinstance(
            self.min_samples_split, (int, np.integer)
        ):
            raise TypeError(
                f"min_samples_split must be an integer >= 2, got {type(self.min_samples_split).__name__}."
            )
        if self.min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be an integer >= 2, got {self.min_samples_split}."
            )

        # Validate min_samples_leaf
        if isinstance(self.min_samples_leaf, bool) or not isinstance(
            self.min_samples_leaf, (int, np.integer)
        ):
            raise TypeError(
                f"min_samples_leaf must be an integer >= 1, got {type(self.min_samples_leaf).__name__}."
            )
        if self.min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be an integer >= 1, got {self.min_samples_leaf}."
            )

        # Validate max_features
        if isinstance(self.max_features, str):
            if self.max_features not in ("sqrt", "log2"):
                raise ValueError(
                    f"max_features string must be 'sqrt' or 'log2', got '{self.max_features}'."
                )
        elif isinstance(self.max_features, bool) or not isinstance(
            self.max_features, (int, np.integer)
        ):
            raise TypeError(
                f"max_features must be 'sqrt', 'log2', or a positive integer, got {type(self.max_features).__name__}."
            )
        else:
            if self.max_features <= 0:
                raise ValueError(
                    f"max_features integer must be positive (> 0), got {self.max_features}."
                )

        # Validate random_state
        if self.random_state is not None:
            if isinstance(self.random_state, bool) or not isinstance(
                self.random_state, (int, np.integer)
            ):
                raise TypeError(
                    f"random_state must be an integer or None, got {type(self.random_state).__name__}."
                )

        # Validate min_impurity_decrease
        if isinstance(self.min_impurity_decrease, bool) or not isinstance(
            self.min_impurity_decrease, (int, float)
        ):
            raise TypeError(
                f"min_impurity_decrease must be a non-negative float, got {type(self.min_impurity_decrease).__name__}."
            )
        if not np.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0.0:
            raise ValueError(
                f"min_impurity_decrease must be a finite float >= 0.0, got {self.min_impurity_decrease}."
            )

    @staticmethod
    def _gini_impurity(y: Union[np.ndarray, List[int], pd.Series]) -> float:
        """Calculate Gini impurity for a binary target vector y.

        Formula:
            Gini = 1 - (p_0^2 + p_1^2)
        where p_0 = n_0 / n and p_1 = n_1 / n.

        For an empty array, returns 0.0.
        For a pure node, returns 0.0.
        For an evenly balanced two-class node, returns 0.5.

        Args:
            y: Binary target labels {0, 1}.

        Returns:
            Scalar Gini impurity in [0.0, 0.5].
        """
        y_arr = np.asarray(y)
        n = len(y_arr)
        if n == 0:
            return 0.0
        n1 = int(np.count_nonzero(y_arr == 1))
        n0 = n - n1
        p0 = n0 / n
        p1 = n1 / n
        return float(1.0 - (p0 * p0 + p1 * p1))

    @staticmethod
    def _calculate_candidate_thresholds(
        values: Union[np.ndarray, List[float], pd.Series],
    ) -> np.ndarray:
        """Calculate midpoint candidate split thresholds between consecutive distinct values.

        Formula:
            threshold = (v_i + v_{i+1}) / 2.0
        for consecutive sorted distinct values v_i and v_{i+1}.

        Args:
            values: 1D array-like of feature values.

        Returns:
            1D NumPy array of candidate thresholds. Returns empty array if < 2 distinct values.
        """
        v = np.asarray(values, dtype=float)
        if len(v) < 2:
            return np.array([], dtype=float)
        unique_vals = np.unique(v)
        if len(unique_vals) < 2:
            return np.array([], dtype=float)
        return (unique_vals[:-1] + unique_vals[1:]) / 2.0

    def _get_max_features_count(self, n_features: int) -> int:
        """Compute candidate feature count for each split node based on configuration.

        Args:
            n_features: Total count of features in the training dataset.

        Returns:
            Integer feature subset count bounded in [1, n_features].
        """
        if self.max_features == "sqrt":
            return max(1, int(math.isqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(math.floor(math.log2(n_features))))
        elif isinstance(self.max_features, (int, np.integer)):
            return min(n_features, max(1, int(self.max_features)))
        raise ValueError(f"Unsupported max_features configuration: {self.max_features}")

    def _validate_X(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Validate feature DataFrame X according to project constraints.

        Args:
            X: Input feature DataFrame.
            is_fit: True if called during fit(), False if called during inference.

        Returns:
            Validated pd.DataFrame.

        Raises:
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If X is empty, has missing/infinite values, duplicate columns,
                        contains 'Potability', or has column ordering mismatches.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}.")

        if X.empty or len(X.columns) == 0:
            raise ValueError("Input DataFrame X is empty (contains 0 samples or 0 columns).")

        # Reject target leakage: Potability must never be passed as a feature column
        if TARGET_COLUMN in X.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' found in feature matrix X. "
                f"Target column must not be an input feature to prevent target leakage."
            )

        # Check for duplicate column names
        if len(X.columns) != len(set(X.columns)):
            duplicates = X.columns[X.columns.duplicated()].tolist()
            raise ValueError(f"Input DataFrame contains duplicate column names: {duplicates}")

        # Check for non-numeric data types
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise TypeError(f"Column '{col}' must have numeric dtype, got '{X[col].dtype}'.")

        # Reject missing values (imputation must precede classification)
        if X.isnull().any().any():
            null_summary = {col: int(cnt) for col, cnt in X.isnull().sum().items() if cnt > 0}
            raise ValueError(f"Input contains missing (NaN) values: {null_summary}.")

        # Reject infinite values (+inf or -inf)
        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("Input DataFrame contains infinite values (+inf or -inf).")

        if is_fit:
            # Check canonical ordering if the canonical water quality features are present
            if set(CANONICAL_FEATURES).issubset(set(X.columns)):
                actual_canonical = [c for c in X.columns if c in CANONICAL_FEATURES]
                if actual_canonical != CANONICAL_FEATURES:
                    raise ValueError(
                        f"Feature columns do not follow canonical ordering.\n"
                        f"Expected: {CANONICAL_FEATURES}\n"
                        f"Actual:   {actual_canonical}"
                    )
        else:
            # During inference: feature columns and order must strictly match training
            actual_cols = list(X.columns)
            if actual_cols != self.feature_names_in_:
                raise ValueError(
                    f"Feature mismatch between fit and inference.\n"
                    f"Expected: {self.feature_names_in_}\n"
                    f"Actual:   {actual_cols}"
                )

        return X

    def _validate_y(
        self,
        y: Union[pd.Series, np.ndarray],
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Validate target labels y against feature DataFrame X.

        Args:
            y: Target binary labels as a pandas Series or 1D NumPy array.
            X: Associated feature DataFrame.

        Returns:
            Validated 1D NumPy array of binary integer labels {0, 1}.

        Raises:
            TypeError: If y is not a pandas Series or NumPy ndarray, or contains non-numeric values.
            ValueError: If y is empty, has mismatched length/index, non-binary values, or single class.
        """
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(f"y must be a pandas Series or 1D numpy array, got {type(y).__name__}.")

        if len(y) == 0:
            raise ValueError("Target array y is empty (contains 0 samples).")

        if len(X) != len(y):
            raise ValueError(
                f"Sample count mismatch: X has {len(X)} rows, but y has {len(y)} records."
            )

        if isinstance(y, pd.Series):
            if not pd.api.types.is_numeric_dtype(y):
                raise TypeError(f"Target Series y must have numeric dtype, got '{y.dtype}'.")
            if not X.index.equals(y.index):
                raise ValueError(
                    "Index mismatch: X and y must have identical row indices and ordering."
                )
            y_arr = y.to_numpy(dtype=float, copy=True)
        else:
            if y.ndim != 1:
                raise ValueError(f"y must be a 1-dimensional array, got shape {y.shape}.")
            if not np.issubdtype(y.dtype, np.number):
                raise TypeError(f"Target array y must have numeric dtype, got '{y.dtype}'.")
            y_arr = y.astype(float, copy=True)

        # Reject missing or infinite target values
        if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
            raise ValueError("Target y contains missing (NaN) or infinite (+inf / -inf) values.")

        # Validate unique classes
        unique_vals = np.unique(y_arr)
        if len(unique_vals) == 1:
            raise ValueError(
                f"Training target contains only one class ({unique_vals[0]}). "
                f"Both binary classes {{0, 1}} must be represented during fitting."
            )

        float_unique = set(np.round(unique_vals, 10))
        if float_unique != {0.0, 1.0}:
            raise ValueError(
                f"Target labels must be strictly binary {{0, 1}}, got: {unique_vals.tolist()}"
            )

        return y_arr.astype(int)

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> ManualRandomForest:
        """Fit Random Forest ensemble using bootstrap sampling and recursive tree splits.

        For each of the n_estimators:
        1. Generates a separate bootstrap sample with replacement from training indices.
        2. Derives a deterministic tree sub-RNG for split node feature selections.
        3. Recursively constructs a decision tree using midpoint Gini impurity minimization.
        4. Accumulates weighted Gini impurity reductions for feature importance.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).
            y: Binary target labels {0, 1} as a pandas Series or 1D NumPy array.

        Returns:
            Fitted classifier instance (self).

        Raises:
            TypeError: If X or y have invalid types.
            ValueError: If inputs violate validation rules.
        """
        X_validated = self._validate_X(X, is_fit=True)
        y_validated = self._validate_y(y, X_validated)

        # Extract NumPy views without modifying original inputs
        X_mat = X_validated.to_numpy(dtype=float, copy=True)
        y_vec = y_validated.astype(int, copy=True)

        n_samples, n_features = X_mat.shape
        max_features_count = self._get_max_features_count(n_features)

        # Reset fitted attributes for safe re-fitting
        self.feature_names_in_ = list(X_validated.columns)
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1], dtype=int)
        self.estimators_ = []

        master_rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap sample: draw n_samples with replacement from {0, ..., n_samples-1}
            boot_indices = master_rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_mat[boot_indices]
            y_boot = y_vec[boot_indices]

            # Generate deterministic child generator for this tree's internal splits
            tree_seed = int(master_rng.integers(0, np.iinfo(np.int32).max))
            tree_rng = np.random.default_rng(tree_seed)

            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features_count=max_features_count,
                min_impurity_decrease=self.min_impurity_decrease,
                rng=tree_rng,
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        self.n_estimators_ = len(self.estimators_)

        # Aggregate and normalize feature importance across all trees
        total_importances = np.sum([t.feature_importances_ for t in self.estimators_], axis=0)
        sum_imp = float(np.sum(total_importances))
        if sum_imp > 0.0:
            self.feature_importances_ = total_importances / sum_imp
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate uncalibrated class vote proportions across the ensemble.

        Formula:
            P(class k | x) = (1 / T) * sum_{t=1}^T I(y_hat_t == k)
        where T is the number of trees in the ensemble.

        CRITICAL STATISTICAL NOTICE:
        These values represent internal ensemble vote fractions and are NOT calibrated
        posterior probabilities. They must not be interpreted as physical certainty of water safety.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            2D NumPy array of shape (n_samples, 2):
                - Column 0: Proportion of trees voting Non-potable (class 0).
                - Column 1: Proportion of trees voting Potable (class 1).
            Row values sum to 1.0.

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "ManualRandomForest instance is not fitted yet. "
                "Call fit() before using predict_proba()."
            )

        X_validated = self._validate_X(X, is_fit=False)
        X_mat = X_validated.to_numpy(dtype=float, copy=True)

        # Collect binary predictions from all fitted trees: shape (n_estimators, n_samples)
        all_tree_preds = np.array([tree.predict(X_mat) for tree in self.estimators_])

        votes_1 = np.sum(all_tree_preds == 1, axis=0)
        votes_0 = self.n_estimators_ - votes_1

        p0 = votes_0 / self.n_estimators_
        p1 = votes_1 / self.n_estimators_

        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels {0, 1} for input samples using majority voting.

        Majority Decision Rule:
            y_hat = 1 if votes(1) > votes(0) else 0

        Tie-Breaking Rule:
            In case of an exact 50/50 vote tie between trees, class 0 is deterministically chosen.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            1D NumPy array of integers {0, 1} of shape (n_samples,).

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        proba = self.predict_proba(X)
        p0 = proba[:, 0]
        p1 = proba[:, 1]
        # Majority voting with tie-breaking: predict 1 strictly if p1 > p0, else 0 (tie -> 0)
        return np.where(p1 > p0, 1, 0).astype(int)

