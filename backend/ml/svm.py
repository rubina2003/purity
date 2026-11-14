"""Manual Support Vector Machine (SVM) Implementation for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.2 - Manual Support Vector Machine Classifier Implementation.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere.
- Pure custom Support Vector Machine binary classification implemented using NumPy and Pandas.
- Supports Linear and Radial Basis Function (RBF) kernels implemented from scratch.
- Manual Sequential Minimal Optimization (SMO) algorithm for dual quadratic programming.
- Soft-margin classification with regularization parameter C.
- Deterministic heuristic for selecting the second Lagrange multiplier (maximum error difference).
- Exact alpha clipping to feasible box bounds [L, H] preserving the dual equality constraint.
- Rigorous handling of degenerate eta pairs via dual objective evaluation at endpoints L and H.
- Strict convergence verification requiring all training samples to satisfy KKT conditions.
- Explicit support vector extraction and sparse kernel representation for efficient inference.
- Rejects non-numeric features, missing values, infinite values, and target leakage.
- Decision outputs represent raw SVM decision scores / decision margins (uncalibrated, not probabilities).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.eda import CANONICAL_FEATURES, TARGET_COLUMN


class ManualSVM:
    """Manual Support Vector Machine (SVM) binary classifier implemented from scratch using NumPy.

    Optimizes the dual soft-margin formulation:
        max_alpha  sum_{i=1}^m alpha_i - 0.5 * sum_{i=1}^m sum_{j=1}^m alpha_i alpha_j y_i y_j K(x_i, x_j)
        subject to 0 <= alpha_i <= C   for all i = 1, ..., m
                   sum_{i=1}^m alpha_i y_i = 0

    Where y_i in {-1.0, +1.0} are internal SVM target labels, C is the soft-margin penalty,
    and K(x_i, x_j) is the kernel evaluation (Linear or RBF).

    Optimization is performed via a deterministic Sequential Minimal Optimization (SMO) solver
    with analytical endpoint objective evaluation for degenerate (eta >= 0) pairs.

    Attributes:
        C: Soft-margin regularization parameter controlling margin slack penalty (> 0).
        kernel: Kernel type ('linear' or 'rbf').
        gamma: RBF kernel coefficient (positive float or 'scale').
        tol: Karush-Kuhn-Tucker (KKT) optimality condition tolerance (> 0).
        max_passes: Maximum consecutive passes without alpha updates required for convergence (> 0).
        max_iter: Maximum total optimization passes allowed (> 0).
        alpha_eps: Numerical tolerance for alpha bound comparisons and change detection (> 0).
        alphas_: Learned 1D array of Lagrange multipliers of shape (n_samples,).
        bias_: Learned decision function scalar bias intercept term b.
        intercept_: Scalar bias intercept identical to bias_.
        support_vectors_: 2D NumPy array of shape (n_support, n_features) containing support vector coordinates.
        support_: 1D integer array containing the row indices of support vectors in training data.
        support_indices_: Alias identical to support_.
        support_vector_labels_: 1D array containing {-1.0, +1.0} labels of support vectors.
        dual_coef_: 1D array of shape (n_support,) containing alpha_i * y_i for each support vector.
        classes_: Binary class labels array np.array([0, 1]).
        feature_names_in_: List of input feature names observed during fit().
        n_features_in_: Number of features observed during fit().
        n_support_: Integer count of active support vectors.
        gamma_: Resolved numeric value of gamma used in RBF kernel (None for linear).
        kernel_: Standardized string of the fitted kernel ('linear' or 'rbf').
        n_iter_: Number of complete optimization passes executed.
        n_passes_: Number of consecutive passes without updates upon termination.
        n_updates_: Total number of successful alpha pair updates executed.
        converged_: Boolean indicating whether genuine KKT convergence was attained within max_iter.
        is_fitted_: Boolean flag indicating whether the model has been fitted.
    """

    SUPPORTED_KERNELS = ("linear", "rbf")

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[float, str] = "scale",
        tol: float = 1e-3,
        max_passes: int = 5,
        max_iter: int = 1000,
        alpha_eps: float = 1e-5,
    ) -> None:
        """Initialize the manual SVM classifier with hyperparameters.

        Args:
            C: Soft-margin regularization parameter. Must be positive and finite (> 0). Default: 1.0.
            kernel: Kernel function type ('linear' or 'rbf'). Default: 'rbf'.
            gamma: Kernel coefficient for 'rbf'. Numeric positive float or 'scale'. Default: 'scale'.
            tol: Tolerance for KKT condition violation checking. Default: 1e-3.
            max_passes: Consecutive passes without changes required for convergence. Default: 5.
            max_iter: Maximum total optimization passes across the dataset. Default: 1000.
            alpha_eps: Numerical epsilon for clipping and change detection. Default: 1e-5.

        Raises:
            TypeError: If hyperparameters have invalid types (e.g. booleans).
            ValueError: If hyperparameters have invalid numeric values or unknown kernel names.
        """
        self.C: float = C
        self.kernel: str = kernel
        self.gamma: Union[float, str] = gamma
        self.tol: float = tol
        self.max_passes: int = max_passes
        self.max_iter: int = max_iter
        self.alpha_eps: float = alpha_eps

        self._validate_constructor_params()

        # Model state attributes initialized to None prior to fit()
        self.alphas_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.support_vectors_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None
        self.support_indices_: Optional[np.ndarray] = None
        self.support_vector_labels_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0
        self.n_support_: int = 0
        self.gamma_: Optional[float] = None
        self.kernel_: Optional[str] = None
        self.n_iter_: int = 0
        self.n_passes_: int = 0
        self.n_updates_: int = 0
        self.converged_: bool = False
        self.is_fitted_: bool = False

    def _validate_constructor_params(self) -> None:
        """Validate constructor parameters for correct types and numeric bounds.

        Raises:
            TypeError: If parameters are of invalid type or booleans.
            ValueError: If parameters violate mathematical boundary constraints.
        """
        # Validate C
        if isinstance(self.C, bool) or not isinstance(self.C, (int, float)):
            raise TypeError(f"C must be a positive float, got {type(self.C).__name__}.")
        if not np.isfinite(self.C) or self.C <= 0.0:
            raise ValueError(f"C must be a positive finite number (> 0), got {self.C}.")
        self.C = float(self.C)

        # Validate kernel
        if not isinstance(self.kernel, str):
            raise TypeError(f"kernel must be a string, got {type(self.kernel).__name__}.")
        normalized_kernel = self.kernel.strip().lower()
        if normalized_kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{self.kernel}'. Supported kernels: {self.SUPPORTED_KERNELS}."
            )
        self.kernel = normalized_kernel

        # Validate gamma
        if isinstance(self.gamma, bool):
            raise TypeError(f"gamma must be 'scale' or a positive float, got {type(self.gamma).__name__}.")
        if isinstance(self.gamma, str):
            normalized_gamma = self.gamma.strip().lower()
            if normalized_gamma != "scale":
                raise ValueError(
                    f"Invalid gamma string '{self.gamma}'. Only 'scale' is supported as a string option."
                )
            self.gamma = normalized_gamma
        elif isinstance(self.gamma, (int, float)):
            if not np.isfinite(self.gamma) or self.gamma <= 0.0:
                raise ValueError(f"Numeric gamma must be positive and finite (> 0), got {self.gamma}.")
            self.gamma = float(self.gamma)
        else:
            raise TypeError(f"gamma must be 'scale' or a positive float, got {type(self.gamma).__name__}.")

        # Validate tol
        if isinstance(self.tol, bool) or not isinstance(self.tol, (int, float)):
            raise TypeError(f"tol must be a positive float, got {type(self.tol).__name__}.")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError(f"tol must be positive and finite (> 0), got {self.tol}.")
        self.tol = float(self.tol)

        # Validate max_passes
        if isinstance(self.max_passes, bool) or not isinstance(self.max_passes, (int, np.integer)):
            raise TypeError(f"max_passes must be a positive integer, got {type(self.max_passes).__name__}.")
        if self.max_passes <= 0:
            raise ValueError(f"max_passes must be a positive integer (> 0), got {self.max_passes}.")
        self.max_passes = int(self.max_passes)

        # Validate max_iter
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)):
            raise TypeError(f"max_iter must be a positive integer, got {type(self.max_iter).__name__}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer (> 0), got {self.max_iter}.")
        self.max_iter = int(self.max_iter)

        # Validate alpha_eps
        if isinstance(self.alpha_eps, bool) or not isinstance(self.alpha_eps, (int, float)):
            raise TypeError(f"alpha_eps must be a positive float, got {type(self.alpha_eps).__name__}.")
        if not np.isfinite(self.alpha_eps) or self.alpha_eps <= 0.0:
            raise ValueError(f"alpha_eps must be positive and finite (> 0), got {self.alpha_eps}.")
        self.alpha_eps = float(self.alpha_eps)

    def _resolve_gamma(self, X: np.ndarray) -> Optional[float]:
        """Resolve and compute gamma coefficient based on training data and kernel type.

        For RBF kernel with 'scale':
            gamma = 1.0 / (n_features * Var(X))
        Fallback: If Var(X) == 0, falls back to 1.0 / n_features (or 1.0) to prevent division by zero.

        For Linear kernel:
            Gamma is not used and resolves to None.

        Args:
            X: 2D NumPy training feature matrix.

        Returns:
            Resolved positive float gamma for RBF kernel, or None for linear kernel.
        """
        if self.kernel == "linear":
            return None

        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)

        # self.gamma == "scale"
        n_features = X.shape[1]
        overall_variance = float(np.var(X))

        if overall_variance <= 1e-12:
            # Fallback if all training samples have identical feature values (zero variance)
            return 1.0 / float(n_features) if n_features > 0 else 1.0

        return 1.0 / (float(n_features) * overall_variance)

    def _compute_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
    ) -> np.ndarray:
        """Compute the kernel Gram matrix K between feature matrices X1 and X2.

        Formulas:
            Linear: K(x_1, x_2) = x_1^T x_2
            RBF:    K(x_1, x_2) = exp(-gamma * ||x_1 - x_2||^2)

        Squared Euclidean distance is calculated vectorially:
            ||x_1 - x_2||^2 = ||x_1||^2 + ||x_2||^2 - 2 * (x_1^T x_2)
        Any small negative values caused by floating-point rounding are clamped to 0.0.

        Args:
            X1: 2D NumPy feature array of shape (m1, n_features).
            X2: 2D NumPy feature array of shape (m2, n_features).

        Returns:
            2D NumPy array of shape (m1, m2) containing pairwise kernel evaluations.

        Raises:
            ValueError: If feature dimension mismatch or unsupported kernel.
            FloatingPointError: If kernel values become non-finite.
        """
        X1_arr = np.atleast_2d(X1)
        X2_arr = np.atleast_2d(X2)

        if X1_arr.shape[1] != X2_arr.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X1 has {X1_arr.shape[1]} features, "
                f"but X2 has {X2_arr.shape[1]} features."
            )

        if self.kernel == "linear":
            K = np.dot(X1_arr, X2_arr.T)
        elif self.kernel == "rbf":
            gamma = self.gamma_ if self.gamma_ is not None else 1.0
            x1_sq = np.sum(X1_arr ** 2, axis=1, keepdims=True)  # (m1, 1)
            x2_sq = np.sum(X2_arr ** 2, axis=1, keepdims=True)  # (m2, 1)
            sq_dists = x1_sq + x2_sq.T - 2.0 * np.dot(X1_arr, X2_arr.T)
            # Clamp potential small negative values from floating point cancellation
            sq_dists = np.maximum(sq_dists, 0.0)
            K = np.exp(-gamma * sq_dists)
        else:
            raise ValueError(f"Unsupported kernel '{self.kernel}'.")

        if not np.all(np.isfinite(K)):
            raise FloatingPointError("Kernel computation produced non-finite values.")

        return K

    def _compute_bias_update(
        self,
        b_old: float,
        i: int,
        j: int,
        alpha_i_old: float,
        alpha_i_new: float,
        alpha_j_old: float,
        alpha_j_new: float,
        y_i: float,
        y_j: float,
        E_i: float,
        E_j: float,
        k_ii: float,
        k_jj: float,
        k_ij: float,
    ) -> float:
        """Compute the updated bias intercept using standard SMO optimality rules.

        Calculates:
            b1 = b_old - E_i - y_i * (alpha_i_new - alpha_i_old) * K_ii - y_j * (alpha_j_new - alpha_j_old) * K_ij
            b2 = b_old - E_j - y_i * (alpha_i_new - alpha_i_old) * K_ij - y_j * (alpha_j_new - alpha_j_old) * K_jj

        Rule:
            - If 0 < alpha_i_new < C: use b1
            - Else if 0 < alpha_j_new < C: use b2
            - Else: use (b1 + b2) / 2

        Returns:
            Updated scalar float bias b.
        """
        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        b1 = b_old - E_i - y_i * delta_alpha_i * k_ii - y_j * delta_alpha_j * k_ij
        b2 = b_old - E_j - y_i * delta_alpha_i * k_ij - y_j * delta_alpha_j * k_jj

        i_non_bound = self.alpha_eps < alpha_i_new < (self.C - self.alpha_eps)
        j_non_bound = self.alpha_eps < alpha_j_new < (self.C - self.alpha_eps)

        if i_non_bound:
            return float(b1)
        elif j_non_bound:
            return float(b2)
        else:
            return float((b1 + b2) / 2.0)

    def _take_step(
        self,
        i: int,
        j: int,
        y: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
    ) -> bool:
        """Attempt to optimize Lagrange multiplier pair (alpha_i, alpha_j).

        Performs:
            1. Box boundary computation [L, H] based on label equality.
            2. Second derivative (eta = 2*K_ij - K_ii - K_jj) computation.
            3. Alpha_j step computation:
               - If eta < -1e-12: standard unconstrained quadratic step clipped to [L, H].
               - If eta >= -1e-12 (degenerate curvature): analytical dual objective evaluation
                 at endpoints L and H, selecting the endpoint yielding superior dual progress.
            4. Alpha_i update preserving the dual equality constraint:
                   alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
            5. Bias update and vectorized error cache update.

        Args:
            i: Index of first Lagrange multiplier.
            j: Index of second Lagrange multiplier.
            y: 1D array of internal targets {-1.0, +1.0}.
            K: Full training kernel Gram matrix of shape (m, m).
            E: Error cache array of shape (m,).

        Returns:
            True if alpha pair was successfully updated; False if skipped.
        """
        if i == j:
            return False

        alpha_i_old = float(self.alphas_[i])
        alpha_j_old = float(self.alphas_[j])
        y_i = float(y[i])
        y_j = float(y[j])
        E_i = float(E[i])
        E_j = float(E[j])

        # 1. Compute bounds L and H
        if y_i != y_j:
            L = max(0.0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0.0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)

        if abs(H - L) < self.alpha_eps:
            return False

        # 2. Compute second derivative eta
        k_ii = float(K[i, i])
        k_jj = float(K[j, j])
        k_ij = float(K[i, j])
        eta = 2.0 * k_ij - k_ii - k_jj

        if eta < -1e-12:
            # Standard case: objective along feasible line segment is strictly concave (eta < 0)
            alpha_j_unc = alpha_j_old - (y_j * (E_i - E_j)) / eta
            # Clip alpha_j to [L, H]
            if alpha_j_unc > H:
                alpha_j_new = H
            elif alpha_j_unc < L:
                alpha_j_new = L
            else:
                alpha_j_new = alpha_j_unc
        else:
            # Degenerate case: eta >= -1e-12 (objective along line is linear or convex)
            # Evaluate the dual objective change at feasible endpoints L and H:
            # Delta W(alpha_j) = y_j * (E_i - E_j) * (alpha_j - alpha_j_old) + 0.5 * eta * (alpha_j - alpha_j_old)^2
            delta_L = L - alpha_j_old
            delta_H = H - alpha_j_old
            slope = y_j * (E_i - E_j)
            dW_L = slope * delta_L + 0.5 * eta * (delta_L ** 2)
            dW_H = slope * delta_H + 0.5 * eta * (delta_H ** 2)

            if dW_L > dW_H + self.alpha_eps and dW_L > self.alpha_eps:
                alpha_j_new = L
            elif dW_H > dW_L + self.alpha_eps and dW_H > self.alpha_eps:
                alpha_j_new = H
            else:
                # No significant objective improvement at either endpoint
                alpha_j_new = alpha_j_old

        # Check if alpha_j step is meaningful
        if abs(alpha_j_new - alpha_j_old) < self.alpha_eps:
            return False

        # 4. Update alpha_i to preserve equality constraint: y_i*a_i + y_j*a_j = const
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)

        # 5. Compute updated bias
        b_old = float(self.bias_)
        b_new = self._compute_bias_update(
            b_old, i, j, alpha_i_old, alpha_i_new, alpha_j_old, alpha_j_new,
            y_i, y_j, E_i, E_j, k_ii, k_jj, k_ij
        )

        # 6. Vectorized Error Cache Update: E_k = f(x_k) - y_k
        delta_i = y_i * (alpha_i_new - alpha_i_old)
        delta_j = y_j * (alpha_j_new - alpha_j_old)
        delta_b = b_new - b_old

        E += delta_i * K[:, i] + delta_j * K[:, j] + delta_b

        # Commit updates
        self.alphas_[i] = alpha_i_new
        self.alphas_[j] = alpha_j_new
        self.bias_ = b_new

        return True

    def _examine_example(
        self,
        i: int,
        y: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
    ) -> bool:
        """Examine sample i for KKT condition violation and deterministically choose second alpha j.

        Selection strategy:
            1. Primary candidate: Maximum error difference j = argmax_{k != i} |E_i - E_k|
            2. Fallback candidate: Deterministic pass over non-bound samples (0 < alpha < C)
            3. Fallback candidate: Deterministic pass over all remaining samples

        Args:
            i: Index of first sample.
            y: Internal target labels {-1.0, +1.0}.
            K: Precomputed Gram kernel matrix.
            E: Error cache.

        Returns:
            True if an alpha pair was updated; False otherwise.
        """
        y_i = float(y[i])
        alpha_i = float(self.alphas_[i])
        E_i = float(E[i])
        r_i = y_i * E_i

        # Check KKT condition violation within configured tolerance
        violates = (r_i < -self.tol and alpha_i < self.C - self.alpha_eps) or \
                   (r_i > self.tol and alpha_i > self.alpha_eps)

        if not violates:
            return False

        m = len(y)

        # Heuristic 1: Primary candidate maximizing |E_i - E_j|
        diffs = np.abs(E - E_i)
        diffs[i] = -1.0  # Exclude self
        j_max = int(np.argmax(diffs))

        if diffs[j_max] > self.alpha_eps:
            if self._take_step(i, j_max, y, K, E):
                return True

        # Heuristic 2: Deterministic scan over non-bound candidates
        non_bound_indices = [
            k for k in range(m)
            if k != i and self.alpha_eps < self.alphas_[k] < (self.C - self.alpha_eps)
        ]
        for k in non_bound_indices:
            if k != j_max and self._take_step(i, k, y, K, E):
                return True

        # Heuristic 3: Deterministic scan over remaining all candidates
        for k in range(m):
            if k != i and k != j_max and k not in non_bound_indices:
                if self._take_step(i, k, y, K, E):
                    return True

        return False

    def _compute_errors(self, y: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Independently recalculate the error cache from scratch: E = K @ (alphas * y) + b - y."""
        return np.dot(K, self.alphas_ * y) + self.bias_ - y

    def _check_kkt_convergence(self, y: np.ndarray, E: np.ndarray) -> bool:
        """Perform a comprehensive verification of KKT optimality conditions across all training samples.

        A sample i satisfies KKT conditions within tolerance tol if:
            - alpha_i == 0        => y_i * E_i >= -tol
            - alpha_i == C        => y_i * E_i <= tol
            - 0 < alpha_i < C     => |y_i * E_i| <= tol

        Returns:
            True if all samples satisfy the KKT conditions within tol; False otherwise.
        """
        for i in range(len(y)):
            ri = float(y[i] * E[i])
            alpha_i = float(self.alphas_[i])
            if (ri < -self.tol and alpha_i < (self.C - self.alpha_eps)) or \
               (ri > self.tol and alpha_i > self.alpha_eps):
                return False
        return True

    def _validate_X(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Validate input feature DataFrame structure, types, values, and schema integrity.

        Args:
            X: Input feature DataFrame.
            is_fit: Boolean indicating whether validation occurs during fit() or inference.

        Returns:
            Validated DataFrame.

        Raises:
            TypeError: If X is not a pandas DataFrame or contains non-numeric columns.
            ValueError: If X is empty, has duplicate columns, missing values, infinite values,
                        contains the target column (target leakage), or feature columns mismatch.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        if X.empty or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input DataFrame is empty (contains 0 records or 0 columns).")

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

        # Reject missing values (imputation must precede SVM)
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
            Validated 1D NumPy array of float labels {0.0, 1.0}.

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
            # NumPy array validation
            if y.ndim != 1:
                raise ValueError(f"y must be a 1-dimensional array, got shape {y.shape}.")
            if not np.issubdtype(y.dtype, np.number):
                raise TypeError(f"Target array y must have numeric dtype, got '{y.dtype}'.")
            y_arr = y.astype(float, copy=True)

        # Reject missing or infinite target values
        if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
            raise ValueError("Target y contains missing (NaN) or infinite (+inf / -inf) values.")

        # Validate unique classes: must be strictly binary {0, 1}
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

        return y_arr

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> ManualSVM:
        """Fit the Support Vector Machine using Sequential Minimal Optimization (SMO).

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).
            y: Binary target labels {0, 1} as a pandas Series or 1D NumPy array.

        Returns:
            Fitted classifier instance (self).

        Raises:
            TypeError: If X or y have invalid types.
            ValueError: If inputs fail schema or value validation.
            FloatingPointError: If kernel values become non-finite.
        """
        X_validated = self._validate_X(X, is_fit=True)
        y_validated = self._validate_y(y, X_validated)

        # Extract matrices without mutating originals
        X_mat = X_validated.to_numpy(dtype=float, copy=True)
        m, n = X_mat.shape

        # Convert target labels: {0, 1} -> {-1.0, +1.0}
        y_svm = np.where(y_validated == 1.0, 1.0, -1.0)

        # Resolve kernel and gamma
        self.kernel_ = self.kernel
        self.gamma_ = self._resolve_gamma(X_mat)

        # Build full training Gram kernel matrix O(m^2)
        K = self._compute_kernel(X_mat, X_mat)

        # Initialize optimization state
        self.alphas_ = np.zeros(m, dtype=float)
        self.bias_ = 0.0
        self.classes_ = np.array([0, 1], dtype=int)
        self.feature_names_in_ = list(X_validated.columns)
        self.n_features_in_ = n

        # Initial decision scores f(x_i) = 0.0 for all i (since alphas=0, b=0)
        # Error cache E_i = f(x_i) - y_i = 0 - y_i = -y_i
        E = -y_svm.copy()

        passes = 0
        n_iter = 0
        n_updates = 0

        # Sequential Minimal Optimization Main Loop
        while passes < self.max_passes and n_iter < self.max_iter:
            num_changed_alphas = 0
            for i in range(m):
                if self._examine_example(i, y_svm, K, E):
                    num_changed_alphas += 1
                    n_updates += 1

            n_iter += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        # Optimization Termination Recording
        self.n_iter_ = n_iter
        self.n_passes_ = passes
        self.n_updates_ = n_updates

        # Independently recalculate error cache to verify consistency
        E_recalc = self._compute_errors(y_svm, K)

        # Comprehensive KKT condition verification across all training samples
        is_kkt_satisfied = self._check_kkt_convergence(y_svm, E_recalc)
        self.converged_ = bool(passes >= self.max_passes and is_kkt_satisfied)
        self.intercept_ = float(self.bias_)

        # Extract Support Vectors: samples with alpha > alpha_eps
        sv_mask = self.alphas_ > self.alpha_eps
        self.support_ = np.flatnonzero(sv_mask)
        self.support_indices_ = self.support_.copy()
        self.support_vectors_ = X_mat[sv_mask].copy()
        self.support_vector_labels_ = y_svm[sv_mask].copy()
        self.dual_coef_ = (self.alphas_[sv_mask] * self.support_vector_labels_).copy()
        self.n_support_ = int(len(self.support_))

        self.is_fitted_ = True
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the raw linear/kernel decision scores: f(x) = sum_{i in SV} dual_coef_i * K(x_i, x) + b.

        Positive values indicate membership in class 1 (Potable);
        negative values indicate class 0 (Non-potable).
        These raw decision scores (or decision margins) must not be interpreted as probabilities
        or direct geometric Euclidean distances in the original input feature space.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            1D NumPy array of shape (n_samples,) containing raw decision scores.

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "ManualSVM instance is not fitted yet. Call fit() before using decision_function()."
            )

        X_validated = self._validate_X(X, is_fit=False)
        X_mat = X_validated.to_numpy(dtype=float, copy=True)

        if self.n_support_ == 0:
            # Fallback for degenerate edge cases where no support vectors were identified
            return np.full(len(X_mat), self.bias_, dtype=float)

        # Compute kernel between inference samples and active support vectors only
        K_test = self._compute_kernel(X_mat, self.support_vectors_)  # Shape: (n_test, n_support)
        scores = np.dot(K_test, self.dual_coef_) + self.bias_
        return scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels {0, 1} for input samples using the decision boundary sign.

        Decision rule:
            y_hat = 1 if f(x) >= 0.0 else 0

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            1D NumPy array of integers {0, 1} containing predicted binary labels.

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, 0).astype(int)
