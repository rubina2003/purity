"""Manual Logistic Regression Implementation for Purity.

Academic Context: BCA, 8th Semester, Tribhuvan University - Project III.
Project: Purity - Water Quality Classification and Potability Prediction System.
Phase: 3.1 - Manual Logistic Regression Classifier Implementation.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- Strictly NO Scikit-learn (sklearn) is imported or used anywhere.
- Pure custom Logistic Regression binary classification implemented using NumPy and Pandas.
- Fully vectorized NumPy operations for forward propagation, loss, and gradient descent.
- Numerically stable sigmoid function preventing overflow/underflow for extreme logits.
- Numerically stable binary cross-entropy loss using the log-sum-exp / softplus formulation.
- Deterministic zero initialization for weights and bias without random states.
- Compatible with DataFrames produced by Phase 2 data splitting, imputation, and scaling.
- Strict input validation: column ordering, types, missingness, and target leakage prevention.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backend.ml.eda import CANONICAL_FEATURES, TARGET_COLUMN


class ManualLogisticRegression:
    """Manual Logistic Regression binary classifier implemented from scratch using NumPy.

    Models the posterior probability of the positive class (Potability = 1) using the
    logistic sigmoid link function applied to a linear combination of input features:
        z = w^T x + b = X w + b
        y_hat = P(y=1|x) = sigma(z) = 1 / (1 + exp(-z))

    Parameters are optimized via batch gradient descent minimizing binary cross-entropy loss:
        L = - (1/m) * sum [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]

    Attributes:
        learning_rate: Gradient descent step size alpha (> 0).
        epochs: Maximum number of batch gradient descent iterations (> 0).
        threshold: Decision threshold for binary classification in [0.0, 1.0].
        weights_: Learned weight vector of shape (n_features,).
        bias_: Learned scalar bias intercept term.
        classes_: Binary class labels array np.array([0, 1]).
        feature_names_in_: List of input feature names observed during fit().
        n_features_in_: Number of features observed during fit().
        loss_history_: List of loss values evaluated at each training iteration.
        n_iter_: Number of completed gradient descent iterations.
        is_fitted_: Boolean flag indicating whether the model has been fitted.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the manual logistic regression classifier with hyperparameters.

        Args:
            learning_rate: Learning rate alpha controlling gradient descent step size.
                           Must be a positive finite float (> 0). Default: 0.01.
            epochs: Maximum number of gradient descent iterations.
                    Must be a positive integer (> 0). Default: 1000.
            threshold: Probability decision threshold for binary classification.
                       Must be between 0.0 and 1.0 inclusive. Default: 0.5.

        Raises:
            TypeError: If hyperparameters have invalid types (e.g. booleans or strings).
            ValueError: If hyperparameters have invalid values (e.g. <= 0 or out of bounds).
        """
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.threshold: float = threshold

        self._validate_constructor_params()

        # Model state attributes initialized to empty/None prior to fit()
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_in_: List[str] = []
        self.n_features_in_: int = 0
        self.loss_history_: List[float] = []
        self.n_iter_: int = 0
        self.is_fitted_: bool = False

    def _validate_constructor_params(self) -> None:
        """Validate constructor hyperparameters for type and numeric bounds.

        Raises:
            TypeError: If hyperparameters have invalid types.
            ValueError: If hyperparameters have invalid numeric values.
        """
        # Validate learning_rate
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate must be a positive float, got {type(self.learning_rate).__name__}."
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive and finite (> 0), got {self.learning_rate}."
            )
        self.learning_rate = float(self.learning_rate)

        # Validate epochs
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, (int, np.integer)):
            raise TypeError(
                f"epochs must be a positive integer, got {type(self.epochs).__name__}."
            )
        if self.epochs <= 0:
            raise ValueError(
                f"epochs must be a positive integer (> 0), got {self.epochs}."
            )
        self.epochs = int(self.epochs)

        # Validate threshold
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise TypeError(
                f"threshold must be a float between 0.0 and 1.0, got {type(self.threshold).__name__}."
            )
        if not np.isfinite(self.threshold) or not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"threshold must be between 0.0 and 1.0 inclusive, got {self.threshold}."
            )
        self.threshold = float(self.threshold)

    def _sigmoid(self, z: Union[float, int, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
        """Compute the logistic sigmoid activation function with numerical stability.

        The standard logistic formula is:
            sigma(z) = 1 / (1 + exp(-z))

        To prevent 64-bit floating point overflow (exp(z) -> inf for z > ~709) and
        avoid numerical inaccuracies for large negative numbers, a piecewise numerically
        stable formulation is used:
            For z >= 0: sigma(z) = 1 / (1 + exp(-z))
            For z < 0:  sigma(z) = exp(z) / (1 + exp(z))

        In both branches, exp() is only ever evaluated on non-positive values (<= 0),
        guaranteeing that exp() never overflows. Logits are also safely clipped to [-709, 709].

        Args:
            z: Input logit(s) as a scalar float, int, list, or NumPy array.

        Returns:
            Computed probability value(s) in the closed interval [0.0, 1.0].
        """
        z_arr = np.asarray(z, dtype=float)
        # Clip to safely bounded range for float64 representation
        z_safe = np.clip(z_arr, -709.0, 709.0)

        pos_mask = z_safe >= 0
        result = np.empty_like(z_safe, dtype=float)

        # For z >= 0: exp(-z) <= 1, no overflow possible
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z_safe[pos_mask]))

        # For z < 0: exp(z) < 1, no overflow possible
        exp_neg = np.exp(z_safe[~pos_mask])
        result[~pos_mask] = exp_neg / (1.0 + exp_neg)

        # Ensure probabilities strictly reside in [0.0, 1.0]
        result_clipped = np.clip(result, 0.0, 1.0)

        if z_arr.ndim == 0:
            return float(result_clipped)
        return result_clipped

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform forward propagation to calculate linear logits and estimated probabilities.

        Calculates:
            z = X w + b
            y_hat = sigma(z)

        Args:
            X: 2D NumPy feature matrix of shape (n_samples, n_features).

        Returns:
            Tuple of:
                - z: 1D NumPy array of raw decision scores (logits) of shape (n_samples,).
                - y_hat: 1D NumPy array of estimated probabilities of shape (n_samples,).
        """
        z = np.dot(X, self.weights_) + self.bias_
        y_hat = self._sigmoid(z)
        return z, y_hat

    def _compute_loss(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        z: Optional[np.ndarray] = None,
    ) -> float:
        """Compute binary cross-entropy loss with log-domain numerical stability.

        The standard binary cross-entropy loss across m samples is:
            L = - (1/m) * sum [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]

        When logits z are supplied, the mathematically equivalent and stable softplus
        formulation is evaluated directly to eliminate catastrophic cancellation and log(0):
            L_i = max(z_i, 0) - y_i * z_i + log(1 + exp(-|z_i|))

        When logits are omitted, probability clipping to [1e-15, 1 - 1e-15] is enforced.

        Args:
            y: Ground-truth binary target labels {0.0, 1.0} of shape (n_samples,).
            y_hat: Estimated probabilities of shape (n_samples,).
            z: Optional raw decision scores (logits) of shape (n_samples,).

        Returns:
            Scalar float binary cross-entropy loss.
        """
        m = len(y)
        if m == 0:
            return 0.0

        if z is not None:
            z_arr = np.asarray(z, dtype=float)
            # Numerically stable log-sum-exp formulation
            loss_arr = np.maximum(z_arr, 0.0) - y * z_arr + np.log(1.0 + np.exp(-np.abs(z_arr)))
            return float(np.mean(loss_arr))

        # Probability-based calculation with epsilon clipping to avoid log(0)
        eps = 1e-15
        y_hat_safe = np.clip(y_hat, eps, 1.0 - eps)
        loss_arr = -(y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe))
        return float(np.mean(loss_arr))

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_hat: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute analytical gradients of binary cross-entropy loss with respect to weights and bias.

        Vectorized gradient derivations:
            dL/dw = (1/m) * X^T (y_hat - y)
            dL/db = (1/m) * sum(y_hat - y) = mean(y_hat - y)

        Args:
            X: 2D NumPy feature matrix of shape (n_samples, n_features).
            y: 1D NumPy array of true binary targets {0.0, 1.0} of shape (n_samples,).
            y_hat: 1D NumPy array of predicted probabilities of shape (n_samples,).

        Returns:
            Tuple of:
                - dw: 1D NumPy array of weight gradients of shape (n_features,).
                - db: Scalar float bias gradient.
        """
        m = X.shape[0]
        error = y_hat - y
        dw = (1.0 / m) * np.dot(X.T, error)
        db = float(np.mean(error))
        return dw, db

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

        return y_arr

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> ManualLogisticRegression:
        """Fit the logistic regression model using batch gradient descent.

        Performs full-batch gradient descent for `epochs` iterations, updating weights
        and bias according to analytical gradients computed over the complete dataset.
        Zero initialization is used deterministically for weights and bias.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).
            y: Binary target labels {0, 1} as a pandas Series or 1D NumPy array.

        Returns:
            Fitted classifier instance (self).

        Raises:
            TypeError: If X or y have invalid types.
            ValueError: If inputs fail validation.
            FloatingPointError: If parameter updates or losses become non-finite.
        """
        X_validated = self._validate_X(X, is_fit=True)
        y_validated = self._validate_y(y, X_validated)

        # Extract underlying NumPy matrices without modifying original inputs
        X_mat = X_validated.to_numpy(dtype=float, copy=True)
        y_vec = y_validated.astype(float, copy=True)

        n_samples, n_features = X_mat.shape

        # Deterministic zero initialization
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.classes_ = np.array([0, 1], dtype=int)
        self.feature_names_in_ = list(X_validated.columns)
        self.n_features_in_ = n_features
        self.loss_history_ = []
        self.n_iter_ = 0

        # Batch Gradient Descent Optimization Loop
        for epoch in range(self.epochs):
            # 1. Forward propagation
            z, y_hat = self._forward(X_mat)

            # 2. Compute and record binary cross-entropy loss
            loss = self._compute_loss(y_vec, y_hat, z=z)
            if not np.isfinite(loss):
                raise FloatingPointError(
                    f"Encountered non-finite loss ({loss}) at iteration {epoch}."
                )
            self.loss_history_.append(float(loss))

            # 3. Compute analytical gradients
            dw, db = self._compute_gradients(X_mat, y_vec, y_hat)
            if not np.all(np.isfinite(dw)) or not np.isfinite(db):
                raise FloatingPointError(
                    f"Encountered non-finite gradient at iteration {epoch}."
                )

            # 4. Parameter update
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db
            self.n_iter_ += 1

            if not np.all(np.isfinite(self.weights_)) or not np.isfinite(self.bias_):
                raise FloatingPointError(
                    f"Model parameters became non-finite at iteration {epoch}."
                )

        self.is_fitted_ = True
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the raw linear decision scores (logits): z = X w + b.

        These raw scores represent the linear log-odds (logits) before applying the sigmoid function.
        Positive values indicate higher likelihood of class 1; negative values indicate class 0.

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
                "ManualLogisticRegression instance is not fitted yet. "
                "Call fit() before using decision_function()."
            )

        X_validated = self._validate_X(X, is_fit=False)
        X_mat = X_validated.to_numpy(dtype=float, copy=True)
        z = np.dot(X_mat, self.weights_) + self.bias_
        return z

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute estimated class probabilities for input samples via the sigmoid function.

        Applies the sigmoid activation to the raw linear decision scores (logits):
            P(y=1|x) = sigma(z) = 1 / (1 + exp(-z))
            P(y=0|x) = 1.0 - P(y=1|x)

        Note:
            These values represent estimated probabilities derived directly from the
            logistic sigmoid link function. Formal empirical probability calibration
            has not been established or evaluated in this implementation.

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            2D NumPy array of shape (n_samples, 2):
                - Column 0: Estimated probability of class 0 (Non-potable).
                - Column 1: Estimated probability of class 1 (Potable).
            Probabilities for each sample sum to approximately 1.0.

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        z = self.decision_function(X)
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels {0, 1} for input samples using the configured threshold.

        Decision rule:
            y_hat = 1 if P(y=1|X) >= threshold else 0

        Args:
            X: Input feature DataFrame of shape (n_samples, n_features).

        Returns:
            1D NumPy array of integers {0, 1} of shape (n_samples,) containing predicted classes.

        Raises:
            RuntimeError: If called before model is fitted.
            TypeError: If X is not a pandas DataFrame or contains non-numeric data.
            ValueError: If feature columns mismatch or contain NaNs/infs.
        """
        proba = self.predict_proba(X)
        p1 = proba[:, 1]
        return (p1 >= self.threshold).astype(int)
