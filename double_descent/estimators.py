import numpy as np
from scipy import linalg


def _validate(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty matrix")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a vector with one entry per row of X")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite")
    return X, y


def ols(X, y):
    """
    OLS for p < n (uses minimum-norm LS fit).
    """
    X, y = _validate(X, y)
    if X.shape[1] >= X.shape[0]:
        raise ValueError("ols requires p < n. Use minimum_norm for p >= n")
    return fit_linear(X, y)


def minimum_norm(X, y):
    """
    Minimum Euclidean norm of least-squares solutions.
    """
    return fit_linear(X, y)


def ridge(X, y, alpha=1.0):
    """
    Minimises ||y - X b||^2 / n + alpha ||b||^2.
    """
    return fit_linear(X, y, alpha)


def fit_linear(X, y, alpha=0.0, return_singular_values=False):
    """
    Selects OLS/minimum norm at alpha=0, or ridge at alpha>0.
    """
    X, y = _validate(X, y)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0:
        beta_hat, _, _, s = linalg.lstsq(X, y, lapack_driver="gelsd")
    else:
        U, s, Vt = linalg.svd(X, full_matrices=False)
        weights = s / (s**2 + X.shape[0] * alpha)
        beta_hat = Vt.T @ (weights * (U.T @ y))
    return (beta_hat, s) if return_singular_values else beta_hat
