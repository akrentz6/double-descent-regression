import numpy as np
from scipy import linalg


def mse(y_true, y_pred):
    """
    Mean squared error between equal size vectors.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("expected equally shaped, nonempty vectors")
    return float(np.mean((y_true - y_pred) ** 2))


def design_diagnostics(X, singular_values=None):
    """
    Numerical rank and conditioning of the min(n, p) singular values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a nonempty finite matrix")
    s = linalg.svdvals(X) if singular_values is None else np.asarray(singular_values)
    tolerance = max(X.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tolerance))
    return {
        "rank": rank,
        "smallest_singular_value": float(s[-1]),
        "condition_number": float(s[0] / s[-1]) if rank == s.size else float("inf"),
    }
