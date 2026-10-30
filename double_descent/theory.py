import numpy as np


def asymptotic_risk(gamma, noise_std=0.5, signal_norm=1.0):
    """
    Returns total, squared bias, and variance for scalar or array p/n.
    """
    gamma = np.asarray(gamma, dtype=float)
    bias = signal_norm**2 * np.maximum(1 - 1 / gamma, 0)
    variance = np.zeros_like(gamma)
    if noise_std > 0:
        variance.fill(np.inf)
        below, above = gamma < 1, gamma > 1
        variance[below] = noise_std**2 * gamma[below] / (1 - gamma[below])
        variance[above] = noise_std**2 / (gamma[above] - 1)
    return {
        "total": bias + variance,
        "bias": bias,
        "variance": variance
    }


def finite_sample_risk(n, p, noise_std=0.5, signal_norm=1.0):
    """
    Expected excess risk for i.i.d. N(0, 1) designs and integer n, p.
    """
    p = np.asarray(p, dtype=float)
    bias = signal_norm**2 * np.maximum(1 - n / p, 0)
    variance = np.zeros_like(p)
    if noise_std > 0:
        variance.fill(np.inf)
        below, above = p < n - 1, p > n + 1
        variance[below] = noise_std**2 * p[below] / (n - p[below] - 1)
        variance[above] = noise_std**2 * n / (p[above] - n - 1)
    return {
        "total": bias + variance,
        "bias": bias,
        "variance": variance
    }
