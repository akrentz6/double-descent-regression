import numpy as np


def make_coefficients(p, rng, signal_norm=1.0):
    """
    Draw a random direction with ||beta|| = signal_norm.
    """
    if p < 1:
        raise ValueError("p must be positive")
    if not np.isfinite(signal_norm) or signal_norm < 0:
        raise ValueError("signal_norm must be finite and nonnegative")
    beta = rng.standard_normal(p)
    return signal_norm * beta / np.linalg.norm(beta)


def sample_linear_data(n, beta, rng, noise_std=1.0):
    """
    Returns X, y where y = X beta + N(0, noise_std^2).
    """
    beta = np.asarray(beta, dtype=float)
    if n < 1:
        raise ValueError("n must be positive")
    if beta.ndim != 1 or beta.size == 0 or not np.all(np.isfinite(beta)):
        raise ValueError("beta must be a nonempty finite vector")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    X = rng.standard_normal((n, beta.size))
    noise = rng.standard_normal(n)
    return X, X @ beta + noise_std * noise
