import numpy as np

from double_descent.data import make_coefficients, sample_linear_data


def test_noiseless_data():
    rng = np.random.default_rng(0)
    beta = make_coefficients(4, rng, signal_norm=2)
    X, y = sample_linear_data(10, beta, rng, noise_std=0)

    assert X.shape == (10, 4)
    assert y.shape == (10,)
    assert np.isclose(np.linalg.norm(beta), 2)
    assert np.allclose(y, X @ beta)


def test_reproducible_data():
    beta = np.array([1., 2.])
    X1, y1 = sample_linear_data(10, beta, np.random.default_rng(3))
    X2, y2 = sample_linear_data(10, beta, np.random.default_rng(3))

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


if __name__ == "__main__":
    test_noiseless_data()
    test_reproducible_data()
    print("Data tests passed")
