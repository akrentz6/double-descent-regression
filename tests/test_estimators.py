import numpy as np

from double_descent.estimators import fit_linear, minimum_norm, ols, ridge


def test_ols():
    X = np.array([[1., 0.], [0., 1.], [1., 1.]])
    beta = np.array([2., -1.])

    assert np.allclose(ols(X, X @ beta), beta)


def test_minimum_norm():
    X = np.array([[1., 0., 1.], [0., 1., 1.]])
    y = np.array([1., 2.])
    beta_hat = minimum_norm(X, y)

    assert np.allclose(X @ beta_hat, y)
    assert np.allclose(beta_hat, [0., 1., 1.], atol=1e-12)
    assert np.allclose(fit_linear(X, y), beta_hat, atol=1e-12)


def test_square_design():
    X = np.array([[1., 2.], [3., 1.]])
    y = np.array([5., 5.])

    assert np.allclose(fit_linear(X, y), [1., 2.])


def test_rank_deficient_design():
    X = np.array([[1., 2., 3.], [2., 4., 6.]])
    y = np.array([1., 2.])

    assert np.allclose(minimum_norm(X, y), np.array([1., 2., 3.]) / 14)


def test_ridge():
    X = np.array([[1., 0.], [0., 2.], [0., 0.]])
    y = np.array([2., 3., 1.])

    # n * alpha = 1, so the diagonal denominators are 2 and 5.
    assert np.allclose(ridge(X, y, alpha=1 / 3), [1., 1.2])
    assert np.allclose(ridge(X, y, alpha=0), ols(X, y))


if __name__ == "__main__":
    test_ols()
    test_minimum_norm()
    test_square_design()
    test_rank_deficient_design()
    test_ridge()
    print("Estimator tests passed")
