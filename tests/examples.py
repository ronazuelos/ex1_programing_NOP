import numpy as np


def example_quadratic1(x, is_hessian_needed):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if is_hessian_needed else None
    return f, g, h


def example_quadratic2(x, is_hessian_needed):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T.dot(Q).dot(x)
    g = 2 * Q.dot(x)
    h = 2 * Q if is_hessian_needed else None
    return f, g, h


def example_quadratic3(x, is_hessian_needed):
    Q1 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q_final = Q1.T.dot(Q2).dot(Q1)

    f = x.T.dot(Q_final).dot(x)
    g = 2 * Q_final.dot(x)
    h = 2 * Q_final if is_hessian_needed else None
    return f, g, h


def rosenbrock(x, is_hessian_needed):
    f = 100 * ((x[1] - (x[0] ** 2)) ** 2) + ((1 - x[0]) ** 2)
    g = np.array([-400 * x[0] * x[1] + 400 * (x[0] ** 3) + 2 * x[0] - 2,
                  200 * x[1] - 200 * (x[0] ** 2)])
    h = None
    if is_hessian_needed:
        h = np.array([[-400 * x[1] + 1200 * (x[0] ** 2) + 2, -400 * x[0]],
                      [-400 * x[0],         200]])
    return f, g, h


def linear(x, is_hessian_needed):
    a = np.array([3, 3])
    f = a.T.dot(x)
    g = a
    h = None
    if is_hessian_needed:
        h = np.zeros((2, 2))
    return f, g, h


def exp_function(x, is_hessian_needed):
    f = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] -
                                               3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    h = None
    if is_hessian_needed:
        h = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                       3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                      [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                       9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
    return f, g, h
