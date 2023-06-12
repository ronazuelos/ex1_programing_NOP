import unittest
import numpy as np
from src.utils import plot_contour_graph, plot_values_graph
from src.unconstrained_min import gradient_descent_line_search, newton_line_search, BFGS_line_search, SR1_line_search
from tests.examples import example_quadratic1, example_quadratic2, example_quadratic3, rosenbrock, linear, exp_function


def run_test(name, f, x0=[1.0, 1.0], obj_tol=10**-12, param_tol=10**-8, max_iter=100,
             contour_x_lim=[-2, 2], contour_y_lim=[-2, 2], contour_levels=100):
  
    x0 = np.array(x0) if isinstance(x0, list) else x0
    gd_x, gd_val, gd_found, gd_hist = gradient_descent_line_search(f, x0.copy(), obj_tol=obj_tol, param_tol=param_tol, max_iter=max_iter)
    n_x, n_val, n_found, n_hist = newton_line_search(f, x0.copy(), obj_tol=obj_tol, param_tol=param_tol, max_iter=min(max_iter, 100))
    bfgs_x, bfgs_val, bfgs_found, bfgs_hist = BFGS_line_search(f, x0.copy(), obj_tol=obj_tol, param_tol=param_tol, max_iter=min(max_iter, 100))
    sr1_x, sr1_val, sr1_found, sr1_hist = SR1_line_search(f, x0.copy(), obj_tol=obj_tol, param_tol=param_tol, max_iter=min(max_iter, 100))

    print("----- Final Results -----")
    print(f'{"Gradient Descent":>16} - {len(gd_hist["path"])-1} Iterations, Final location: ({gd_x[0]},{gd_x[1]}), Final value: {gd_val}, Minimum: {gd_found}' )
    print(f'{"Newton":>16} - {len(n_hist["path"])-1} Iterations, Final location: ({n_x[0]},{n_x[1]}), Final value: {n_val}, Minimum: {n_found}' )
    print(f'{"BFGS":>16} - {len(bfgs_hist["path"])-1} Iterations, Final location: ({bfgs_x[0]},{bfgs_x[1]}), Final value: {bfgs_val}, Minimum: {bfgs_found}' )
    print(f'{"SR1":>16} - {len(sr1_hist["path"])-1} Iterations, Final location: ({sr1_x[0]},{sr1_x[1]}), Final value: {sr1_val}, Minimum: {sr1_found}' )
    print()

    values_dict = {
        'Gradient Descent': gd_hist['values'],
        'Newton': n_hist['values'],
        'BFGS': bfgs_hist['values'],
        'SR1': sr1_hist['values']
    }
    plot_values_graph(values_dict, title=f'Function values per Iteration of {name}')

    print()

    paths_dict = {
        'Gradient Descent': np.array(gd_hist['path']),
        'Newton': np.array(n_hist['path']),
        'BFGS': np.array(bfgs_hist['path']),
        'SR1': np.array(sr1_hist['path'])
    }
    plot_contour_graph(f, contour_x_lim, contour_y_lim, paths=paths_dict,
                        levels=contour_levels, title=f'Contour of {name} Objective Function')


class TestMinimize(unittest.TestCase):

    def test_quadratic1(self):
        run_test('Quadratic #1', example_quadratic1)

    def test_quadratic2(self):
        run_test('Quadratic #2', example_quadratic2)

    def test_quadratic3(self):
        run_test('Quadratic #3', example_quadratic3)

    def test_rosenbrock(self):
        run_test('Rosenbrock', rosenbrock, x0=[-1., 2.], max_iter=10000, contour_y_lim=[-2, 5])

    def test_linear(self):
        run_test('Linear', linear, contour_x_lim=[-300, 2], contour_y_lim=[-300, 2])

    def test_exp(self):
        run_test('Exponential', exp_function, contour_x_lim=[-1, 1], contour_y_lim=[-1, 1], contour_levels=50)
