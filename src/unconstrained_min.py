import numpy as np


def wolfe_condition_with_backtracking(f, x, val, gradient, direction, alpha=0.01, beta=0.5):
    step_length = 1.0
    curr_val, _, _ = f(x + step_length * direction, False)

    while curr_val > val + alpha * step_length * gradient.dot(direction):
        step_length *= beta
        curr_val, _, _ = f(x + step_length * direction, False)

    return step_length


def gradient_descent_line_search(f, x0, obj_tol, param_tol, max_iter):
    path_history = dict(path=[], values=[])
    is_min_found = False
    curr_val, curr_gradient, _ = f(x0, False)

    x = x0
    print(f'----- Gradient Descent -----')
    print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
    path_history['path'].append(x.copy())
    path_history['values'].append(curr_val)

    for i in range(max_iter):
        direction = -curr_gradient
        step_len = wolfe_condition_with_backtracking(
            f, x, curr_val, curr_gradient, direction)

        # Update x
        prev_x = x.copy()
        x += step_len * direction

        # Update value and gradient
        prev_val, prev_gradient = curr_val, curr_gradient
        curr_val, curr_gradient, _ = f(x, False)

        # Save values
        print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        path_history['path'].append(x.copy())
        path_history['values'].append(curr_val)

        # Check if reached min or converged
        if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_gradient.any():
            is_min_found = True
            break

    return x, curr_val, is_min_found, path_history


def newton_line_search(f, x0, obj_tol, param_tol, max_iter):
    path_history = dict(path=[], values=[])
    is_min_found = False
    curr_val, curr_gradient, curr_hess = f(x0, True)

    x = x0
    print(f'----- Newton -----')
    print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
    path_history['path'].append(x.copy())
    path_history['values'].append(curr_val)

    for i in range(max_iter):
        try:
            direction = np.linalg.solve(curr_hess, -curr_gradient)
        except:
            # if the hessian is not positive definite
            break

        step_len = wolfe_condition_with_backtracking(
            f, x, curr_val, curr_gradient, direction)

        # Update x
        prev_x = x.copy()
        x += step_len * direction

        # Update value and gradient
        prev_val, prev_gradient, prev_hess = curr_val, curr_gradient, curr_hess
        curr_val, curr_gradient, curr_hess = f(x, True)

        # Save values
        print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        path_history['path'].append(x.copy())
        path_history['values'].append(curr_val)

        # Check if reached min or converged
        if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_gradient.any():
            is_min_found = True
            break

    return x, curr_val, is_min_found, path_history


def BFGS_line_search(f, x0, obj_tol, param_tol, max_iter):
    path_history = dict(path=[], values=[])
    is_min_found = False
    curr_val, curr_gradient, curr_B = f(x0, True)

    x = x0
    print(f'----- BFGS -----')
    print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
    path_history['path'].append(x.copy())
    path_history['values'].append(curr_val)

    for i in range(max_iter):
        try:
            direction = np.linalg.solve(curr_B, -curr_gradient)
        except:
            # if the hessian is not positive definite
            break

        step_len = wolfe_condition_with_backtracking(
            f, x, curr_val, curr_gradient, direction)

        # Update x
        prev_x = x.copy()
        x += step_len * direction

        # Update value and gradient
        prev_val, prev_gradient = curr_val, curr_gradient
        curr_val, curr_gradient, _ = f(x, False)

        # Approximate the inverse Hessian
        s, y = x - prev_x, curr_gradient - prev_gradient
        m1 = y.dot(s)
        m2 = s.T.dot(curr_B.dot(s))
        if m1 != 0 and m2 != 0:
            m3 = curr_B.dot(s)
            curr_B = curr_B - (np.outer(m3, m3) / m2) + (np.outer(y, y) / m1)

        # Save values
        print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        path_history['path'].append(x.copy())
        path_history['values'].append(curr_val)

        # Check if reached min or converged
        if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_gradient.any():
            is_min_found = True
            break

    return x, curr_val, is_min_found, path_history


def SR1_line_search(f, x0, obj_tol, param_tol, max_iter):
    path_history = dict(path=[], values=[])
    is_min_found = False
    curr_val, curr_gradient, curr_B = f(x0, True)

    x = x0
    print(f'----- SR1 -----')
    print(f'Iteration {"0":>4}/{max_iter} - location={x}\tobj_value={curr_val}')
    path_history['path'].append(x.copy())
    path_history['values'].append(curr_val)

    for i in range(max_iter):
        try:
            direction = np.linalg.solve(curr_B, -curr_gradient)
        except:
            # if the hessian is not positive definite
            break

        step_len = wolfe_condition_with_backtracking(
            f, x, curr_val, curr_gradient, direction)

        # Update x
        prev_x = x.copy()
        x += step_len * direction

        # Update value and gradient
        prev_val, prev_gradient = curr_val, curr_gradient
        curr_val, curr_gradient, _ = f(x, False)

        # Approximate the inverse Hessian
        s, y = x - prev_x, curr_gradient - prev_gradient
        m1 = curr_B.dot(s)
        m2 = np.dot(y - m1, s)
        if m2 != 0:
            curr_B = curr_B + np.outer(y - m1, y - m1) / m2

        # Save values
        print(f'Iteration {i+1:>4}/{max_iter} - location={x}\tobj_value={curr_val}')
        path_history['path'].append(x.copy())
        path_history['values'].append(curr_val)

        # Check if reached min or converged
        if abs(curr_val - prev_val) < obj_tol or np.linalg.norm(x - prev_x) < param_tol or not curr_gradient.any():
            is_min_found = True
            break

    return x, curr_val, is_min_found, path_history
