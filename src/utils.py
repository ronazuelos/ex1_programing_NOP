import matplotlib.pyplot as plt
import numpy as np


def plot_contour_graph(f, x_lim, y_lim, title, paths={}, levels=100):
    x = np.linspace(x_lim[0], x_lim[1])
    y = np.linspace(y_lim[0], y_lim[1])

    xs, ys = np.meshgrid(x, y)
    f_vals = np.vectorize(lambda x1, x2: f(
        np.array([x1, x2]), False)[0])(xs, ys)

    fig, ax = plt.subplots(1, 1)
    contour = ax.contourf(x, y, f_vals, levels)
    fig.colorbar(contour)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    if len(paths):
        for name, path in paths.items():
            plt.plot(path[:, 0], path[:, 1], label=name)
        plt.legend()
    plt.show()


def plot_values_graph(values_dict, title):
    fig, ax = plt.subplots(1, 1)
    for name, values in values_dict.items():
        x = np.linspace(0, len(values)-1, len(values))
        ax.plot(x, values, marker='.', label=name)
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Function values')
    plt.legend()
    plt.show()
