import math
from typing import List

import matplotlib.pyplot as plt

def plot_np_array(data_array, title='', x_label='lon', y_label='lat'):
    """
    plots the patch given as the np array
    Args:
        data_array: the array to plot
        title: title for the graph
        x_label:
        y_label:

    Returns: Nothing

    """
    plt.imshow(data_array)
    if title != '':
        plt.title(title)
    plt.x_label = x_label
    plt.y_label = y_label
    plt.show()

def plot_n_np_arrays(data_arrays, fig_size = (14,14), title='', x_label='lon', y_label='lat',
                     lines_number=1, titles :List[str] = None):
    """
    plots the patchs given as the np array
    Args:
        data_arrays: a list of np arrays to plot
        title: title for the graph
        x_label:
        y_label:

    Returns: Nothing

    """
    n = len(data_arrays)
    n_rows = lines_number
    n_cols = int(n/n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize = fig_size)
    for i, ax in enumerate(axs.flat):
        if i < n:
            if titles is not None:
                ax.set_title(titles[i])
            ax.set_xlabel = x_label
            ax.set_ylabel = y_label
            ax.set_title = f'k = {i}'
            ax.imshow(data_arrays[i])

    if title != '':
        fig.suptitle(title, fontsize = 22)

    plt.show()

def plot_n_np_arrays_one_row(data_arrays, fig_size=(12, 12), title='', x_label='lon', y_label='lat'):
    plot_n_np_arrays(data_arrays= data_arrays, fig_size=fig_size, title=title,
                     x_label=x_label, y_label=y_label, lines_number=1)