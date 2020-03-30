import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


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


def plot_n_np_arrays(data_arrays, fig_size=(14, 14), title='', x_label='lon', y_label='lat',
                     lines_number=1, titles: List[str] = None):
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
    n_cols = int(n / n_rows)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for i, ax in enumerate(axs.flat):
        if i < n:
            if titles is not None:
                ax.set_title(titles[i])
            ax.set_xlabel = x_label
            ax.set_ylabel = y_label
            ax.set_title = f'k = {i}'
            ax.imshow(data_arrays[i])

    if title != '':
        fig.suptitle(title, fontsize=22)

    plt.show()


def plot_n_np_arrays_one_row(data_arrays, fig_size=(12, 12), title='', x_label='lon', y_label='lat'):
    '''

    Args:
        data_arrays:
        fig_size:
        title:
        x_label:
        y_label:

    Returns: plots the data_arrays in one row

    '''
    plot_n_np_arrays(data_arrays=data_arrays, fig_size=fig_size, title=title,
                     x_label=x_label, y_label=y_label, lines_number=1)


def get_dataset_as_tensor(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    '''

    Args:
        dataset:

    Returns: dataset as a tensor

    '''
    dataset_length = len(dataset)
    return get_random_part_of_dataset(dataset, dataset_length, shuffle=False)


def get_random_part_of_dataset(dataset: Dataset, size_wanted: int, shuffle=True) -> Tuple[Tensor, Tensor]:
    '''
    Args:
        dataset:
        size_wanted: number of rows wanted in the generated Tensor
        shuffle: get shuffled from the dataset(randomized) or not.
    Returns: a tuple: (the images tensor, the labels tensor)

    '''

    data_loader = DataLoader(dataset, shuffle=shuffle, num_workers=0, batch_size=size_wanted)
    images_as_tensor, y = next(iter(data_loader))
    return images_as_tensor, y


def get_grid_sample_images_at_indexes(all_images: torch.tensor, indexes: torch.tensor,
                                      randomize=False, number_to_log: int = 5):
    '''

    Args:
        all_images:  all the images to sample from
        indexes: the indexes of the rows which are relevant to smaple from -
        ignore all others
        randomize:
        number_to_log:

    Returns:  a grid of all the appropriate patches.

    '''
    images = all_images[indexes]
    images = convert_multi_radius_tensor_to_printable(images)
    return get_grid_sample_images(images, randomize, number_to_log)


def get_grid_sample_images(images: Tensor, randomize: bool, number_to_log: int) -> Tensor:
    '''

    Args:
        images: all the images to sample from
        randomize: to sample or to return the first ones
        number_to_log: number of images to put in the grid

    Returns: a grid of all the appropriate patches.

    '''
    if randomize:
        num_images = images.shape[0]
        rand_indexes = [random.randint(0, num_images - 1) for i in range(number_to_log)]
        sample_imgs = [images[rand_num] for rand_num in rand_indexes]
    else:
        sample_imgs = images[0:number_to_log]
    grid = torchvision.utils.make_grid(sample_imgs)
    return grid


def convert_multi_radius_tensor_to_printable(tesor: Tensor) -> Tensor:
    '''

    Args:
        tesor: The tensor of the multi-radiii examples
        with shape: (num_lines, len(radii), h, w)

    Returns: a vector of shape (num_lines, 1, len(radii)*h, w)

    '''
    num_samples_in_dataset, _, _, w = tesor.shape
    printable_tensor = tesor.view(num_samples_in_dataset, 1, -1, w)
    return printable_tensor
