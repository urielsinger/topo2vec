import math
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch import Tensor
import itertools

import numpy as np

from topo2vec.constants import CACHE_BASE_DIR


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
    plot_n_np_arrays([data_array], title=title, x_label=x_label, y_label=y_label)


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
        num_images = len(images)
        rand_indexes = [random.randint(num_images - 1) for i in range(number_to_log)]
        sample_imgs = [images[rand_num] for rand_num in rand_indexes]
    else:
        sample_imgs = images[:number_to_log]
    grid = torchvision.utils.make_grid(sample_imgs)
    return grid


def convert_multi_radius_tensor_to_printable(tensor: Tensor) -> Tensor:
    '''

    Args:
        tensor: The tensor of the multi-radiii examples
        with shape: (num_lines, len(original_radiis), h, w)

    Returns: a vector of shape (num_lines, 1, len(original_radiis)*h, w)

    '''
    num_samples_in_dataset, _, _, w = tensor.shape
    printable_tensor = tensor.view(num_samples_in_dataset, 1, -1, w)
    return printable_tensor

def convert_multi_radius_ndarray_to_printable(array: np.ndarray, dir = True) -> np.ndarray:
    '''

    Args:
        tensor: The tensor of the multi-radiii examples
        with shape: (num_lines, len(original_radiis), h, w)
        dir = True -> vertically
        dir = False -> horizontally

    Returns: a vector of shape (num_lines, 1, len(original_radiis)*h, w)

    '''
    num_samples_in_dataset, _, h, w = array.shape
    if dir:
        printable_ndarray = array.reshape(num_samples_in_dataset, -1, w)
    else:
        printable_ndarray = array.transpose((0,1,3,2)).reshape(num_samples_in_dataset, -1, h).transpose((0,2,1))
    return printable_ndarray

def convert_multi_radius_list_to_printable(images_list:List[np.ndarray], dir = True) -> np.ndarray:
    '''

    Args:
        tensor: The tensor of the multi-radiii examples
        with shape: (num_lines, len(original_radiis), h, w)
        dir = True -> vertically
        dir = False -> horizontally

    Returns: a vector of shape (num_lines, 1, len(original_radiis)*h, w)

    '''
    return convert_multi_radius_ndarray_to_printable(np.stack(images_list), dir)

def plot_confusion_matrix(cm: object,
                          target_names: object,
                          title: object = 'Confusion matrix',
                          cmap: object = None,
                          normalize: object = True) -> object:
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    fig, ax = plt.subplots()
    cm = cm
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)
    values_format = '.2g'

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_[i, j] = ax.text(j, i,
                               format(cm[i, j], values_format),
                               ha="center", va="center",
                               color=color)

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=target_names,
           yticklabels=target_names,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=None)
    return ax, fig

def plot_to_image(figure):
    tmp_path = os.path.join(CACHE_BASE_DIR, 'tmp.tif')
    plt.savefig(tmp_path, format=format('tif'), figure=figure)
    plt.close(figure)
    im = Image.open(tmp_path).convert('RGB')
    image = np.array(im)
    return image

def np_array_to_png_list(np_array:np.ndarray) ->List:
    images_list = []
    for im in np_array:
        im_rescaled = 255.0 /im.max() * (im-im.min()).astype(np.uint8)
        image = Image.fromarray(im_rescaled)
        images_list.append(image)
    return images_list

def plot_list_to_tensorboard(feature_extractor, closest_images_list_image_space: List[Tensor], number_to_log: int,
                             title: str):
    '''

    Args:
        closest_images_list_image_space: the images
        number_to_log:
        title:

    Returns:

    '''
    for i, images_per_point in enumerate(closest_images_list_image_space):
        images_per_point = convert_multi_radius_tensor_to_printable(images_per_point)
        grid = get_grid_sample_images(images_per_point, False, number_to_log)
        feature_extractor.logger.experiment.add_image(f'{title}_{i}', grid, 0)