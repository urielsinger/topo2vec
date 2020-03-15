import random

from torch.utils.data import DataLoader, RandomSampler

from topo2vec.common.visualizations import plot_n_np_arrays
from topo2vec.datasets.classification_dataset import ClassificationDataset
import numpy as np

from topo2vec.data_locations import *

NUM_TO_SHOW = 16
dataset = list(dataset)
n = len(peaks)
random_dataset = random.choices(dataset, k=NUM_TO_SHOW)
random_dataset = [random_peak[0].reshape(21,21) for random_peak in random_dataset]
plot_n_np_arrays(random_dataset)
