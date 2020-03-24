from torch import nn, optim

from topo2vec.constants import TRAIN_HALF, VALIDATION_HALF, N45_50_E5_15_PEAKS
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.experiments.classification_task import ClassificationTask

num_classes = 1


def _generate_datasets(radii, total_dataset_size):
    '''

    Args:
        radii: The radii to be used in the dataset
        total_dataset_size: the size of the dataset needed

    Returns:

    '''
    size_train = int(0.8 * total_dataset_size)
    size_val = int(0.2 * total_dataset_size)
    train_set = OneVsRandomDataset(radii, size_train, TRAIN_HALF, class_path=N45_50_E5_15_PEAKS)
    val_set = OneVsRandomDataset(radii, size_val, VALIDATION_HALF, class_path=N45_50_E5_15_PEAKS)
    return train_set, val_set, None, None

model_hyperparams = {
            'radii': [[8, 16, 24], [24]],
            'learning_rate': [1e-4, 1e-5, 1e-6],
            'total_dataset_size': [10000, 30000],
            'max_epochs': [100],
            'optimizer_cls': [optim.Adam],
            'datasets_generator': [_generate_datasets],
            'arch': ['simpleconvnet'],
            'num_classes': [2],
            'loss_func': [nn.CrossEntropyLoss()],
            'name': ['streams_vs_all']
        }
lab = ClassificationTask(model_hyperparams)
lab.run()

