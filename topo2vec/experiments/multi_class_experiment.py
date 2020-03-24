from typing import List

from torch import nn, optim

from topo2vec.constants import N45_50_E5_15_CLIFFS, \
    N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS, VALIDATION_HALF, TRAIN_HALF, N45_50_E5_15_STREAMS, CLIFFS_TEST, \
    RIVERS_TEST, PEAKS_TEST
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.experiments.classification_task import ClassificationTask

class_paths = [N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS]#], N45_50_E5_15_STREAMS]
class_names = ['Cliff', 'River', 'Peak']#, 'Stream']

class_paths_test = [CLIFFS_TEST, RIVERS_TEST, PEAKS_TEST]

def _generate_datasets(radii: List[int], total_dataset_size: int):
    '''

    Args:
        radii: the radii to be used
        total_dataset_size: the max dataset size asked for

    Returns: the train set and test sets generated

    '''
    size_train = int(0.8 * total_dataset_size)
    size_val = int(0.2 * total_dataset_size)
    size_test = 55
    train_set = SeveralClassesDataset(radii, TRAIN_HALF, size_train, class_paths, class_names)
    val_set = SeveralClassesDataset(radii, VALIDATION_HALF, size_val, class_paths, class_names)
    test_dataset = SeveralClassesDataset(radii, VALIDATION_HALF, size_test, class_paths_test, class_names)
    return train_set, val_set, test_dataset

model_hyperparams = {
            'radii': [[8], [10], [8, 16], [16], [8, 16, 24], [20], [24]],
            'learning_rate': [1e-4, 1e-5, 1e-6],
            'total_dataset_size': [75000],
            'max_epochs': [100],
            'optimizer_cls': [optim.Adam],
            'datasets_generator': [_generate_datasets],
            'arch': ['simpleconvnet'],
            'num_classes': [len(class_paths)],
            'loss_func': [nn.CrossEntropyLoss()],
            'name': ['quad_class']
        }
lab = ClassificationTask(model_hyperparams)
lab.run()
