from typing import List

from torch import nn, optim
from torch.utils.data import ConcatDataset

from topo2vec.constants import N45_50_E5_15_CLIFFS, \
    N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS, VALIDATION_HALF, TRAIN_HALF, N45_50_E5_15_STREAMS, CLIFFS_TEST, \
    RIVERS_TEST, PEAKS_TEST, POINT_TO_SEARCH_SIMILAR
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.random_dataset import RandomDataset
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
    size_random = 100000
    random_dataset = RandomDataset(size_random, radii, VALIDATION_HALF)
    size_typical_images = 5
    typical_images_dataset = ClassDataset(POINT_TO_SEARCH_SIMILAR, 1, radii, size_typical_images,VALIDATION_HALF)
    random_dataset = ConcatDataset([typical_images_dataset, random_dataset])
    return train_set, val_set, test_dataset, random_dataset, typical_images_dataset

model_hyperparams = {
            'radii': [[8, 16, 24], [16]],
            'learning_rate': [1e-4],
            'total_dataset_size': [1000, 75000],
            'max_epochs': [10, 100],
            'optimizer_cls': [optim.Adam],
            'datasets_generator': [_generate_datasets],
            'arch': ['basicconvnetlatent'],
            'num_classes': [len(class_paths)],
            'loss_func': [nn.CrossEntropyLoss()],
            'name': ['quad_class']
        }
lab = ClassificationTask(model_hyperparams)
lab.run()
