from typing import List

from shapely.geometry import Polygon, Point
from torch import nn

from topo2vec.constants import N45_50_E5_15_CLIFFS, \
    N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.experiments.classification_lab import ClassificationLab

num_classes = 1
train_half = Polygon([Point(5, 45), Point(5, 50), Point(10, 50),
                      Point(10, 45), Point(5, 45)])
validation_half = Polygon([Point(10, 50), Point(10, 45), Point(15, 45),
                           Point(15, 50), Point(10, 50)])

class_paths = [N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS]

class_names = ['Cliff', 'River', 'Peak']


class TriClassLab(ClassificationLab):
    def __init__(self):
        super().__init__()
        self.model_hyperparams.update({
            'arch': ['simpleconvnet'],
            'num_classes': [len(class_paths)],
            'loss_func': [nn.CrossEntropyLoss()]
        })

    def _generate_datasets(self, radii: List[int], total_dataset_size: int):
        '''

        Args:
            radii: the radii to be used
            total_dataset_size: the max dataset size asked for

        Returns: the train set and test sets generated

        '''
        size_train = int(0.8 * total_dataset_size)
        size_val = int(0.2 * total_dataset_size)
        train_set = SeveralClassesDataset(radii, train_half, size_train, class_paths, class_names)
        val_set = SeveralClassesDataset(radii, validation_half, size_val, class_paths, class_names)
        return train_set, val_set


lab = TriClassLab()
lab.run()
