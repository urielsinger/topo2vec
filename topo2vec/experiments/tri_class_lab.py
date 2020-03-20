from shapely.geometry import Polygon, Point
from torch import nn

from topo2vec.CONSTANTS import N49_E05_RIVERS, N49_E05_CLIFFS, N49_E05_STREAMS
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.experiments.classification_lab import ClassificationLab

num_classes = 1
train_half = Polygon([Point(5, 49), Point(5, 49.5), Point(6, 49.5),
                      Point(6, 49), Point(5, 49)])
validation_half = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                           Point(6, 49.5), Point(5, 49.5)])

class_paths = [N49_E05_RIVERS, N49_E05_CLIFFS, N49_E05_STREAMS]
class_paths = [N49_E05_RIVERS, N49_E05_CLIFFS, N49_E05_RIVERS]
class_paths = [N49_E05_CLIFFS, N49_E05_STREAMS]

class_names = ['River', 'Cliff', 'Stream']



class TriClassLab(ClassificationLab):
    def __init__(self):
        super().__init__()
        self.model_hyperparams.update({
            'arch': ['simpleconvnet'],
            'num_classes': [len(class_paths)],
            'loss_func': [nn.CrossEntropyLoss()]
        })

    def _generate_datasets(self, radii, total_dataset_size):
        size_train = int(0.8 * total_dataset_size)
        size_val = int(0.2 * total_dataset_size)
        train_set = SeveralClassesDataset(radii, train_half, class_paths, class_names)
        val_set = SeveralClassesDataset(radii, validation_half, class_paths, class_names)
        return train_set, val_set

lab = TriClassLab()
lab.run()
