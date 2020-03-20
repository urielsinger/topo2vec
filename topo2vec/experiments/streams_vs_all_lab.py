from shapely.geometry import Polygon, Point

from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.experiments.classification_lab import ClassificationLab

num_classes = 1
train_half = Polygon([Point(5, 49), Point(5, 49.5), Point(6, 49.5),
                      Point(6, 49), Point(5, 49)])
validation_half = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                           Point(6, 49.5), Point(5, 49.5)])


class StreamsVsAllLab(ClassificationLab):
    def __init__(self):
        super().__init__()
        self.model_hyperparams.update({
            'arch': ['simpleconvnet'],
            'num_classes': [1]
        })

    def _generate_datasets(self, radii, total_dataset_size):
        size_train = int(0.8 * total_dataset_size)
        size_val = int(0.2 * total_dataset_size)
        train_set = OneVsRandomDataset(radii, size_train, train_half)
        val_set = OneVsRandomDataset(radii, size_val, validation_half)
        return train_set, val_set

lab = StreamsVsAllLab()
lab.run()
