import random
import time

import numpy
import torch
from torch.backends import cudnn

from topo2vec.constants import BASE_LOCATION
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.datasets.random_dataset import RandomDataset

CLASSES_POINTS_FOLDER = BASE_LOCATION + f'data/overpass_classes_data/cliffs_(45,5,50,15).json'
from topo2vec.background import VALIDATION_HALF, TRAIN_HALF

original_radiis = [[9, 9, 9]]
random_seed = 888
radii = [8]
classification_dataset = ClassDataset(CLASSES_POINTS_FOLDER, 1,
                                      original_radiis=original_radiis, wanted_size=10,
                                      outer_polygon=VALIDATION_HALF, radii=radii,
                                      dataset_type_name='calls_for_random' + str(time.time()),
                                      seed=random_seed)

wanted_indices = list(range(0, int(10 / 2), 1))
classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
random_dataset = RandomDataset(len(classification_dataset), original_radiis, VALIDATION_HALF, radii=radii,
                               seed=random_seed)
random_seed2 = 900
random.seed(random_seed2)
torch.manual_seed(random_seed2)
cudnn.deterministic = True
numpy.random.seed(random_seed2)
one = OneVsRandomDataset(original_radiis, 10, TRAIN_HALF,
                         CLASSES_POINTS_FOLDER,
                         # f'scale_exp_{self.scale_exp_class_name}_vs_random_train',
                         radii=radii, random_seed=random_seed)
#everything here was the same no matter what i've done
