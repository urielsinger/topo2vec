from typing import List

import torch
from torch.utils.data import ConcatDataset

from topo2vec.constants import N49_E05_STREAMS
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.random_dataset import RandomDataset


class OneVsRandomDataset:
    def __init__(self, original_radiis: List[int], size: int, outer_polygon,
                 class_path=N49_E05_STREAMS, class_label=1, radii=None):
        '''
        init a dataset for the one class vs all task.
        Args:
            radii: original_radiis list
            size: max size of the wanted dataset
            outer_polygon: the polygon to keep the points from
            class_path:
            class_label:
        '''
        classification_dataset = ClassDataset(class_path, class_label,
                                              original_radiis=original_radiis, wanted_size=int(size),
                                              outer_polygon=outer_polygon, radii=radii, dataset_type_name='cllas_for_random')

        wanted_indices = list(range(0, int(size / 2), 1))
        classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
        random_dataset = RandomDataset(len(classification_dataset), original_radiis, outer_polygon, radii=radii)
        combined_dataset = ConcatDataset([classification_dataset, random_dataset])
        print(f'size: {len(combined_dataset)}')
        self.combined_dataset = combined_dataset

    def __getitem__(self, index):
        return self.combined_dataset[index]

    def __len__(self):
        return len(self.combined_dataset)