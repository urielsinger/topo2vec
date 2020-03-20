from typing import List

import torch
from torch.utils.data import ConcatDataset

from topo2vec.CONSTANTS import N49_E05_STREAMS
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset
from topo2vec.datasets.random_dataset import RandomDataset


class OneVsRandomDataset:
    def __init__(self, radii: List[int], size, outer_polygon, class_path=N49_E05_STREAMS,
                 class_label=1):
        classification_dataset = ClassDataset(class_path, class_label,
                                              radii=radii,
                                              outer_polygon=outer_polygon)
        if size < 2 * len(classification_dataset):
            wanted_indices = list(range(0, int(size / 2), 1))
        else:
            print('asked for too large dataset')
            print(f'the size you\'re able:{2 * len(classification_dataset)}')
            print(f'and you asked for:{size}')

        classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
        random_dataset = RandomDataset(len(classification_dataset), radii, outer_polygon)
        combined_dataset = ConcatDataset([classification_dataset, random_dataset])
        print(f'size: {len(combined_dataset)}')
        self.combined_dataset = combined_dataset

    def __getitem__(self, index):
        return self.combined_dataset[index]

    def __len__(self):
        return len(self.combined_dataset)