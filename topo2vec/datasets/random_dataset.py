from typing import List

from torch import tensor

from topo2vec.datasets.random_dataset_builder import DatasetBuilder
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


class RandomDataset(MultiRadiusDataset):
    def __init__(self, num_points: int, radii: List[int] = [10]):
        super().__init__(radii)
        dataset_builder = DatasetBuilder()
        random_points = dataset_builder.get_random_dataset(num_points)
        self.add_points_as_patches_to_actual_patches(random_points)


    def __getitem__(self, index):
        return (self.actual_patches[index], tensor([0.]))

    def __len__(self):
        return len(self.actual_patches)
