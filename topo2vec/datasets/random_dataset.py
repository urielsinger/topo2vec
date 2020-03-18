from typing import List

from torch import tensor

from topo2vec.datasets.random_dataset_builder import DatasetBuilder
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


class RandomDataset(MultiRadiusDataset):
    def __init__(self, num_points: int, radii: List[int], *args):
        '''

        Args:
            num_points: mi,ber of random points to generate
            radii: radii needed around each point.
        '''
        super().__init__(radii)
        dataset_builder = DatasetBuilder(*args)
        random_points = dataset_builder.get_random_dataset(num_points)

        self.add_points_as_patches_to_actual_patches(random_points)
        if self.use_masks:
            self.add_points_as_patches_to_mask_patches(random_points)



    def __getitem__(self, index):
        '''
        a simple getitem, but of course if the mask is used,
        returns it too.
        Args:
            index:

        Returns:

        '''
        #if self.mask_patches is not None and self.use_masks:
        #    return (self.actual_patches[index], self.mask_patches[index], tensor([0.]))
        return (self.actual_patches[index], tensor([0.]))



