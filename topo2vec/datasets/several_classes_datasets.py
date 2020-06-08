import logging
from typing import List

from shapely.geometry import Polygon
from torch.utils.data import ConcatDataset, Subset

from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.multi_radius_dataset import MultiRadiiDataset


class SeveralClassesDataset(MultiRadiiDataset):
    '''
    A dataset that is a union of some class-datasets, to be used specifically in
    multiclaass/binary classification purposes
    '''
    def __init__(self, original_radiis: List[int], outer_polygon: Polygon, wanted_size: int,
                 class_paths: List[str], class_names: List[str], dataset_type_name: str,
                 radii: List[int] = None, random_seed=None):
        '''

        Args:
            original_radiis: The radiis
            outer_polygon:
            wanted_size: wnted size of the dataset. will be rounded to the smalles larger multiplier of the class_nmum
            class_paths: The paths to the clasess' points
            class_names: the classes name in order
            dataset_type_name: a name for the dataset. shopuld be unique for caching.
            radii: the firs int inside the list is the radius of each patch in the final patches inside the class
            random_seed: a random seed for making the process deterministic
        '''
        all_datasets = []
        self.class_names = class_names
        self.class_names_to_indexes = {}
        for i in range(len(self.class_names)):
            self.class_names_to_indexes[self.class_names[i]] = i
        class_wanted_size = int(wanted_size / (len(class_names)*len(original_radiis))) + 1
        for i, class_path in enumerate(class_paths):
            class_dataset = ClassDataset(class_path, i, original_radiis=original_radiis,
                                         wanted_size=class_wanted_size,
                                         outer_polygon=outer_polygon,
                                         dataset_type_name=dataset_type_name,
                                         radii=radii, seed=random_seed)

            # logging.info(f'{dataset_type_name} dataset: {len(class_dataset)} {class_names[i]} points')
            all_datasets.append(class_dataset)

        size = min([len(dataset) for dataset in all_datasets])
        wanted_indices = list(range(0, size, 1))
        all_datasets = [Subset(dataset, wanted_indices) for dataset in all_datasets]
        self.combined_dataset = ConcatDataset(all_datasets)


    def class_name_to_index(self, class_name: str) -> int:
        '''

        Args:
            class_name: the name

        Returns: the index of the class in the self.combined_dataset

        '''
        return self.class_names_to_indexes[class_name]

    def __getitem__(self, index):
        return self.combined_dataset[index]

    def __len__(self):
        return len(self.combined_dataset)
