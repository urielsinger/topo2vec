from typing import List

from shapely.geometry import Polygon
from torch.utils.data import ConcatDataset, Subset

from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


class SeveralClassesDataset(MultiRadiusDataset):
    def __init__(self, radii: List[int], outer_polygon: Polygon, wanted_size: int,
                 class_paths: List[str], class_names: List[str], dataset_type_name: str):
        all_datasets = []
        self.class_names = class_names
        self.class_names_to_indexes = {}
        for i in range(len(self.class_names)):
            self.class_names_to_indexes[self.class_names[i]] = i
        class_wanted_size = int(wanted_size / len(class_names)) + 1
        for i, class_path in enumerate(class_paths):
            class_dataset = ClassDataset(class_path, i, radii=radii,
                                         wanted_size=class_wanted_size,
                                         outer_polygon=outer_polygon,
                                         dataset_type_name=dataset_type_name)

            print(f'{dataset_type_name} dataset: {len(class_dataset)} {class_names[i]} points')
            all_datasets.append(class_dataset)

        size = min([len(dataset) for dataset in all_datasets])
        wanted_indices = list(range(0, size, 1))
        all_datasets = [Subset(dataset, wanted_indices) for dataset in all_datasets]
        self.combined_dataset = ConcatDataset(all_datasets)

    def class_name_to_index(self, class_name: str) -> int:
        return self.class_names_to_indexes[class_name]

    def __getitem__(self, index):
        return self.combined_dataset[index]

    def __len__(self):
        return len(self.combined_dataset)
