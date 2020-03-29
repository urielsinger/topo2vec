from typing import List

from shapely.geometry import Polygon
from torch.utils.data import ConcatDataset, Subset

from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


class SeveralClassesDataset(MultiRadiusDataset):
    def __init__(self, radii: List[int], outer_polygon: Polygon, wanted_size: int,
                 class_paths: List[str], class_names: List[str]):
        all_datasets = []
        class_wanted_size = int(wanted_size / len(class_names)) + 1
        for i, class_path in enumerate(class_paths):
            class_dataset = ClassDataset(class_path, i, radii=radii,
                                         wanted_size=class_wanted_size,
                                         outer_polygon=outer_polygon)

            print(f'{len(class_dataset)} {class_names[i]} points')
            all_datasets.append(class_dataset)

        size = min([len(dataset) for dataset in all_datasets])
        wanted_indices = list(range(0, size, 1))
        all_datasets = [Subset(dataset, wanted_indices) for dataset in all_datasets]
        self.combined_dataset = ConcatDataset(all_datasets)

    def __getitem__(self, index):
        return self.combined_dataset[index]

    def __len__(self):
        return len(self.combined_dataset)
