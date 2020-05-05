from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def get_dataset_as_tensor(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    '''

    Args:
        dataset:

    Returns: dataset as a tensor

    '''
    dataset_length = len(dataset)
    return get_random_part_of_dataset(dataset, dataset_length, shuffle=False)


def get_random_part_of_dataset(dataset: Dataset, size_wanted: int, shuffle=True) -> Tuple[Tensor, Tensor]:
    '''
    Args:
        dataset:
        size_wanted: number of rows wanted in the generated Tensor
        shuffle: get shuffled from the dataset(randomized) or not.
    Returns: a tuple: (the images tensor, the labels tensor)

    '''

    data_loader = DataLoader(dataset, shuffle=shuffle, num_workers=0, batch_size=size_wanted)
    batch = next(iter(data_loader))
    return batch