from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def get_dataset_as_tensor(dataset: Dataset, shuffle=False) -> Tuple[Tensor, Tensor]:
    '''

    Args:
        dataset:

    Returns: dataset as a tensor

    '''
    dataset_length = len(dataset)
    return get_random_part_of_dataset(dataset, dataset_length, shuffle=shuffle)


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

def get_dataset_latent_space_as_np(feature_extractor, images_as_tensor: Tensor):
    '''

    Args:
        images_as_tensor: the images to put in the feature_extractor

    Returns: an np.ndarray

    '''
    images_as_tensor = images_as_tensor.float()
    if feature_extractor.hparams.use_gpu:
        images_as_tensor = images_as_tensor.cuda()
    _, images_latent_as_tensor = feature_extractor.forward(images_as_tensor)
    images_latent_as_np = images_latent_as_tensor.data
    if feature_extractor.hparams.use_gpu:
        images_latent_as_np = images_latent_as_np.cpu()
    images_latent_as_np = images_latent_as_np.numpy()
    return images_latent_as_np