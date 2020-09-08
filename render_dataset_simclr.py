from topo2vec.background import VALIDATION_HALF, CLASS_PATHS, CLASS_NAMES, TRAIN_HALF
from topo2vec.datasets.random_dataset import RandomDataset
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
radii = [17]
original_radiis = [radii]
size_val = 10000
RANDOM_SEED = 665
europe_dataset_ordinary = RandomDataset(size_val, original_radiis, TRAIN_HALF, radii, 0, seed=RANDOM_SEED)
europe_dataset_ordinary.save_to_pickle(location='data/to_zareck', name='random')