from topo2vec.background import VALIDATION_HALF, CLASS_PATHS, CLASS_NAMES
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset

original_radiis = [[8,16,24]]
size_val = 16000
RANDOM_SEED = 665
europe_dataset_ordinary = SeveralClassesDataset(original_radiis, VALIDATION_HALF, size_val, CLASS_PATHS, CLASS_NAMES,
                                                        'europe_dataset_for_eval_regular', random_seed=RANDOM_SEED)

for class_dataset in europe_dataset_ordinary.all_datasets:
    class_dataset.save_to_pickle()