from typing import List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from topo2vec.CONSTANTS import N49_E05_RIVERS, N49_E05_CLIFFS, N49_E05_STREAMS
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.modules.classifier import Classifier

logs_path = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs'

class_paths = [N49_E05_RIVERS, N49_E05_CLIFFS, N49_E05_STREAMS]
class_names = ['River', 'Cliff', 'Stream']



def run_model(radii: List[int], learning_rate: float, total_dataset_size:int):
    name = f'streams_vs_all_radii_{str(radii)}_lr_{str(learning_rate)}_size_{total_dataset_size}'
    print(f'started running, name = {name}')
    SeveralClassesDataset()
    tri_class_dataset = SeveralClassesDataset(radii, total_dataset_size,
                                              class_paths, class_names)
    model = Classifier(radii=radii, learning_rate=learning_rate)
    logger = TensorBoardLogger(logs_path, name=name)
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(model)

radiis = [[8], [8, 16], [16], [8, 16, 24], [24]]
learning_rates = [1e-4, 1e-5, 1e-6]
total_dataset_sizes = [1000, 10000, 100000]

radiis = [[16]]
learning_rates = [1e-4]
total_dataset_sizes = [50000]

def run_experiment():
    for radii in radiis:
        for lr in learning_rates:
            for total_dataset_size in total_dataset_sizes:
                run_model(radii, lr, total_dataset_size)
