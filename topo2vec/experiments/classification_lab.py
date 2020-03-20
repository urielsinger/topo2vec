from abc import abstractmethod
from typing import List

from topo2vec.experiments.lab import Lab

class ClassificationLab(Lab):
    def __init__(self):
        super().__init__()

        self.model_hyperparams.update({
            'radii': [[8], [8, 16], [16], [8, 16, 24], [24]],
            'learning_rate': [1e-4, 1e-5, 1e-6],
            'total_dataset_size': [1000, 10000, 40000],
        })

    def run_experiment(self, radii: List[int], learning_rate: float,
                       total_dataset_size: int, max_epochs: int,
                       **hparams):
        name = f'streams_vs_all_radii_{str(radii)}_lr_{str(learning_rate)}_size_{total_dataset_size}'
        print(f'started running, name = {name}')

        train_set, val_set = self._generate_datasets(radii, total_dataset_size)
        super(ClassificationLab, self).run_experiment(train_dataset=train_set,
                                                      validation_dataset=val_set, max_epochs=max_epochs,
                                                      name=name, radii=radii,
                                                      **hparams)

    @abstractmethod
    def _generate_datasets(self, radii, total_dataset_size):
        pass


