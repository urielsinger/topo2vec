from abc import abstractmethod
from typing import List, Dict

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import ParameterGrid
import pytorch_lightning as pl

from topo2vec.constants import LOGS_PATH
from topo2vec.modules.classifier import Classifier


class ClassificationTask:
    '''
    A lab where classifier objects are tested in different hypeparams
    '''
    def __init__(self, model_hyperparams:Dict):
        self.model_hyperparams = model_hyperparams

    def _run_experiment(self, radii: List[int], learning_rate: float,
                        total_dataset_size: int, max_epochs: int, name:str,
                        datasets_generator, **hparams):

        name = f'{name}_{str(radii)}_lr_{str(learning_rate)}' \
            f'_size_{total_dataset_size}_num_classes_{hparams["num_classes"]}'
        print(f'started running, name = {name}')

        train_set, val_set, test_set, random_set, typical_images_set = datasets_generator(radii, total_dataset_size)
        module = Classifier(train_dataset=train_set, validation_dataset=val_set,
                            test_dataset=test_set, radii=radii, random_dataset=random_set,
                            typical_images_dataset=typical_images_set,
                            learning_rate=learning_rate, **hparams)
        logger = TensorBoardLogger(LOGS_PATH, name=name)
        trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
        trainer.fit(module)
        trainer.test()

    def run(self):
        '''
        run all the experiments using the self.model_hyperparams dictionary
        '''
        param_grid = list(ParameterGrid(self.model_hyperparams))
        for params_subset in param_grid:
            self._run_experiment(**params_subset)

    @abstractmethod
    def _generate_datasets(self, radii, total_dataset_size):
        pass


