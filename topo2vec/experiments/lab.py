from abc import abstractmethod

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import ParameterGrid
from torch import optim, nn
import pytorch_lightning as pl
from topo2vec.modules.classifier import Classifier

logs_path = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs'


class Lab:
    '''
    Where all experiments are handled - while doing hyperparams grid search
    if needed - the hyperparams can be overriden
    in the labs inheriting from this lab.
    '''
    def __init__(self):
        self.model_hyperparams = {
            'max_epochs': [100],
            'optimizer_cls': [optim.Adam]
        }

    def _run_experiment(self, max_epochs, name, **hparams):
        '''
        run a specific experiment
        Args:
            max_epochs:
            name:
            **hparams:

        Returns:

        '''
        module = Classifier(**hparams)
        logger = TensorBoardLogger(logs_path, name=name)
        trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
        trainer.fit(module)

    def run(self):
        '''
        run all the experiments using the self.model_hyperparams dictionary
        '''
        param_grid = list(ParameterGrid(self.model_hyperparams))
        for params_subset in param_grid:
            self._run_experiment(**params_subset)
