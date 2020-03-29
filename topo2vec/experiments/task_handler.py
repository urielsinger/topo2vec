import os
from argparse import Namespace

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from topo2vec import modules
from topo2vec.common import visualizations
from topo2vec.common.other_scripts import str_to_int_list
from topo2vec.constants import LOGS_PATH
from topo2vec.modules.knearestneighbourstester import KNearestNeighboursTester

from sklearn import svm

class TaskHandler:
    '''
    A lab where classifier objects are tested in different hypeparams
    '''

    def __init__(self, model_hyperparams):
        self.model_hyperparams = model_hyperparams

    def _run_experiment(self, hparams: Namespace):

        name = f'{hparams.name}_{str(hparams.radii)}_lr_{str(hparams.learning_rate)}' \
            f'_size_{hparams.total_dataset_size}_num_classes_{hparams.num_classes}'
        print(f'started running, name = {name}')

        # init the model
        save_path = os.path.join(hparams.save_path, name + str('.pt'))
        pytorch_module = modules.__dict__[hparams.pytorch_module]
        if hparams.pretrained and save_path is not None:
            model = pytorch_module(hparams)
            model.load_state_dict(torch.load(save_path))
            model.eval()
        else:
            model = pytorch_module(hparams)

        #init the logger
        logger = TensorBoardLogger(LOGS_PATH, name=name)

        # init the trainer
        if hparams.pretrained:
            trainer = pl.Trainer(max_epochs=0, logger=logger)
        else:
            trainer = pl.Trainer(max_epochs=hparams.max_epochs, logger=logger)

        if len(list(model.parameters())) != 0:
            trainer.fit(model)

        #test time
        if model.test_dataset is not None:
            trainer.test()

        if hparams.test_knn:
            knn = KNearestNeighboursTester(random_set_size=hparams.random_set_size,
                                           radii=str_to_int_list(hparams.radii),
                                           feature_extractor=model, k=hparams.k)
            knn.prepare_data()
            knn.test()


        #save if needed
        if hparams.save_model:
            torch.save(model.state_dict(), save_path)


    def run(self):
        '''
        run all the experiments using the self.model_hyperparams dictionary

        currently only one experiment is being held.
        hrere the hyperparams search will be added
        '''
        self._run_experiment(self.model_hyperparams)
