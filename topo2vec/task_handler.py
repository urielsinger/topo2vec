import logging
import os
import random
from argparse import Namespace

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.backends import cudnn

from topo2vec import modules
from common.list_conversions_utils import str_to_int_list
from topo2vec.constants import LOGS_PATH, FINAL_MODEL_DIR
from topo2vec.modules.knearestneighbourstester import KNearestNeighboursTester

import optuna
import numpy as np

class TaskHandler:
    '''
    A lab where classifier objects are tested in different hypeparams
    '''

    def __init__(self, model_hyperparams):
        self.model_hyperparams = model_hyperparams

    def _run_experiment(self, hparams: Namespace) -> float:
        '''
        runs an experiment
        Args:
            hparams: the hparams for the training, the model, and the architecture.

        Returns: the accuracy of the experiment on the validation

        '''
        if hparams.seed is not None:
            random.seed(hparams.seed)
            torch.manual_seed(hparams.seed)
            cudnn.deterministic = True
            np.random.seed(hparams.seed)

        name = f'{hparams.name}_{hparams.arch}_{str(hparams.radii)}_lr_' \
            f'{str(hparams.learning_rate)}' \
            f'_size_{hparams.total_dataset_size}_num_classes_{hparams.num_classes}' \
            f'_latent_size_{hparams.latent_space_size}_train_all_resnet_{hparams.train_all_resnet}'
        logging.info(f'started running, name = {name}')

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
        logger = TensorBoardLogger(hparams.logs_path, name=name)

        # init the trainer
        if hparams.pretrained:
            trainer = pl.Trainer(max_epochs=0, logger=logger)
        else:
            if hparams.use_gpu:
                trainer = pl.Trainer(max_epochs=hparams.max_epochs, logger=logger, gpus=1)
            else:
                trainer = pl.Trainer(max_epochs=hparams.max_epochs, logger=logger)

        if len(list(model.parameters())) != 0:
            trainer.fit(model)

        torch.cuda.empty_cache()

        #test time
        # if model.test_dataset is not None:
        #     trainer.test(model)

        # if hparams.test_knn:
        #     knn = KNearestNeighboursTester(random_set_size=hparams.random_set_size_for_knn,
        #                                    original_radiis=str_to_int_list(hparams.original_radiis),
        #                                    radii=str_to_int_list(hparams.radii),
        #                                    feature_extractor=model, k=hparams.k,
        #                                    method=hparams.knn_method_for_typical_choosing,
        #                                    json_file_of_group=hparams.json_file_of_group_for_knn)
        #     knn.prepare_data()
        #     knn.test_and_plot_via_feature_extractor_tensorboard()



        #save if needed
        if hparams.save_model:
            torch.save(model.state_dict(), save_path)

        if hparams.save_to_final:
            save_path = os.path.join(FINAL_MODEL_DIR, hparams.final_file_name)
            torch.save(model.state_dict(), save_path)

        return float(model.get_hyperparams_value_for_maximizing())


    def _build_hparams_and_run_experiment(self, trial):
        args = self.model_hyperparams
        if args.pytorch_module == 'Autoencoder' or args.pytorch_module == 'Outpainting':
            vars(args)['arch'] = trial.suggest_categorical('arch', ['AdvancedAmphibAutoencoder',
                                                                    'BasicAutoencoder',
                                                                    'BasicAmphibAutoencoder'])
        # if args.pytorch_module == 'Classifier':
        #     vars(args)['arch'] = trial.suggest_categorical('arch', ['BasicConvNetLatent',
        #                                                             'AdvancedConvNetLatent'])

        vars(args)['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-8, 1e-3)
        #vars(args)['latent_space_size'] = trial.suggest_int('latent_space_size', 5, 70)
        # vars(args)['total_dataset_size'] = trial.suggest_categorical('total_dataset_size', [2500, 10000])

        return self._run_experiment(args)

    def run(self):
        self._run_experiment(self.model_hyperparams)

    def run_hparams_search(self):
        '''
        run all the experiments using the self.model_hyperparams dictionary

        currently only one experiment is being held.
        hrere the hyperparams search will be added
        '''
        pruner = optuna.pruners.MedianPruner()

        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(self._build_hparams_and_run_experiment, n_trials=100, timeout=60*60*2,
                       n_jobs=1)

        logging.info("Number of finished trials: {}".format(len(study.trials)))

        logging.info("Best trial:")
        trial = study.best_trial

        logging.info("  Max validation accuracy: {}".format(trial.value))

        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info("    {}: {}".format(key, value))


