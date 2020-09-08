import argparse
import logging
from argparse import Namespace
from typing import Dict

import sklearn
import torchvision as torchvision
import torch
from pytorch_lightning import LightningModule

from torch import Tensor
from torch.utils.data import DataLoader

import topo2vec.models as models
from topo2vec.background import TRAIN_HALF, VALIDATION_HALF, CLASS_PATHS, CLASS_NAMES, LOAD_CLASSES_LARGE, \
    CLASS_PATHS_TEST, CLASS_NAMES_TEST, CLASS_NAMES_SPECIAL, CLASS_PATHS_SPECIAL
from common.dataset_utils import get_paths_and_names_wanted
from common.pytorch.pytorch_lightning_utilities import get_random_part_of_dataset, get_dataset_as_tensor
from common.list_conversions_utils import str_to_int_list
from common.pytorch.visualizations import convert_multi_radius_tensor_to_printable, get_grid_sample_images_at_indexes, \
    plot_to_image, plot_confusion_matrix
from topo2vec.constants import SAVE_PATH, GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE, LOGS_PATH
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.modules.svm_on_latent_tester import svm_classifier_test_build_datasets
import numpy as np


class Classifier(LightningModule):

    def __init__(self, hparams: Namespace):
        """
        a simple classifier to train on MultiRadiusDataset dataset
        using hparams, containing:
        arch - the architecture of the classifier
        and all other params defined in the "multi_class_experiment" script.
        """
        super(Classifier, self).__init__()
        self.hparams = hparams
        self.model = models.__dict__[hparams.arch](hparams)
        self.num_classes = hparams.num_classes
        self.original_radiis = str_to_int_list(hparams.original_radiis)
        self.radii = str_to_int_list(hparams.radii)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.total_dataset_size = hparams.total_dataset_size
        self.train_portion = hparams.train_portion
        self.embedding_visualization_size = hparams.embedding_visualization_size
        self.size_test = hparams.size_test
        self.final_validation_accuracy = 0
        self.max_validation_accuracy = 0
        self.final_test_accuracy = 0
        self.class_names = CLASS_NAMES
        logging.info(self.class_names)
        self.class_paths = CLASS_PATHS

        # for the scale experiment
        self.scale_exp = int(hparams.scale_exp)
        self.scale_exp_class_name = hparams.scale_exp_class_name
        self.scale_exp_class_path = hparams.scale_exp_class_path
        self.scale_exp_random_seed = hparams.scale_exp_random_seed

    def prepare_data(self):
        '''

        a pytorch-lightning function.

        '''
        size_train = int(self.train_portion * self.total_dataset_size)
        size_val = int((1 - self.train_portion) * self.total_dataset_size)
        if self.scale_exp:
            self.train_dataset = OneVsRandomDataset(self.original_radiis, size_train, TRAIN_HALF,
                                                    self.scale_exp_class_path,
                                                    # f'scale_exp_{self.scale_exp_class_name}_vs_random_train',
                                                    radii=self.radii, random_seed=self.scale_exp_random_seed)
            self.validation_dataset = OneVsRandomDataset(self.original_radiis, size_val, VALIDATION_HALF,
                                                         self.scale_exp_class_path,
                                                         # f'scale_exp_{self.scale_exp_class_name}_vs_random_validation',
                                                         radii=self.radii, random_seed=self.scale_exp_random_seed)
        else:
            self.train_dataset = SeveralClassesDataset(self.original_radiis, TRAIN_HALF, size_train, self.class_paths,
                                                       self.class_names,
                                                       'num_classes_' + str(self.num_classes) + '_train', self.radii)

            self.validation_dataset = SeveralClassesDataset(self.original_radiis, VALIDATION_HALF, size_val,
                                                            self.class_paths, self.class_names,
                                                            'num_classes_' + str(self.num_classes) + '_validation',
                                                            self.radii)

        if LOAD_CLASSES_LARGE:
            self.test_dataset = SeveralClassesDataset(self.original_radiis, VALIDATION_HALF, self.size_test,
                                                      CLASS_PATHS_TEST,
                                                      CLASS_NAMES_TEST,
                                                      'num_classes_' + str(self.num_classes) + '_test', self.radii)
        else:
            self.test_dataset = None

    def forward(self, x: Tensor):
        return self.model(x.float())

    def get_classifications(self, x: Tensor):
        outputs, latent = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def get_classification_and_probability(self, x: Tensor):
        outputs, latent = self.forward(x)
        probability, predicted = torch.max(outputs.data, 1)
        return predicted, probability

    def get_accuracy_for_small_dataset(self, dataset):
        X, y = get_dataset_as_tensor(dataset)
        outputs, _ = self.forward(X.float())
        _, predicted = torch.max(outputs.data, 1)
        batch_size, channels, _, _ = X.size()
        accuracy = torch.tensor([float(torch.sum(predicted == y.squeeze())) / batch_size])
        return accuracy

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=0, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, shuffle=False, num_workers=0, batch_size=64)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, shuffle=False, num_workers=0, batch_size=16)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict:
        x, y = batch
        outputs, _ = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict:
        if self.test_dataset is not None:
            basic_dict = self._evaluation_step(batch, 'test')
            return basic_dict
        return {}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict:
        return self._evaluation_step(batch, 'validation')

    def _evaluation_step(self, batch: Tensor, name: str) -> Dict:
        '''
        all the things needed to be done in each of both validation and test steps
        this is the part that will be overriden by sub-classes.

        Args:
            batch:
            name:

        Returns:
        '''
        x, y = batch
        outputs, _ = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        batch_size, channels, _, _ = x.size()
        tp = torch.tensor([float(torch.sum(predicted == y.squeeze()))])
        total = torch.tensor([float(batch_size)])
        TPS = {}
        totals = {}
        for i in range(self.num_classes):
            predicted_i = (predicted == i)
            y_i = (y == i).squeeze()
            TPS[f'{name}_{CLASS_NAMES[i]}_tp'] = torch.tensor(
                [float(torch.sum(predicted_i * y_i.squeeze()))])
            totals[f'{name}_{CLASS_NAMES[i]}_total'] = torch.tensor(
                [float(torch.sum(torch.sum(y_i)))])

        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        # a TP_TN_FP_FN is only relevant in 2 classes. the confusion matrix is better in every other case.
        if self.num_classes == 2:
            self._run_images_TP_TN_FP_FN_evaluation(predicted, x, y)

        y = y.squeeze().int()
        predicted = predicted.squeeze()
        if self.hparams.use_gpu:
            y = y.cpu()
            predicted = predicted.cpu()

        return {**{name + '_loss': loss, name + '_tp': tp, name + '_total': total, 'y': y, 'pred': predicted}, **TPS,
                **totals}

    def _run_images_TP_TN_FP_FN_evaluation(self, predicted, x, y):
        '''

        Args:
            predicted: the predicted values for each x
            x: the batch's data
            y: the GT of the batch

        Returns:

        '''
        x = x.float()
        # im_mean = x.view(batch_size, channels, -1).mean(2).view(batch_size, channels, 1, 1)
        # im_std = x.view(batch_size, channels, -1).std(2).view(batch_size, channels, 1, 1)
        x_normalized = x  # (x - im_mean) / (im_std)
        x_normalized = convert_multi_radius_tensor_to_printable(x_normalized)

        false_negatives_idxs = ((predicted != y) & (y == 1))
        self._sample_images_and_log_one_hot(x_normalized, false_negatives_idxs, 'FN: positives, labeled wrong')

        false_positives_idxs = ((predicted != y) & (y == 0))
        self._sample_images_and_log_one_hot(x_normalized, false_positives_idxs, 'FP: negatives, labeled wrong')

        true_negatives_idxs = ((predicted == y) & (y == 0))
        self._sample_images_and_log_one_hot(x_normalized, true_negatives_idxs, 'TN: negatives, labeled right')

        true_positives_idxs = ((predicted == y) & (y == 1))
        self._sample_images_and_log_one_hot(x_normalized, true_positives_idxs, 'TP: positives, labeled right')

        grid = torchvision.utils.make_grid(x_normalized)
        self.logger.experiment.add_image('example images', grid, 0)

    def log_confusion_matrix(self, y_true, y_pred, labels):
        '''
         log the confusion matrix of the classes trained on
        Args:
            y_true:
            y_pred:
            labels:

        Returns:

        '''
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize='true')
        if len(np.unique(y_pred)) == 4 and len(np.unique(y_true)) == 4:
            ax, fig = plot_confusion_matrix(cm=confusion_matrix,
                                            normalize=False,
                                            target_names=list(
                                                np.array(CLASS_NAMES)[np.sort(np.unique(y_pred)).astype(int)]),
                                            title="Confusion Matrix")
            image = plot_to_image(fig)
            tensor_image = Tensor(image)
            self.logger.experiment.add_image(f'confusion matrix: {labels}', tensor_image, 0, dataformats='HWC')

    def _sample_images_and_log_one_hot(self, all_images: torch.tensor,
                                       one_hot_vector: torch.tensor,
                                       title: str, number_to_log: int = 5):
        if torch.sum(one_hot_vector) > 0:
            actual_idxs = torch.nonzero(one_hot_vector)
            grid = get_grid_sample_images_at_indexes(all_images, actual_idxs[:, 0],
                                                     number_to_log=number_to_log)
            self.logger.experiment.add_image(title, grid, 0)

    def test_epoch_end(self, outputs: list) -> Dict:
        '''
        a function consisting of all the operations needed after the test epoch
        Args:
            outputs:

        Returns:

        '''
        if self.test_dataset is not None:
            self._log_embedding_visualization()
            basic_dict = self._evaluation_epoch_end(outputs, 'test')
            if 'test_acc' in basic_dict['log']:
                self.final_test_accuracy = basic_dict['log']['test_acc']

            svm_classifier_ordinary_classes_test_log_dict = svm_classifier_test_build_datasets(self, CLASS_PATHS,
                                                                                               CLASS_NAMES,
                                                                                               'ordinary',
                                                                                               self.test_dataset,
                                                                                               self.hparams.random_set_size_for_svm)

            class_paths_special, class_names_special = get_paths_and_names_wanted(
                self.hparams.special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

            svm_special_test_dataset = SeveralClassesDataset(self.original_radiis, VALIDATION_HALF,
                                                             self.hparams.test_set_size_for_svm,
                                                             class_paths_special, class_names_special,
                                                             'test_svm_special', self.radii)

            svm_classifier_special_classes_test_log_dict = \
                svm_classifier_test_build_datasets(self, class_paths_special, class_names_special,
                                                   'special', svm_special_test_dataset,
                                                   self.hparams.random_set_size_for_svm_special)

            new_log_dict = {**svm_classifier_ordinary_classes_test_log_dict,
                            **svm_classifier_special_classes_test_log_dict,
                            **basic_dict['log']}
            basic_dict['log'] = new_log_dict
            return basic_dict
        return {}

    def validation_epoch_end(self, outputs: list) -> Dict:
        '''
        a function consisting of all the operations needed after each validation epoch
        Args:
            outputs:

        Returns:

        '''
        basic_dict = self._evaluation_epoch_end(outputs, 'validation')
        self.log_confusion_matrix(torch.cat([x['y'] for x in outputs]).numpy(),
                                  torch.cat([x['pred'] for x in outputs]).numpy(), CLASS_NAMES)

        if 'validation_acc' in basic_dict['log']:
            self.final_validation_accuracy = basic_dict['log']['validation_acc']
            if self.final_validation_accuracy > self.max_validation_accuracy:
                self.max_validation_accuracy = self.final_validation_accuracy
        return basic_dict

    def _evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        '''
        a function consisting of all the operations needed after evaluation_experiments epoch, both validation and test epochs
        this is the part that will be overriden by sub-classes.
        Args:
            outputs:
            name: whether it is the test or validation evaluation_experiments epoch end

        Returns:

        '''
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        avg_accuracy = torch.tensor([torch.stack([x[name + '_tp'] for x in outputs]).sum() / torch.stack(
            [torch.tensor([x[name + '_total']]) for x in outputs]).sum()])
        avg_accuracies = {}
        for i in range(self.num_classes):
            avg_accuracies[f'{name}_{CLASS_NAMES[i]}_acc'] = torch.tensor([torch.stack(
                [x[f'{name}_{CLASS_NAMES[i]}_tp'] for x in outputs]).sum() / torch.stack(
                [x[f'{name}_{CLASS_NAMES[i]}_total'] for x in outputs]).sum()])

        tensorboard_logs = {name + '_loss': avg_loss,
                            name + '_acc': avg_accuracy,
                            **avg_accuracies}
        return {'avg_' + name + '_loss': avg_loss, 'log': tensorboard_logs}

    def _log_embedding_visualization(self):
        '''
        log the embedding space of the latent to the tensorboard
        '''
        x, y = get_random_part_of_dataset(self.test_dataset,
                                          self.embedding_visualization_size)
        x = x.float()
        if self.hparams.use_gpu:
            x = x.cuda()
        _, embedding = self.forward(x)
        images_set_to_show = x[:, 0, :, :].unsqueeze(1)
        self.logger.experiment.add_embedding(embedding, metadata=y, label_img=images_set_to_show)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # weight_decay=self.hparams.weight_decay
        return [optimizer]

    def get_hyperparams_value_for_maximizing(self):
        '''

        Returns: the value we want to maximize when running an optuna hyper-params search for classifiers


        '''
        return self.final_validation_accuracy

    @staticmethod
    def get_args_parser():
        parser = argparse.ArgumentParser(add_help=False)

        # general / saving constants #
        ##############################

        parser.add_argument('--name', type=str,
                            help='a special name for the current model running')
        parser.add_argument('--save_model', dest='save_model', action='store_true',
                            help='asked to save the model in save_path location')
        parser.add_argument('--save_path', metavar='DIR', default=SAVE_PATH, type=str,
                            help='path to save model')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model, stored in save_path/{name}.pt')

        parser.add_argument('--save_to_final', dest='save_to_final', action='store_true',
                            help='save the model to the location of the "final" model - the one used in the server_api'
                                 ' itself')
        parser.add_argument('--final_file_name', dest='final_file_name', default='final_model.pt', type=str)

        parser.add_argument('--logs_path', default=LOGS_PATH, type=str,
                            help='tensorboard logs poath')

        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('--use_gpu', dest='use_gpu', action='store_true',
                            help='put for using a gpu, if you have one and cuda configured properly')

        # the model #
        #############

        parser.add_argument('--original_radiis', type=str, default='[[8, 16, 24],[4, 8, 12],[12,24,36]]')
        parser.add_argument('--radii', type=str, default='[8, 16, 24]')

        parser.add_argument('--arch', type=str)
        parser.add_argument('--discriminator', type=str)
        parser.add_argument('--num_classes', type=int, default=len(CLASS_PATHS),
                            help='number of the classes in the dataset. ')
        parser.add_argument('--pytorch_module', type=str,
                            help='choose what pytorch module to use: classsifier / autoencoder')
        parser.add_argument('--latent_space_size', type=int, default=50,
                            help='size of the desired latent space of the autoencoder.')
        parser.add_argument('--train_all_resnet', dest='train_all_resnet', action='store_true',
                            help='put if using a resnet architecture and want to train it all')

        # if the model is for scale exsperiment: should be trained on one vs random #
        #############################################################################
        parser.add_argument('--scale_exp', dest='scale_exp', action='store_true',
                            help='if exists - this is a scale experiment object')
        parser.add_argument('--scale_exp_class_name', type=str,
                            help='name of the class in the experiment')
        parser.add_argument('--scale_exp_class_path', type=str,
                            help='path to the json file of the class of this experiment')
        parser.add_argument('--scale_exp_random_seed', type=int, default=55,
                            help='random seed for building the train dataset')

        # the classes data #
        ####################

        parser.add_argument('--train_portion', type=float, default=0.8,
                            help='portion of the total_dataset_Size to put into training.')
        parser.add_argument('--size_test', type=int, default=55,
                            help='what is the total size of the wanted test data (from data/overpass_classes_Data/tests)')

        # training constants #
        ######################

        parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='learning_rate')
        parser.add_argument('--total_dataset_size', type=int, default=75000)
        parser.add_argument('--max_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')

        # evealuation / Tensorboard constants #
        #######################################

        # embeding #

        # how many points we want to put in the embedding space
        parser.add_argument('--embedding_visualization_size', type=list, default=100)
        parser.add_argument('--index_in', type=int, default=1)
        parser.add_argument('--index_out', type=int, default=2)

        # knn #

        # knn_test for the latent space - search for points similar in latent space and
        # find out what the latent space is actually about.
        parser.add_argument('--test_knn', dest='test_knn', action='store_true',
                            help='test the latent space to find the knn of the main things')
        parser.add_argument('--random_set_size_for_knn', type=int, default=10000,
                            help='the random set size for the knn evaluation_experiments')
        parser.add_argument('--k', type=int, default=5,
                            help='the number of neerest neighbours for the knn ebvaluation')

        # regular is just choosing points from file and getting similar to them
        # group_from_file is taking all the points from the file saved in --json_file_of_group_for_knn
        parser.add_argument('--knn_method_for_typical_choosing', type=str, default='group_from_file',
                            help='"regular" or "group_from_file"')
        parser.add_argument('--json_file_of_group_for_knn', type=str, default=GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE)

        # svm on latent #

        # svm classification on top of the model - for evaluation_experiments only
        parser.add_argument('--svm_classify_latent_space', dest='svm_classify_latent_space', action='store_true',
                            help='classify the latent space using one linear layer to check if it is good')
        parser.add_argument('--random_set_size_for_svm', type=int, default=1000)
        parser.add_argument('--random_set_size_for_svm_special', type=int, default=1000,
                            help='svm special is trying to differ between classes not trained on (e.g. alpine_huts, antennas, etc. -'
                                 'saved in --special_classes_for_validation')
        parser.add_argument('--special_classes_for_validation', type=str, default='["alpine_huts", "waterfalls"]',
                            help='a list of the names of the names of the classes where the svm_special is stored')
        parser.add_argument('--test_set_size_for_svm', type=int, default=100)

        # for classifiying on top of latent
        parser.add_argument('--retrain', dest='retrain', action='store_true',
                            help='if true - retrain the basic model that give sthe latent')
        return parser
