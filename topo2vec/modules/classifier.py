import argparse
from argparse import Namespace
from typing import Dict

import sklearn
import torchvision as torchvision
import torch
from pytorch_lightning import LightningModule
from sklearn import svm

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


import topo2vec.models as models
from topo2vec.background import TRAIN_HALF, VALIDATION_HALF, CLASS_PATHS, CLASS_NAMES, LOAD_CLASSES_LARGE, \
    CLASS_PATHS_TEST, CLASS_NAMES_TEST, CLASS_NAMES_SPECIAL, CLASS_PATHS_SPECIAL
from topo2vec.common.other_scripts import str_to_int_list, get_random_part_of_dataset, get_dataset_as_tensor, \
    get_paths_and_names_wanted, svm_accuracy_on_dataset_in_latent_space
from topo2vec.common.visualizations import convert_multi_radius_tensor_to_printable, get_grid_sample_images_at_indexes, \
    plot_to_image, plot_confusion_matrix
from topo2vec.constants import SAVE_PATH, GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE, LOGS_PATH
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset


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
        self.class_paths = CLASS_PATHS

    def prepare_data(self):
        size_train = int(self.train_portion * self.total_dataset_size)
        size_val = int((1 - self.train_portion) * self.total_dataset_size)

        self.train_dataset = SeveralClassesDataset(self.radii, TRAIN_HALF, size_train, self.class_paths, self.class_names,
                                                   'num_classes_' + str(self.num_classes) + '_train')

        self.validation_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, size_val, CLASS_PATHS, CLASS_NAMES,
                                                        'num_classes_' + str(self.num_classes) + '_validation')

        if LOAD_CLASSES_LARGE:
            self.test_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, self.size_test, CLASS_PATHS_TEST,
                                                      CLASS_NAMES_TEST,
                                                      'num_classes_' + str(self.num_classes) + '_test')
        else:
            self.test_dataset = None

    def forward(self, x: Tensor):
        return self.model(x.float())

    def get_classifications(self, x: Tensor):
        outputs, latent = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted


    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=0, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, shuffle=True, num_workers=0, batch_size=64)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, shuffle=True, num_workers=0, batch_size=16)

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
        x, y = batch
        outputs, _ = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        batch_size, channels, _, _ = x.size()
        accuracy = torch.tensor([float(torch.sum(predicted == y.squeeze())) / batch_size])
        accuracies = {}
        for i in range(self.num_classes):
            predicted_i = (predicted == i)
            y_i = (y == i).squeeze()
            accuracies[f'{name}_{CLASS_NAMES[i]}_acc'] = torch.tensor(
                [float(torch.sum(predicted_i * y_i.squeeze()) / (torch.sum(y_i) + 0.000001))])

        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        if self.num_classes == 2:
            self._run_images_TP_TN_FP_FN_evaluation(predicted, x, y)

        y = y.squeeze().int()
        predicted = predicted.squeeze()
        if self.hparams.use_gpu:
            y = y.cpu()
            predicted = predicted.cpu()

        self.log_confusion_matrix(y.numpy(), predicted.numpy(), CLASS_NAMES)

        return {**{name + '_loss': loss, name + '_acc': accuracy}, **accuracies}

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
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='pred')
        ax, fig = plot_confusion_matrix(cm=confusion_matrix,
                              normalize=False,
                              target_names=CLASS_NAMES,
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
        if self.test_dataset is not None:
            self._log_embedding_visualization()
            basic_dict = self._evaluation_epoch_end(outputs, 'test')
            if 'test_acc' in basic_dict['log']:
                self.final_test_accuracy = basic_dict['log']['test_acc']

            svm_classifier_ordinary_classes_test_log_dict = self._svm_classifier_test(CLASS_PATHS, CLASS_NAMES,
                                                                                      'ordinary', self.test_dataset,
                                                                                      self.hparams.random_set_size_for_svm)

            class_paths_special, class_names_special = get_paths_and_names_wanted(
                self.hparams.special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

            svm_special_test_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF,
                                                             self.hparams.test_set_size_for_svm,
                                                             class_paths_special, class_names_special,
                                                             'test_svm_special')

            svm_classifier_special_classes_test_log_dict = \
                self._svm_classifier_test(class_paths_special, class_names_special,
                                          'special', svm_special_test_dataset,
                                          self.hparams.random_set_size_for_svm_special)

            new_log_dict = {**svm_classifier_ordinary_classes_test_log_dict,
                            **svm_classifier_special_classes_test_log_dict,
                            **basic_dict['log']}
            basic_dict['log'] = new_log_dict
            return basic_dict
        return {}

    def _svm_classifier_test(self, class_paths_to_test: str, class_names_to_test: str, name: str,
                             test_dataset: Dataset, train_dataset_size):
        if self.hparams.svm_classify_latent_space:
            svm_train_dataset = SeveralClassesDataset(self.radii, TRAIN_HALF, train_dataset_size,
                                                      class_paths_to_test, class_names_to_test, 'train_svm_' + name)
            svm_validation_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, train_dataset_size,
                                                           class_paths_to_test, class_names_to_test,
                                                           'train_svm_' + name)

            X_train, y_train = get_dataset_as_tensor(svm_train_dataset)
            if self.hparams.use_gpu:
                X_train = X_train.cuda()

            _, latent_train = self.forward(X_train)
            SVMClassifier = svm.SVC()

            if self.hparams.use_gpu:
                latent_train = latent_train.cpu()
                y_train = y_train.cpu()

            latent_train_numpy = latent_train.numpy()
            y_train_numpy = y_train.numpy()

            SVMClassifier.fit(latent_train_numpy, y_train_numpy)

            train_accuracy = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                     svm_train_dataset, self)

            validation_accuracy = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                          svm_validation_dataset, self)

            test_accuracy = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                    test_dataset, self)
            self.svm_validation_accuracy = validation_accuracy
            self.svm_test_accuracy = test_accuracy

            return {f'svm_train_{name}_accuracy': train_accuracy,
                    f'svm_validation_{name}_accuracy': validation_accuracy,
                    f'svm_test_{name}_accuracy': test_accuracy}
        return {}

    def validation_epoch_end(self, outputs: list) -> Tensor:
        basic_dict = self._evaluation_epoch_end(outputs, 'validation')
        if 'validation_acc' in basic_dict['log']:
            self.final_validation_accuracy = basic_dict['log']['validation_acc']
            if self.final_validation_accuracy > self.max_validation_accuracy:
                self.max_validation_accuracy = self.final_validation_accuracy
        return basic_dict

    def _evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x[name + '_acc'] for x in outputs]).mean()
        avg_accuracies = {}
        for i in range(self.num_classes):
            avg_accuracies[f'{name}_{CLASS_NAMES[i]}_acc'] = torch.stack(
                [x[f'{name}_{CLASS_NAMES[i]}_acc'] for x in outputs]).mean()

        tensorboard_logs = {name + '_loss': avg_loss,
                            name + '_acc': avg_accuracy,
                            **avg_accuracies}
        return {'avg_' + name + '_loss': avg_loss, 'log': tensorboard_logs}

    def _log_embedding_visualization(self):
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
        # )
        return [optimizer]

    def get_hyperparams_value(self):
        return self.max_validation_accuracy

    @staticmethod
    def get_args_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--embedding_visualization_size', type=list, default=100)
        parser.add_argument('--radii', type=str, default='[8, 16, 24]')
        parser.add_argument('--arch', type=str)
        parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='learning_rate')
        parser.add_argument('--total_dataset_size', type=int, default=75000)
        parser.add_argument('--max_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--num_classes', type=int, default=len(CLASS_PATHS),
                            help='number of the classes in the dataset. ')
        parser.add_argument('--name', type=str,
                            help='number of the classes in the dataset. ')
        parser.add_argument('--pytorch_module', type=str)
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--save_model', dest='save_model', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--save_path', metavar='DIR', default=SAVE_PATH, type=str,
                            help='path to save model')
        parser.add_argument('--random_set_size', type=int, default=10000,
                            help='seed for initializing training. ')
        parser.add_argument('--k', type=int, default=5,
                            help='seed for initializing training. ')
        parser.add_argument('--train_portion', type=float, default=0.8,
                            help='portion of the total_dataset_Size to put into training.')
        parser.add_argument('--latent_space_size', type=int, default=50,
                            help='size of the desired latent space of the autoencoder.')
        parser.add_argument('--test_knn', dest='test_knn', action='store_true',
                            help='test the latent space to find the knn of the main things')
        parser.add_argument('--svm_classify_latent_space', dest='svm_classify_latent_space', action='store_true',
                            help='classify the latent space using one linear layer to check if it is good')
        parser.add_argument('--random_set_size_for_svm', type=int, default=1000)
        parser.add_argument('--random_set_size_for_svm_special', type=int, default=1000)
        parser.add_argument('--size_test', type=int, default=55)
        parser.add_argument('--knn_method_for_typical_choosing', type=str, default='group_from_file',
                            help='"regular" or "group_from_file"')
        parser.add_argument('--special_classes_for_validation', type=str, default='["alpine_huts", "power_towers"]',
                            help='"regular" or "group_from_file"')
        parser.add_argument('--test_set_size_for_svm', type=int, default=100,
                            help='"regular" or "group_from_file"')
        parser.add_argument('--json_file_of_group_for_knn', type=str, default=GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE)
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--use_gpu', dest='use_gpu', action='store_true',
                            help='put for using a gpu')
        parser.add_argument('--save_to_final', dest='save_to_final', action='store_true',
                            help='save the model to the final place')
        parser.add_argument('--logs_path', default=LOGS_PATH, type=str,
                            help='tensorboard logs poath')

        return parser
