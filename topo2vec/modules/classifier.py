import argparse
from argparse import Namespace
from typing import Dict

import torchvision as torchvision
import torch
from pytorch_lightning import LightningModule
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader

import topo2vec.models as models
from topo2vec.background import TRAIN_HALF, VALIDATION_HALF, class_paths, class_names, LOAD_CLASSES_LARGE
from topo2vec.common import visualizations
from topo2vec.common.other_scripts import str_to_int_list
from topo2vec.common.visualizations import convert_multi_radius_tensor_to_printable, get_grid_sample_images_at_indexes, \
    get_random_part_of_dataset
from topo2vec.constants import class_paths_test, STATE_DICT_PATH
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams.learning_rate)
        self.total_dataset_size = hparams.total_dataset_size
        self.train_portion = hparams.train_portion
        self.embedding_size = hparams.embedding_size

    def prepare_data(self):
        size_train = int(self.train_portion * self.total_dataset_size)
        size_val = int((1 - self.train_portion) * self.total_dataset_size)

        self.train_dataset = SeveralClassesDataset(self.radii, TRAIN_HALF, size_train, class_paths, class_names)
        self.validation_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, size_val, class_paths, class_names)

        if LOAD_CLASSES_LARGE:
            size_test = 55
            self.test_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, size_test, class_paths_test,
                                                      class_names)
        else:
            self.test_dataset = None

    def forward(self, x: Tensor):
        return self.model(x.float())

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
            basic_dict = self.evaluation_step(batch, 'test')
            return basic_dict
        return {}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict:
        return self.evaluation_step(batch, 'validation')

    def evaluation_step(self, batch: Tensor, name: str) -> Dict:
        x, y = batch
        outputs, _ = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        batch_size, channels, _, _ = x.size()
        accuracy = torch.tensor([float(torch.sum(predicted == y.squeeze())) / batch_size])
        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        if self.num_classes == 2:
            self.run_images_TP_TN_FP_FN_evaluation(predicted, x, y)

        return {name + '_loss': loss, name + '_acc': accuracy}

    def run_images_TP_TN_FP_FN_evaluation(self, predicted, x, y):
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
        self.sample_images_and_log_one_hot(x_normalized, false_negatives_idxs, 'FN: positives, labeled wrong')

        false_positives_idxs = ((predicted != y) & (y == 0))
        self.sample_images_and_log_one_hot(x_normalized, false_positives_idxs, 'FP: negatives, labeled wrong')

        true_negatives_idxs = ((predicted == y) & (y == 0))
        self.sample_images_and_log_one_hot(x_normalized, true_negatives_idxs, 'TN: negatives, labeled right')

        true_positives_idxs = ((predicted == y) & (y == 1))
        self.sample_images_and_log_one_hot(x_normalized, true_positives_idxs, 'TP: positives, labeled right')

        grid = torchvision.utils.make_grid(x_normalized)
        self.logger.experiment.add_image('example images', grid, 0)

    def sample_images_and_log_one_hot(self, all_images: torch.tensor,
                                      one_hot_vector: torch.tensor,
                                      title: str, number_to_log: int = 5):
        if torch.sum(one_hot_vector) > 0:
            actual_idxs = torch.nonzero(one_hot_vector)
            grid = get_grid_sample_images_at_indexes(all_images, actual_idxs[:, 0],
                                                     number_to_log=number_to_log)
            self.feature_extractor.logger.experiment.add_image(title, grid, 0)

    def test_epoch_end(self, outputs: list) -> Dict:
        if self.test_dataset is not None:
            basic_dict = self.evaluation_epoch_end(outputs, 'test')
            self.log_embedding_visualization()
            svm_classifier_test_log_dict = self.svm_classifier_test()
            new_log_dict = {**svm_classifier_test_log_dict, **basic_dict['log']}
            basic_dict['log'] = new_log_dict
            return basic_dict
        return {}

    def svm_classifier_test(self):
        if self.hparams.svm_classify_latent_space:
            svm_train_dataset = SeveralClassesDataset(self.radii, TRAIN_HALF, self.hparams.random_set_size_for_svm,
                                                      class_paths, class_names)
            X_train, y_train = visualizations.get_dataset_as_tensor(svm_train_dataset)
            _, latent_train = self.forward(X_train)
            SVMClassifier = svm.SVC()
            SVMClassifier.fit(latent_train.numpy(), y_train.numpy())

            X_test, y_test = visualizations.get_dataset_as_tensor(self.test_dataset)
            _, latent_test = self.forward(X_test)

            predicted_train = SVMClassifier.predict(latent_train.numpy())
            predicted_test = SVMClassifier.predict(latent_test.numpy())

            train_accuracy = accuracy_score(y_train.numpy(), predicted_train)
            test_accuracy = accuracy_score(y_test.numpy(), predicted_test)

            return {'svm_train_accuracy': train_accuracy, 'svm_test_accuracy': test_accuracy}
        return {}

    def validation_epoch_end(self, outputs: list) -> Tensor:
        basic_dict = self.evaluation_epoch_end(outputs, 'validation')
        return basic_dict

    def evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x[name + '_acc'] for x in outputs]).mean()

        tensorboard_logs = {name + '_loss': avg_loss,
                            name + '_acc': avg_accuracy}
        return {'avg_' + name + '_loss': avg_loss, 'log': tensorboard_logs}

    def log_embedding_visualization(self):
        x, coords = get_random_part_of_dataset(self.test_dataset,
                                               self.embedding_size)
        _, embedding = self.forward(x.float())
        images_set_to_show = x[:, 0, :, :].unsqueeze(1)
        self.logger.experiment.add_embedding(embedding, metadata=coords, label_img=images_set_to_show)

    def configure_optimizers(self):
        return self.optimizer

    @staticmethod
    def get_args_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--embedding_size', type=list, default=100)
        parser.add_argument('--radii', type=str, default='[8, 16, 24]')
        parser.add_argument('--arch', type=str)
        parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='learning_rate')
        parser.add_argument('--total_dataset_size', type=int, default=75000)
        parser.add_argument('--max_epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--num_classes', type=int, default=len(class_paths),
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
        parser.add_argument('--save_path', metavar='DIR', default=STATE_DICT_PATH, type=str,
                            help='path to save model')
        parser.add_argument('--random_set_size', type=int, default=1000,
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

        return parser
