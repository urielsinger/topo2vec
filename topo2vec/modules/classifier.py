import random
from typing import List

import torchvision as torchvision
import torch
from pytorch_lightning import LightningModule
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

import topo2vec.models as models


class Classifier(LightningModule):
    def __init__(self, validation_dataset: Dataset, train_dataset: Dataset,
                 loss_func, optimizer_cls, test_dataset: Dataset,
                 random_dataset: Dataset, typical_images_dataset: Dataset,
                 arch: str = 'simpleconvnet',
                 pretrained: bool = False, radii: List[int] = [10],
                 num_classes: int = 1, learning_rate: float = 0.0001):
        """
        a simple classifier to train on MultiRadiusDataset dataset
        """
        super(Classifier, self).__init__()
        self.model = models.__dict__[arch](pretrained=pretrained,
                                           num_classes=num_classes, radii=radii)
        self.num_classes = num_classes

        self.loss_fn = loss_func
        self.optimizer = optimizer_cls(self.parameters(), lr=learning_rate)

        self.val_set = validation_dataset
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.random_set = random_dataset
        self.typical_images_set = typical_images_dataset
        self.k = 5

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, num_workers=0, batch_size=1000)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=True, num_workers=0, batch_size=1000)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=True, num_workers=0, batch_size=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs, _ = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        if self.random_set is not None:
            self.test_k_nn()
        basic_dict = self.evaluation_step(batch, 'test')
        return basic_dict

    def _get_dataset_as_tensor(self, dataset):
        dataset_length = len(dataset)
        data_loader = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=dataset_length)
        images_as_tensor, _ = next(iter(data_loader))
        return images_as_tensor

    def _get_dataset_latent_space_as_np(self, images_as_tensor):
        _, images_latent_as_tensor = self.forward(images_as_tensor.float())
        images_latent_as_np = images_latent_as_tensor.data.numpy()
        return images_latent_as_np

    def test_k_nn(self):
        random_images_as_tensor = self._get_dataset_as_tensor(self.random_set)
        random_images_set_to_calc_distance = random_images_as_tensor[:, -1, :, :].data.numpy()
        typical_images_as_tensor = self._get_dataset_as_tensor(self.typical_images_set)
        typical_images_to_calc_distance = typical_images_as_tensor[:, -1, :, :].data.numpy()

        # calc closest images in images space
        nn_classifier_images = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_images.fit(random_images_set_to_calc_distance.reshape(
            random_images_set_to_calc_distance.shape[0], -1))
        distances_images, indices_images = nn_classifier_images.kneighbors(
            typical_images_to_calc_distance.reshape(
                typical_images_to_calc_distance.shape[0], -1))

        # images in latent space
        random_images_latent_as_np = self._get_dataset_latent_space_as_np(random_images_as_tensor)
        typical_images_latent_as_np = self._get_dataset_latent_space_as_np(typical_images_as_tensor)

        # calc closest images in latent space
        nn_classifier_latent = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_latent.fit(random_images_latent_as_np)
        distances_latent, indices_latent = nn_classifier_latent.kneighbors(typical_images_latent_as_np)

        random_images_set_to_show = random_images_as_tensor[:, -1, :, :].unsqueeze(1)
        typical_images_set_to_show = typical_images_as_tensor[:, -1, :, :].unsqueeze(1)

        for i in range(len(typical_images_latent_as_np)):
            self.sample_images_and_log(random_images_set_to_show, torch.tensor(indices_latent[i]),
                                       f'closest_samples_latent_{i}', self.k)
            self.sample_images_and_log(random_images_set_to_show, torch.tensor(indices_images[i]),
                                       f'closest_samples_euclid_{i}', self.k)
            grid = torchvision.utils.make_grid(typical_images_set_to_show[i])
            self.logger.experiment.add_image(f'example images_{i}', grid, 0)


    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, 'validation')

    def evaluation_step(self, batch, name):
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
        x = x.float()
        # im_mean = x.view(batch_size, channels, -1).mean(2).view(batch_size, channels, 1, 1)
        # im_std = x.view(batch_size, channels, -1).std(2).view(batch_size, channels, 1, 1)
        x_normalized = x  # (x - im_mean) / (im_std)
        x_normalized = x_normalized[:, 0, :, :].unsqueeze(1)

        false_negatives_idxs = ((predicted != y) & (y == 1))
        self.sample_images_and_log(x_normalized, false_negatives_idxs, 'FN: positives, labeled wrong')

        false_positives_idxs = ((predicted != y) & (y == 0))
        self.sample_images_and_log(x_normalized, false_positives_idxs, 'FP: negatives, labeled wrong')

        true_negatives_idxs = ((predicted == y) & (y == 0))
        self.sample_images_and_log(x_normalized, true_negatives_idxs, 'TN: negatives, labeled right')

        true_positives_idxs = ((predicted == y) & (y == 1))
        self.sample_images_and_log(x_normalized, true_positives_idxs, 'TP: positives, labeled right')

        grid = torchvision.utils.make_grid(x_normalized)
        self.logger.experiment.add_image('example images', grid, 0)

    def sample_images_and_log(self, all_images: torch.tensor, one_hot_vector: torch.tensor,
                              title: str, number_to_log: int = 5):
        if torch.sum(one_hot_vector) > 0:
            actual_idxs = torch.nonzero(one_hot_vector)
            images = all_images[actual_idxs[:, 0]]
            num_images = images.shape[0]
            rand_indexes = [random.randint(0, num_images-1) for i in range(number_to_log)]
            sample_imgs = [images[rand_num] for rand_num in rand_indexes]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(title, grid, 0)

    def test_epoch_end(self, outputs: list):
        basic_dict = self.evaluation_epoch_end(outputs, 'test')
        return basic_dict

    def validation_epoch_end(self, outputs: list):
        basic_dict = self.evaluation_epoch_end(outputs, 'validation')
        return basic_dict

    def evaluation_epoch_end(self, outputs: list, name:str):
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x[name + '_acc'] for x in outputs]).mean()
        tensorboard_logs = {name + '_loss': avg_loss,
                            name + '_acc': avg_accuracy}
        return {'avg_'+name+'_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return self.optimizer
