import random
from typing import List

import torchvision as torchvision
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

import topo2vec.models as models


class Classifier(LightningModule):
    def __init__(self, validation_dataset: Dataset, train_dataset: Dataset,
                 loss_func, optimizer_cls, arch: str = 'simpleconvnet',
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

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, num_workers=0, batch_size=1000)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=True, num_workers=0, batch_size=1000)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x.float())
        loss = self.loss_fn(outputs.float(), y.squeeze().long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x.float())
        _, predicted = torch.max(outputs.data, 1)
        batch_size, channels, _, _ = x.size()
        sumy = torch.sum(predicted == y)
        accuracy = torch.tensor([float(torch.sum(predicted == y.squeeze())) / batch_size])
        loss = self.loss_fn(outputs.float(), y.squeeze().long())

        if self.num_classes == 2:
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

        return {'val_loss': loss, 'val_acc': accuracy}

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

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return self.optimizer
