import torchvision as torchvision
from pytorch_lightning import LightningModule

from topo2vec.datasets.random_dataset import RandomDataset
import torch
from pytorch_lightning import LightningModule
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from topo2vec.datasets.classification_dataset import ClassificationDataset
from topo2vec.CONSTANTS import *


class Classifier(LightningModule):
    def __init__(self, num_classes=1, radius=10, optimizer_cls=optim.Adam, loss_func=F.binary_cross_entropy,
                 learning_rate=1e-4, total_dataset_size=100000):
        '''
        An autoencoder which is a topo2vec
        Args:
            code_size: The hidden layer size
            radius: the radius around each point a coordinate should take
            optimizer_cls: the optimizer the AE uses
            loss_obj: the loss object that will be used
            learning_rate: lr
            num_epochs: number of epochs for the AE fit
        '''
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.radius = radius
        self.w = 2 * radius + 1
        self.h = 2 * radius + 1
        self.patch_size = self.w * self.h
        self.init_layers()
        self.loss_fn = loss_func
        self.optimizer = optimizer_cls(self.parameters(), lr=learning_rate)
        self.total_dataset_size = total_dataset_size
        self.build_dataset()

    def build_dataset(self):
        classification_dataset = ClassificationDataset(radii=[self.radius], first_class_path=N49_E05_STREAMS,
                                                       first_class_label='stream', outer_polygon=None)
        if self.total_dataset_size < len(classification_dataset):
            wanted_indices = list(range(0, int(self.total_dataset_size / 2), 1))
        else:
            print('asked for too large dataset')
        classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
        random_dataset = RandomDataset(radii=[self.radius], num_points=len(classification_dataset))
        dataset = ConcatDataset([classification_dataset, random_dataset])
        total_len = len(dataset)
        size_train = int(0.7 * total_len)
        size_val = int(0.2 * total_len)
        self.train_set, self.test_set, self.val_set = \
            random_split(dataset, [size_train, size_val, total_len - size_train - size_val])
        print(len(self.train_set))

    def init_layers(self, conv1_size=10, conv2_size=20, linear_size=50, kernel_size1=5,
                    kernel_size2=5, pool_kernel_1=2, pool_kernel_2=2):
        '''
        init the autoencoder layers.
        Args:
            conv1_size:
            conv2_size:
            linear_size:
            kernel_size1:
            kernel_size2:
            linear_size_dec:
            pool_kernel_1:
            pool_kernel_2:
        Returns:

        '''

        # encoder init
        self.enc_cnn_1 = nn.Conv2d(1, conv1_size, kernel_size=kernel_size1)
        self.max_pool_1 = nn.MaxPool2d(pool_kernel_1)
        self.enc_cnn_2 = nn.Conv2d(conv1_size, conv2_size, kernel_size=kernel_size2)
        self.max_pool_2 = nn.MaxPool2d(pool_kernel_2)

        # calc the end of the AE's size according to docs
        size_after_cnn_1 = self.w - kernel_size1 + 1
        size_after_relu_1 = int((size_after_cnn_1 + 2 * 0 - 1 * (pool_kernel_1 - 1) - 1) / pool_kernel_1 + 1)
        size_after_cnn_2 = size_after_relu_1 - kernel_size2 + 1
        size_after_relu_2 = int((size_after_cnn_2 + 2 * 0 - 1 * (pool_kernel_2 - 1) - 1) / pool_kernel_2 + 1)

        self.enc_linear_1 = nn.Linear(conv2_size * size_after_relu_2 ** 2, linear_size)
        self.enc_linear_2 = nn.Linear(linear_size, self.num_classes)

    def forward(self, images: torch.tensor) -> torch.tensor:
        '''
        Args:
            images:

        Returns:
        '''
        code = self.enc_cnn_1(images)
        code = F.selu(self.max_pool_1(code))

        code = self.enc_cnn_2(code)
        code = F.selu(self.max_pool_2(code))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = F.sigmoid(self.enc_linear_2(code))
        return code

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, num_workers=0, batch_size=1000)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=True, num_workers=0, batch_size=1000)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=True, num_workers=0, batch_size=1000)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        loss = self.loss_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        y_tag = (y_hat > 0.5)

        batch_size, channels, _, _ = x.size()
        x = x.float()
        im_mean = x.view(batch_size, channels, -1).mean(2).view(batch_size, channels, 1, 1)
        im_std = x.view(batch_size, channels, -1).std(2).view(batch_size, channels, 1, 1)
        x_normalized = x #(x - im_mean) / (im_std)

        false_negatives_idxs = ((y_tag != y) & (y == 1))
        self.sample_images_and_log(x_normalized, false_negatives_idxs, 'FN: positives, labeled wrong')

        false_positives_idxs = ((y_tag != y) & (y == 0))
        self.sample_images_and_log(x_normalized, false_positives_idxs, 'FP: negatives, labeled wrong')

        true_negatives_idxs = ((y_tag == y) & (y == 0))
        self.sample_images_and_log(x_normalized, true_negatives_idxs, 'TN: negatives, labeled right')

        true_positives_idxs = ((y_tag == y) & (y == 1))
        self.sample_images_and_log(x_normalized, true_positives_idxs, 'TP: positives, labeled right')

        grid = torchvision.utils.make_grid(x_normalized)
        self.logger.experiment.add_image('example images', grid, 0)

        loss = self.loss_fn(y_hat, y)
        accuracy = torch.tensor([float(torch.sum(y == y_tag)) / x.shape[0]])

        return {'val_loss': loss, 'val_acc': accuracy}

    def sample_images_and_log(self, all_images: torch.tensor, one_hot_vector: torch.tensor,
                              title: str, number_to_log: int = 5):
        if torch.sum(one_hot_vector) > 0:
            actual_idxs = torch.nonzero(one_hot_vector)
            images = all_images[actual_idxs[:, 0]]
            sample_imgs = images[:number_to_log]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(title, grid, 0)

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # TODO: add images, embeddings,

    def configure_optimizers(self):
        return self.optimizer
