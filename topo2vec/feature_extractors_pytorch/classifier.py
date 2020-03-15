from pytorch_lightning import LightningModule

from topo2vec.datasets.random_dataset import RandomDataset


class classifier(LightningModule):
    def __init__(self):
        super(classifier, self).__init__()

import torch
from pytorch_lightning import LightningModule
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split, SubsetRandomSampler

from topo2vec.datasets.classification_dataset import ClassificationDataset


class Classifier(LightningModule):
    def __init__(self, num_classes = 1, radius = 10, optimizer_cls = optim.Adam, loss_func = F.binary_cross_entropy,
                 learning_rate = 1e-4, total_dataset_size = 100000):
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
        self.optimizer = optimizer_cls(self.parameters(), lr = learning_rate)
        self.total_dataset_size = total_dataset_size
        self.build_dataset()

    def build_dataset(self):
        classification_dataset = ClassificationDataset(radii = [self.radius], first_class_path=,
                                        first_class_label = 'stream', outer_polygon='yes')
        if self.total_dataset_size < len(classification_dataset):
            wanted_indices = list(range(0,int(self.total_dataset_size/2),1))
        else:
            print('asked for too large dataset')
        classification_dataset = torch.utils.data.Subset(classification_dataset, wanted_indices)
        random_dataset = RandomDataset(radii = [self.radius], num_points = len(classification_dataset))
        dataset = ConcatDataset([classification_dataset, random_dataset])
        total_len = len(dataset)
        size_train = int(0.7 * total_len)
        size_val = int(0.2 * total_len)
        self.train_set, self.test_set, self.val_set = \
            random_split(dataset,[size_train, size_val, total_len - size_train - size_val])
        print(len(self.train_set))


    def init_layers(self, conv1_size = 10, conv2_size = 20, linear_size = 50, kernel_size1 = 5,
                    kernel_size2 = 5, pool_kernel_1 = 2, pool_kernel_2 = 2):
        '''
        init the autoencoder layers.
        Args:
            conv1_size:
            conv2_size:
            linear_size:
            kernel_size1:
            kernel_size2:
            linear_size_dec:
        Returns:

        '''

        # encoder init
        self.enc_cnn_1 = nn.Conv2d(1, conv1_size, kernel_size = kernel_size1)
        self.max_pool_1 = nn.MaxPool2d(pool_kernel_1)
        self.enc_cnn_2 = nn.Conv2d(conv1_size, conv2_size, kernel_size = kernel_size2)
        self.max_pool_2 = nn.MaxPool2d(pool_kernel_2)

        # calc the end of the AE's size according to docs
        size_after_cnn_1 = self.w - kernel_size1 + 1
        size_after_relu_1 = int((size_after_cnn_1 + 2 * 0 - 1 * (pool_kernel_1 - 1) - 1)/pool_kernel_1 + 1)
        size_after_cnn_2 = size_after_relu_1 - kernel_size2 + 1
        size_after_relu_2 =  int((size_after_cnn_2 + 2 * 0 - 1 * (pool_kernel_2 - 1) - 1)/pool_kernel_2 + 1)

        self.enc_linear_1 = nn.Linear(conv2_size * size_after_relu_2**2, linear_size)
        self.enc_linear_2 = nn.Linear(linear_size, self.num_classes)


    def forward(self, images:torch.tensor)->torch.tensor:
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
        return DataLoader(self.val_set)

    def test_dataloader(self):
        return DataLoader(self.test_set)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = torch.tensor([float(torch.sum(y == (y_hat>0.5)) / x.shape[0])])

        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy =  torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    #TODO: add images, embeddings,


    def configure_optimizers(self):
        return self.optimizer


