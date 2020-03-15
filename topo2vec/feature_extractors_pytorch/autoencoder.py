import torch
from pytorch_lightning import LightningModule
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from topo2vec.datasets.random_dataset import RandomDataset


class Autoencoder(LightningModule):
    def __init__(self, code_size = 10, radius = 10, optimizer_cls = optim.Adam, loss_func = F.mse_loss,
                 learning_rate = 0.002, train_num_points = 10000, val_num_points = 1000):
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
        super(Autoencoder, self).__init__()
        self.code_size = code_size
        self.radius = radius
        self.w = 2 * radius + 1
        self.h = 2 * radius + 1
        self.patch_size = self.w * self.h
        self.init_layers()
        self.loss_fn = loss_func
        self.optimizer = optimizer_cls(self.parameters(), lr = learning_rate)
        self.train_num_point = train_num_points
        self.val_num_points = val_num_points


    def init_layers(self, conv1_size = 10, conv2_size = 20, linear_size = 50, kernel_size1 = 5,
                    kernel_size2 = 5, linear_size_dec = 160, pool_kernel_1 = 2, pool_kernel_2 = 2):
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
        self.enc_linear_2 = nn.Linear(linear_size, self.code_size)

        # decoder init
        self.dec_linear_1 = nn.Linear(self.code_size, linear_size_dec)
        self.dec_linear_2 = nn.Linear(linear_size_dec, self.patch_size)


    def encode(self, images:torch.tensor)->torch.tensor:
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
        code = self.enc_linear_2(code)
        return code


    def decode(self, codes: torch.tensor)->torch.tensor:
        '''

        Args:
            codes: the hidden codes to unleash (num_samples, code_size)

        Returns: A (num_samples, 1, W, H) array

        '''
        out = F.selu(self.dec_linear_1(codes))
        out = self.dec_linear_2(out)
        out = out.view([codes.size(0), 1, self.w, self.h])
        return out

    def forward(self, images: torch.tensor) -> torch.tensor:
        '''

        Args:
            images:

        Returns:

        '''
        codes = self.encode(images)
        images = self.decode(codes)

        return images, codes

    def train_dataloader(self):
        dataset = RandomDataset(self.train_num_point)
        return DataLoader(dataset, shuffle=True, num_workers=0)

    def val_dataloader(self):
        dataset = RandomDataset(self.val_num_points)
        return  DataLoader(dataset)

    def test_dataloader(self):
        num_point = 1000
        dataset = RandomDataset(num_point)
        return  DataLoader(dataset)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        return {'loss': loss, 'log': {'train_loss', loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return self.optimizer


