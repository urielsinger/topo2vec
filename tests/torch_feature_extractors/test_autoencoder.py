from unittest import TestCase

from topo2vec.feature_extractors_pytorch.autoencoder import Autoencoder

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from topo2vec.feature_extractors_pytorch.autoencoder import Autoencoder

logs_path = '/root/repositories/topo2vec/tb_logs'


class TestAutoencoder(TestCase):
    def test___init__(self):
        the_autoencoder = Autoencoder(val_num_points=10, train_num_points=10)
        logger = TensorBoardLogger(logs_path, name='my_model')
        trainer = pl.Trainer(max_epochs=1, logger=logger)
        trainer.fit(the_autoencoder)
