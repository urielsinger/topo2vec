import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from shapely.geometry import Point

from topo2vec.common.geographic.geo_utils import check_if_point_in_range
from topo2vec.feature_extractors_pytorch.autoencoder import Autoencoder
from topo2vec.feature_extractors_pytorch.classifier import Classifier

logs_path = '/root/repositories/topo2vec/tb_logs'

model = Classifier()
logger = TensorBoardLogger(logs_path, name='classifier')
trainer = pl.Trainer(max_epochs = 50, logger = logger)
trainer.fit(model)



