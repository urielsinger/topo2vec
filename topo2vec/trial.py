import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from topo2vec.feature_extractors_pytorch.classifier import Classifier

logs_path = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs'

model = Classifier(radii=[24])
logger = TensorBoardLogger(logs_path, name='classifier')
trainer = pl.Trainer(max_epochs=50, logger=logger)
trainer.fit(model)
