from topo2vec.modules import Classifier, Autoencoder


class AutoencoderFineTuner(Classifier):
    def __init__(self, hparams, autoencoder: Autoencoder, retrain: bool = False):
        """
        an AutoencoderFineTuner to train on MultiRadiusDataset dataset that recieves:
        hparams: the hparams for the model - the model should be LinearLayerOnTop
        autoencoder: the Autoencoder class that will give the latent transformation
        """
        self.hparams = hparams
        self.hparams.autoencoder = autoencoder
        self.hparams.retrain = retrain
        super(AutoencoderFineTuner, self).__init__(self.hparams)
