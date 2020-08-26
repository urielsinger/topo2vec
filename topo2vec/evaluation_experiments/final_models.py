import os

import torch

from topo2vec.constants import BASE_LOCATION
from topo2vec.modules import Classifier, Autoencoder

FINAL_MODEL_DIR = BASE_LOCATION + 'data/final_model'
FINAL_HPARAMS = Classifier.get_args_parser().parse_args(
    ['--total_dataset_size', '2500',
     '--arch', 'BasicConvNetLatent',
     '--name', 'final_model',
     '--pytorch_module', 'Classifier',
     '--latent_space_size', '35',
     '--num_classes', '4',
     '--svm_classify_latent_space',
     ]
)

def load_model_from_file(final_model_name = 'classifier_BasicConvNetLatent_[8, 16, 24]_lr_0.0009704376798307045_size_10000_num_classes_4_latent_size_35.pt',
                         hparams=FINAL_HPARAMS):
    load_path = os.path.join(FINAL_MODEL_DIR, final_model_name)
    if hparams.pytorch_module =='Classifier':
        final_model_classifier = Classifier(hparams)
    else:
        final_model_classifier = Autoencoder(hparams)
    final_model_classifier.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    final_model_classifier.eval()
    return final_model_classifier

classic_model_best = load_model_from_file()

RESNET_HPARAMS = Classifier.get_args_parser().parse_args(
    [
    '--total_dataset_size', '2500',
     '--arch', 'TopoResNet',
     '--name', 'resnet_model',
     '--pytorch_module', 'Classifier',
    '--latent_space_size', '128',
    '--svm_classify_latent_space',
    '--radii', '[224,224,224]'
     ]
)
topo_resnet_model = Classifier(RESNET_HPARAMS)

RESNET_HPARAMS_TRANSFER = Classifier.get_args_parser().parse_args(
    [
    '--total_dataset_size', '2500',
     '--arch', 'TopoResNet',
     '--name', 'resnet_model',
     '--pytorch_module', 'Classifier',
    '--latent_space_size', '512',
    '--svm_classify_latent_space',
    '--radii', '[224,224,224]'
     ]
)
topo_resnet_full = load_model_from_file('classifier_TopoResNet_[224,224,224]_lr_0.0009200196036593708_size_10000_num_classes_4_latent_size_50_train_all_resnet_True.pt',
                                        RESNET_HPARAMS)

topo_resnet_transfer = load_model_from_file('classifier_TopoResNet_[224,224,224]_lr_0.0009200196036593708_size_10000_num_classes_4_latent_size_50.pt',
                                                 RESNET_HPARAMS)

AE_HPARAMS = Classifier.get_args_parser().parse_args(
    [
    '--total_dataset_size', '10',
     '--arch', 'AdvancedAmphibAutoencoder',
     '--name', 'resnet_model',
     '--pytorch_module', 'Autoencoder',
    '--latent_space_size', '30',
    '--svm_classify_latent_space',
     ]
)
# amphib_autoencoder = load_model_from_file('autoencoder_AdvancedAmphibAutoencoder_[8, 16, 24]_lr_0.0008_size_10_num_classes_4_latent_size_50_train_all_resnet_False.pt',
#                                                  AE_HPARAMS)

amphib_autoencoder = load_model_from_file('autoencoder_AdvancedAmphibAutoencoder_[8, 16, 24]_lr_0.0008_size_1000000_num_classes_4_latent_size_30_train_all_resnet_False.pt',
                                                 AE_HPARAMS)