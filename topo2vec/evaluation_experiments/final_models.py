import os

import torch

from topo2vec.constants import BASE_LOCATION
from topo2vec.modules import Classifier, Autoencoder, Superresolution, pix2pix
from topo2vec.models import BasicConvNetLatentDTM
from topo2vec.modules import Classifier, Autoencoder, Superresolution, pix2pix

FINAL_MODEL_DIR = BASE_LOCATION + 'data/final_model'
FINAL_HPARAMS = Classifier.get_args_parser().parse_args(
    ['--total_dataset_size', '2500',
     '--arch', 'BasicConvNetLatent',
     '--name', 'final_model',
     '--pytorch_module', 'Classifier',
     '--latent_space_size', '35',
     '--num_classes', '4',
     '--svm_classify_latent_space',
     '--original_radii', '[[8]]',
     '--radii', '[8]'
     ]
)


def load_model_from_file(
        final_model_name='classifier_BasicConvNetLatent_[8, 16, 24]_lr_0.0009704376798307045_size_10000_num_classes_4_latent_size_35.pt',
        hparams=FINAL_HPARAMS, final_dir=True):
    if final_dir:
        load_path = os.path.join(FINAL_MODEL_DIR, final_model_name)
    else:
        load_path = final_model_name
    final_model_classifier = eval(hparams.pytorch_module)(hparams)

    final_model_classifier.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    final_model_classifier.eval()
    return final_model_classifier


# classic_model_best = load_model_from_file()

classic_models_best_1000 = load_model_from_file(
    '/home/root/data/pretrained_models/multiclass_convnet_final/1000/classifier_BasicConvNetLatent_[8]_lr_0.0001_size_1000_num_classes_4_latent_size_35_train_all_resnet_False.pt',
    final_dir=False)

classic_models_best_5000 = load_model_from_file(
    '/home/root/data/pretrained_models/multiclass_convnet_final/5000/classifier_BasicConvNetLatent_[8]_lr_0.0001_size_5000_num_classes_4_latent_size_35_train_all_resnet_False_seed667.pt',
    final_dir=False)

classic_models_best_20000 = load_model_from_file(
    '/home/root/data/pretrained_models/multiclass_convnet_final/20000/classifier_BasicConvNetLatent_[8]_lr_0.0001_size_20000_num_classes_4_latent_size_35_train_all_resnet_False_seed666.pt',
    final_dir=False)

# classic_model_best = load_model_from_file()

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
topo_resnet_full = load_model_from_file(
    'classifier_TopoResNet_[224,224,224]_lr_0.0009200196036593708_size_10000_num_classes_4_latent_size_50_train_all_resnet_True.pt',
    RESNET_HPARAMS)

topo_resnet_transfer = load_model_from_file(
    'classifier_TopoResNet_[224,224,224]_lr_0.0009200196036593708_size_10000_num_classes_4_latent_size_50.pt',
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

amphib_autoencoder = load_model_from_file(
    'autoencoder_AdvancedAmphibAutoencoder_[8, 16, 24]_lr_0.0008_size_1000000_num_classes_4_latent_size_30_train_all_resnet_False.pt',
    AE_HPARAMS)

# SUPERRESOLUTION_HPARAMS = Classifier.get_args_parser().parse_args(
#     [
#         '--save_model',
#         '--index_in', '1',
#         '--index_out', '0',
#         '--learning_rate', '0.0015',
#         '--max_epochs', '600',
#         '--total_dataset_size', '20000',
#         '--arch', 'UNet',
#         '--svm_classify_latent_space',
#         '--knn_method_for_typical_choosing', 'regular',
#         '--name', 'outpainting',
#         '--pytorch_module', 'Superresolution',
#         '--random_set_size_for_svm', '2000',
#         '--latent_space_size', '600',
#         '--svm_classify_latent_space',
#         '--test_knn',
#         '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
#         '--radii', '[34, 136]',
#         '--upsample'
#      ]
# )
# superresolution_model_old = load_model_from_file('outpainting_UNet_[34, 136]_lr_0.0015_size_20000_num_classes_4_latent_size_600_train_all_resnet_False.pt',
#                                                  SUPERRESOLUTION_HPARAMS)


SUPERRESOLUTION_HPARAMS = Classifier.get_args_parser().parse_args(
    [
        '--save_model',
        '--index_in', '1',
        '--index_out', '0',
        '--learning_rate', '0.0015',
        '--max_epochs', '300',
        '--total_dataset_size', '100000',
        '--arch', 'UNet',
        '--svm_classify_latent_space',
        '--knn_method_for_typical_choosing', 'regular',
        '--name', 'Superresolution',
        '--pytorch_module', 'Superresolution',
        '--random_set_size_for_svm', '2000',
        '--latent_space_size', '600',
        '--svm_classify_latent_space',
        '--test_knn',
        '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
        '--radii', '[34, 136]',
        '--upsample'
    ]
)
superresolution_model = load_model_from_file(
    'Superresolution_UNet_[34, 136]_lr_0.0015_size_100000_num_classes_4_latent_size_600_train_all_resnet_False.pt',
    SUPERRESOLUTION_HPARAMS)

checkpoint = torch.load(os.path.join(FINAL_MODEL_DIR, "20200908_165032_model_best.pth.tar"))
contastive_model = BasicConvNetLatentDTM(None).cuda()

contastive_model.load_state_dict(checkpoint['model'])

AE_HPARAMS = Classifier.get_args_parser().parse_args(
    [
        '--save_model',
        '--index_in', '1',
        '--index_out', '0',
        '--learning_rate', '0.0015',
        '--max_epochs', '300',
        '--total_dataset_size', '100000',
        '--arch', 'UNet',
        '--svm_classify_latent_space',
        '--knn_method_for_typical_choosing', 'regular',
        '--name', 'Superresolution_upsample-False',
        '--pytorch_module', 'Superresolution',
        '--random_set_size_for_svm', '2000',
        '--latent_space_size', '600',
        '--svm_classify_latent_space',
        '--test_knn',
        '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
        '--radii', '[34, 136]',
    ]
)
unet_model = load_model_from_file(
    'Superresolution_upsample-False_UNet_[34, 136]_lr_0.0015_size_100000_num_classes_4_latent_size_600_train_all_resnet_False.pt',
    AE_HPARAMS)

PIX2PIX_HPARAMS = Classifier.get_args_parser().parse_args(
    [
        '--save_model',
        '--index_in', '1',
        '--index_out', '0',
        '--learning_rate', '0.0002',
        '--max_epochs', '300',
        '--total_dataset_size', '100000',
        '--arch', 'UNet',
        '--discriminator', 'Discriminator',
        '--svm_classify_latent_space',
        '--knn_method_for_typical_choosing', 'regular',
        '--name', 'pix2pix',
        '--pytorch_module', 'pix2pix',
        '--random_set_size_for_svm', '2000',
        '--latent_space_size', '128',
        '--svm_classify_latent_space',
        '--test_knn',
        '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
        '--radii', '[34, 136]',
        '--upsample'
    ]
)
pix2pix_model = load_model_from_file(
    'pix2pix_UNet_[34, 136]_lr_0.0002_size_100000_num_classes_4_latent_size_128_train_all_resnet_False.pt',
    PIX2PIX_HPARAMS)
