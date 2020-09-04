# accuracy in the validation polygon on 4 classes
import logging

import sklearn
import torch
from torch import nn
import sys

from topo2vec.modules.svm_on_latent_tester import svm_classifier_test

sys.path.append('/home/root')
from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.background import VALIDATION_HALF, CLASS_NAMES, CLASS_PATHS, TRAIN_HALF
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.evaluation_experiments.final_models import topo_resnet_transfer, topo_resnet_full, classic_model_best, superresolution_model



original_radiis = [[8,16,24]]
size_val = 200
RANDOM_SEED = 665
europe_dataset_ordinary = SeveralClassesDataset(original_radiis, VALIDATION_HALF, size_val, CLASS_PATHS, CLASS_NAMES,
                                                        'europe_dataset_for_eval_regular', random_seed=RANDOM_SEED)
europe_dataset_superresolution_train = SeveralClassesDataset([[34, 136]], TRAIN_HALF, 800, CLASS_PATHS, CLASS_NAMES,
                                                        'europe_dataset_for_train_superresolution', radii=[34, 136], random_seed=RANDOM_SEED)
europe_dataset_superresolution = SeveralClassesDataset([[34, 136]], VALIDATION_HALF, size_val, CLASS_PATHS, CLASS_NAMES,
                                                        'europe_dataset_for_eval_superresolution', radii=[34, 136], random_seed=RANDOM_SEED)
europe_dataset_resnet = SeveralClassesDataset(original_radiis, VALIDATION_HALF, size_val, CLASS_PATHS, CLASS_NAMES,
                                                        'europe_dataset_for_eval_resnet', [224], random_seed=RANDOM_SEED)

models = [classic_model_best, topo_resnet_full, topo_resnet_transfer, superresolution_model]
models_names = ["classic_model_best", "topo_resnet_full", "topo_resnet_transfer", "superresolution"]
val_datasets = [europe_dataset_ordinary, europe_dataset_resnet, europe_dataset_resnet, (europe_dataset_superresolution_train, europe_dataset_superresolution)]
auc_s =[]

with torch.no_grad():
    for model, name, dataset in zip(models, models_names, val_datasets):
        if name == "superresolution":
            auc = svm_classifier_test(model, dataset[0], dataset[1], dataset[1], 'superresolution')['svm_test_superresolution_auc']
        else:
            X, y = get_dataset_as_tensor(dataset)
            outputs, _ = model.forward(X)
            probas = nn.functional.softmax(outputs).numpy()
            y_np = y.numpy().squeeze()
            auc = sklearn.metrics.roc_auc_score(y_np, probas, multi_class='ovo')
        print(f'{name}:\t{auc}')
        auc_s.append(auc)
