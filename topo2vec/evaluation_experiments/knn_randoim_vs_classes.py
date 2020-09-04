import logging

import torch
import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset

from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.background import CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL, VALIDATION_HALF, TRAIN_HALF
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.evaluation_experiments.final_models import classic_model_best, topo_resnet_model, topo_resnet_full, \
    amphib_autoencoder, superresolution_model
from common.dataset_utils import get_paths_and_names_wanted
from topo2vec.modules.svm_on_latent_tester import svm_classifier_test
import numpy as np

original_radiis = [[8, 16, 24]]
test_set_size_for_knn = 100
RESNET_RADII = [224, 224, 224]
special_classes_for_validation = ['alpine_huts', 'peaks', 'airialway_stations', 'waterfalls']
train_set_size_for_knn = 5 * len(special_classes_for_validation)

class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

train_set_size_for_knn_max = 1000
train_set_size_for_knn_min = 2
step = 100
number_of_examples = list(range(train_set_size_for_knn_min, train_set_size_for_knn_max, step))
number_of_examples = [10, 100, 200, 300, 400, 500]

class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)
classifier = 'svm'
classifier_test = eval(classifier + '_classifier_test')

seeds = list(range(665, 675))
what_to_plot = 'auc'  # accuracy,auc


def knn_accuracy_on_dataset_in_latent_space(knn_classfier, dataset: Dataset, model) -> torch.Tensor:
    '''

    Args:
        knnClassifier: the knn classifier with which the classifying should be done
        dataset: the dataset we want to classify on
        model: the model which produces the latent space

    Returns: the efficient accuracy

    '''
    X, y = get_dataset_as_tensor(dataset)
    if model.hparams.use_gpu:
        X = X.cuda()
    _, latent = model.forward(X)
    if model.hparams.use_gpu:
        latent = latent.cpu()
    predicted = knn_classfier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    return accuracy


def knn_classifier_test(model, knn_train_dataset, knn_validation_dataset, test_dataset, type_of_knn_evaluation_name):
    X_train, y_train = get_dataset_as_tensor(knn_train_dataset)
    if model.hparams.use_gpu:
        X_train = X_train.cuda()
    _, latent_train = model.forward(X_train)
    knn_classifier = KNeighborsClassifier(n_neighbors=len(knn_train_dataset) // 2)

    if model.hparams.use_gpu:
        latent_train = latent_train.cpu()
        y_train = y_train.cpu()

    latent_train_numpy = latent_train.numpy()
    y_train_numpy = y_train.numpy()

    knn_classifier.fit(latent_train_numpy, y_train_numpy.ravel())

    train_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                             knn_train_dataset, model)

    validation_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                                  knn_validation_dataset, model)

    if len(test_dataset.class_names) == 2:
        test_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                                test_dataset, model)

        return {f'knn_train_{type_of_knn_evaluation_name}_accuracy': train_accuracy,
                f'knn_validation_{type_of_knn_evaluation_name}_accuracy': validation_accuracy,
                f'knn_test_{type_of_knn_evaluation_name}_accuracy': test_accuracy}


for special_class_path, special_class_name in zip(class_paths_special, class_names_special):
    knn_special_test_dataset = OneVsRandomDataset(original_radiis, test_set_size_for_knn, VALIDATION_HALF,
                                                  special_class_path, random_seed=665)
    knn_special_test_dataset_resnet = OneVsRandomDataset(original_radiis, test_set_size_for_knn, VALIDATION_HALF,
                                                         special_class_path, radii=RESNET_RADII, random_seed=665)
    knn_special_test_dataset_superresolution = OneVsRandomDataset([[34, 136]], test_set_size_for_knn, VALIDATION_HALF,
                                                                  special_class_path, radii=[34, 136], random_seed=665)

    on_latent_accuracy = []
    on_plain_accuracy = []
    resnet_accuracy = []
    resnet_transfer_accuracy = []
    amphib_ae_accuracy = []
    superresolution_accuracy = []

    for train_set_size_for_knn in tqdm.tqdm(number_of_examples):
        knn_list = []
        res_list = []
        classic_list = []
        res_full_list = []
        amphib_ae_list = []
        superresolution_list = []
        for seed in tqdm.tqdm(seeds):
            logging.info(seed)
            knn_special_train_dataset = OneVsRandomDataset(original_radiis, train_set_size_for_knn, TRAIN_HALF,
                                                           special_class_path, random_seed=seed)
            knn_special_train_dataset_resnet = OneVsRandomDataset(original_radiis, train_set_size_for_knn, TRAIN_HALF,
                                                                  special_class_path, radii=RESNET_RADII,
                                                                  random_seed=seed)
            knn_special_train_dataset_superresolution = OneVsRandomDataset([[34, 136]], train_set_size_for_knn,
                                                                           TRAIN_HALF,
                                                                           special_class_path, radii=[34, 136],
                                                                           random_seed=seed)

            with torch.no_grad():
                classifier_special_classes_test_log_dict_classic = \
                    classifier_test(classic_model_best, knn_special_train_dataset, knn_special_train_dataset,
                                    knn_special_test_dataset, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_resnet = \
                    classifier_test(topo_resnet_model, knn_special_train_dataset_resnet,
                                    knn_special_train_dataset_resnet,
                                    knn_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_resnet_transfer = \
                    classifier_test(topo_resnet_full, knn_special_train_dataset_resnet,
                                    knn_special_train_dataset_resnet,
                                    knn_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_amphib_autoencoder = \
                    classifier_test(amphib_autoencoder, knn_special_train_dataset, knn_special_train_dataset,
                                    knn_special_test_dataset, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_superresolution = \
                    classifier_test(superresolution_model, knn_special_train_dataset_superresolution,
                                    knn_special_train_dataset_superresolution,
                                    knn_special_test_dataset_superresolution, 'special')

            classic_list.append(
                classifier_special_classes_test_log_dict_classic[f'{classifier}_test_special_{what_to_plot}'])
            res_list.append(
                classifier_special_classes_test_log_dict_resnet[f'{classifier}_test_special_{what_to_plot}'])
            res_full_list.append(classifier_special_classes_test_log_dict_resnet_transfer[
                                     f'{classifier}_test_special_{what_to_plot}'])
            amphib_ae_list.append(classifier_special_classes_test_log_dict_amphib_autoencoder[
                                      f'{classifier}_test_special_{what_to_plot}'])
            superresolution_list.append(
                classifier_special_classes_test_log_dict_superresolution[f'{classifier}_test_special_{what_to_plot}'])

        logging.info(f'size of dataset = {knn_special_test_dataset}')
        logging.info('results on the latent space')
        logging.info(classic_list)
        on_latent_accuracy.append(np.mean(classic_list))

        logging.info('results on the resnet latent space')
        logging.info(classifier_special_classes_test_log_dict_resnet)
        resnet_accuracy.append(np.mean(res_list))
        logging.info(res_list)

        logging.info('results on the resnet transfer latent space')
        logging.info(classifier_special_classes_test_log_dict_resnet_transfer)
        resnet_transfer_accuracy.append(np.mean(res_full_list))
        logging.info(res_full_list)

        logging.info('results of knn trained')
        on_plain_accuracy.append(np.mean(knn_list))
        logging.info(knn_list)

        logging.info('results on the amphib ae latent space')
        logging.info(classifier_special_classes_test_log_dict_amphib_autoencoder)
        amphib_ae_accuracy.append(np.mean(amphib_ae_list))
        logging.info(amphib_ae_list)

        logging.info('results on the superresolution latent space')
        logging.info(classifier_special_classes_test_log_dict_superresolution)
        superresolution_accuracy.append(np.mean(superresolution_list))
        logging.info(superresolution_accuracy)

    import matplotlib.pyplot as plt

    plt.plot(number_of_examples, on_latent_accuracy, label=f'classicnet on latent {what_to_plot}')
    plt.plot(number_of_examples, resnet_accuracy, label=f'resnet on latent {what_to_plot}')
    plt.plot(number_of_examples, resnet_transfer_accuracy, label=f'resnet full transfer on latent {what_to_plot}')
    plt.plot(number_of_examples, amphib_ae_accuracy, label=f'amphib ae {what_to_plot}')
    plt.plot(number_of_examples, on_plain_accuracy, label=f'on plain {what_to_plot}')
    plt.plot(number_of_examples, superresolution_accuracy, label=f'superresolution {what_to_plot}')

    plt.legend()
    plt.title(f'{classifier} - few shot: {special_class_name} vs. random')
    plt.xlabel('number of poositive examples')
    plt.ylabel(f'{what_to_plot}')
    plt.savefig(f'results_evaluation_experiments/{special_class_name}_{what_to_plot}_{classifier}_one.png')
    plt.clf()
