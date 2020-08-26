import logging

import torch

from topo2vec.background import CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL, VALIDATION_HALF, TRAIN_HALF
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.evaluation_experiments.final_models import classic_model_best, topo_resnet_model, topo_resnet_full, \
    amphib_autoencoder
from topo2vec.evaluation_experiments.svm_on_plain_experiment import test_svm_on_plain
from common.dataset_utils import get_paths_and_names_wanted
from topo2vec.modules.svm_on_latent_tester import svm_classifier_test
import numpy as np

original_radiis = [[8,16,24]]
test_set_size_for_svm = 100
RESNET_RADII = [224, 224, 224]
special_classes_for_validation = ['alpine_huts', 'peaks', 'airialway_stations', 'waterfalls']
train_set_size_for_svm = 5 * len(special_classes_for_validation)

class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

train_set_size_for_svm_max = 10
train_set_size_for_svm_min = 2
class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

seeds = list(range(665,675))
what_to_plot = 'accuracy'

for special_class_path, special_class_name in zip(class_paths_special, class_names_special):
    svm_special_test_dataset = OneVsRandomDataset(original_radiis, test_set_size_for_svm, VALIDATION_HALF, special_class_path, random_seed=665)
    svm_special_test_dataset_resnet = OneVsRandomDataset(original_radiis, test_set_size_for_svm, VALIDATION_HALF, special_class_path, radii=RESNET_RADII, random_seed=665)

    on_latent_accuracy = []
    on_plain_accuracy = []
    resnet_accuracy = []
    resnet_transfer_accuracy = []
    amphib_ae_accuracy = []
    number_of_examples = list(range(train_set_size_for_svm_min, train_set_size_for_svm_max))
    for train_set_size_for_svm in number_of_examples:
        svm_list = []
        res_list = []
        classic_list = []
        res_full_list = []
        amphib_ae_list = []

        for seed in seeds:
            logging.info(seed)
            svm_special_train_dataset = OneVsRandomDataset(original_radiis, train_set_size_for_svm, TRAIN_HALF, special_class_path, random_seed=seed)
            svm_special_train_dataset_resnet = OneVsRandomDataset(original_radiis, train_set_size_for_svm, TRAIN_HALF,
                                                                  special_class_path, radii=RESNET_RADII, random_seed=seed)

            with torch.no_grad():
                svm_classifier_special_classes_test_log_dict_classic = \
                    svm_classifier_test(classic_model_best, svm_special_train_dataset, svm_special_train_dataset, svm_special_test_dataset, 'special')

            with torch.no_grad():
                svm_classifier_special_classes_test_log_dict_resnet = \
                    svm_classifier_test(topo_resnet_model, svm_special_train_dataset_resnet, svm_special_train_dataset_resnet,
                                        svm_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                svm_classifier_special_classes_test_log_dict_resnet_transfer = \
                    svm_classifier_test(topo_resnet_full, svm_special_train_dataset_resnet, svm_special_train_dataset_resnet,
                                        svm_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                svm_classifier_special_classes_test_log_dict_amphib_autoencoder = \
                    svm_classifier_test(amphib_autoencoder, svm_special_train_dataset, svm_special_train_dataset,
                                        svm_special_test_dataset, 'special')

            svm_classifier_special_classes_test_log_dict_svm = test_svm_on_plain(svm_special_train_dataset, svm_special_test_dataset)
            svm_list.append(svm_classifier_special_classes_test_log_dict_svm[f'svm_test_special_{what_to_plot}'])
            classic_list.append(svm_classifier_special_classes_test_log_dict_classic[f'svm_test_special_{what_to_plot}'])
            res_list.append(svm_classifier_special_classes_test_log_dict_resnet[f'svm_test_special_{what_to_plot}'])
            res_full_list.append(svm_classifier_special_classes_test_log_dict_resnet_transfer[f'svm_test_special_{what_to_plot}'])
            amphib_ae_list.append(svm_classifier_special_classes_test_log_dict_amphib_autoencoder[f'svm_test_special_{what_to_plot}'])

        logging.info(f'size of dataset = {svm_special_test_dataset}')
        logging.info('results on the latent space')
        logging.info(classic_list)
        on_latent_accuracy.append(np.mean(classic_list))

        logging.info('results on the resnet latent space')
        logging.info(svm_classifier_special_classes_test_log_dict_resnet)
        resnet_accuracy.append(np.mean(res_list))
        logging.info(res_list)

        logging.info('results on the resnet transfer latent space')
        logging.info(svm_classifier_special_classes_test_log_dict_resnet_transfer)
        resnet_transfer_accuracy.append(np.mean(res_full_list))
        logging.info(res_full_list)

        logging.info('results of SVM trained')
        on_plain_accuracy.append(np.mean(svm_list))
        logging.info(svm_list)

        logging.info('results on the amphib ae latent space')
        logging.info(svm_classifier_special_classes_test_log_dict_amphib_autoencoder)
        amphib_ae_accuracy.append(np.mean(amphib_ae_list))
        logging.info(amphib_ae_list)

    import matplotlib.pyplot as plt
    plt.plot(number_of_examples, on_latent_accuracy, label=f'classicnet on latent {what_to_plot}')
    plt.plot(number_of_examples, resnet_accuracy, label=f'resnet on latent {what_to_plot}')
    plt.plot(number_of_examples, resnet_transfer_accuracy,  label=f'resnet full transfer on latent {what_to_plot}')
    plt.plot(number_of_examples, amphib_ae_accuracy, label=f'amphib ae {what_to_plot}')
    plt.plot(number_of_examples, on_plain_accuracy, label=f'on plain {what_to_plot}')

    plt.legend()
    plt.title(f'few shot: {special_class_name} vs. random')
    plt.xlabel('number of poositive examples')
    plt.ylabel(f'{what_to_plot}')
    plt.savefig(f'{special_class_name}_{what_to_plot}_regular_w_amphib.png')
    plt.clf()