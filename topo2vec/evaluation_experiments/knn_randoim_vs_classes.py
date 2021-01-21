import logging
import os
import random
import time
from pathlib import Path

import torch
import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.backends import cudnn
from torch.utils.data import Dataset

from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.background import CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL, VALIDATION_HALF, TRAIN_HALF, \
    PROJECT_SCALES_DICT
from topo2vec.constants import BASE_LOCATION
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.evaluation_experiments.final_models import topo_resnet_model, topo_resnet_full, \
    amphib_autoencoder, superresolution_model, contastive_model, classic_models_best_1000, classic_models_best_5000, \
    classic_models_best_20000, pix2pix_model, unet_model
from common.dataset_utils import get_paths_and_names_wanted

from topo2vec.modules import Classifier
from topo2vec.modules.svm_on_latent_tester import knn_classifier_test, svm_classifier_test
import numpy as np

EXP_LOGS_PATH = BASE_LOCATION + 'tb_logs/scale_experiment'

original_radii = 8
original_radiis = [[8, 16, 24]]
test_set_size_for_knn = 1000
RESNET_RADII = [224, 224, 224]
special_classes_for_validation_others = ['alpine_huts', 'antenas', 'airialway_stations']  # 350 achievable

special_classes_for_validation_topo = ['rivers', 'saddles', 'peaks', 'cliffs']  # 1000 achievable

special_classes_for_validation_semi_topo = ['waterfalls', 'sinkholes']  # 450 achievable

special_classes_for_validation = special_classes_for_validation_topo

# train_set_size_for_knn = 5 * len(special_classes_for_validation)

# train_set_size_for_knn_max = 1000
# train_set_size_for_knn_min = 2
# step = 1000
# number_of_examples = list(range(train_set_size_for_knn_min, train_set_size_for_knn_max, step))
number_of_examples_topo = [1000]
number_of_examples_semi_topo = [100, 200, 400]  # [10, 20, 50, 100, 200, 400]
number_of_examples_semi_others = [50, 100, 150, 350]
number_of_examples = number_of_examples_topo

class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)
classifier = 'svm'
classifier_test = eval(classifier + '_classifier_test')

seeds = list(range(660, 670))
what_to_plot = 'accuracy'  # accuracy,auc


def knn_accuracy_on_dataset_in_latent_space(knn_classfier, dataset: Dataset, model) -> torch.Tensor:
    '''

    Args:
        knnClassifier: the knn classifier with which the classifying should be done
        dataset: the dataset we want to classify on
        model: the model which produces the latent space

    Returns: the efficient accuracy

    '''
    X, y = get_dataset_as_tensor(dataset)
    if model.hparams.contrastive:
        X = X.cuda()
    _, latent = model.forward(X)
    if model.hparams.contrastive:
        latent = latent.cpu()
    predicted = knn_classfier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    return accuracy


def knn_classifier_test(model, knn_train_dataset, knn_validation_dataset, test_dataset, type_of_knn_evaluation_name):
    X_train, y_train = get_dataset_as_tensor(knn_train_dataset)
    if model.hparams.contrastive:
        X_train = X_train.cuda()
    _, latent_train = model.forward(X_train)
    knn_classifier = KNeighborsClassifier(n_neighbors=len(knn_train_dataset) // 2)

    if model.hparams.contrastive:
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


SAVE_PATH_knn_random = 'results/knn_random'

baes_path = os.path.join(BASE_LOCATION, SAVE_PATH_knn_random, 'date' + str(time.strftime('%Y-%m-%d %H-%M-%S')))


def get_accuracy_small_net(train_set_size_for_knn, class_name, class_path, random_seed, test_dataset):
    class_json_location = class_path
    max_epochs = 50
    classifier_parser = Classifier.get_args_parser()
    parse_args_list = [
        '--save_model',
        '--learning_rate', '0.0003200196036593708',
        '--total_dataset_size', f'{train_set_size_for_knn}',
        '--max_epochs', f'{max_epochs}',
        '--arch', 'BasicConvNetLatent',
        '--name', 'classifier',
        '--latent_space_size', '35',
        '--scale_exp',
        '--scale_exp_class_name', class_name,
        '--scale_exp_class_path', class_json_location,
        '--num_classes', '2',
        '--radii', f'[8]',
        '--use_gpu'
    ]

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.deterministic = True
    np.random.seed(random_seed)
    np.random.seed(random_seed)
    import pytorch_lightning as pl
    logger = TensorBoardLogger(EXP_LOGS_PATH, name=f'{random_seed}_{class_name}_train_size{train_set_size_for_knn}')

    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
    classifier_current_args = classifier_parser.parse_args(parse_args_list +
                                                           ['--scale_exp_random_seed', str(random_seed),
                                                            '--original_radii', f'[[8]]',
                                                            ])
    model = Classifier(classifier_current_args)
    trainer.fit(model)
    test_accuracy = model.get_accuracy_for_small_dataset(test_dataset)


rescaling = ''
for special_class_path, special_class_name in zip(class_paths_special, class_names_special):
    print(f'making datasets for {special_class_name}')
    if False and special_class_name in PROJECT_SCALES_DICT.dict:
        rescaling = 'rescaling'
        original_radii = PROJECT_SCALES_DICT.dict[special_class_name]
        original_radiis = [[original_radii, 2 * original_radii, 3 * original_radii]]

    knn_special_test_dataset = OneVsRandomDataset(original_radiis, test_set_size_for_knn, VALIDATION_HALF,
                                                  special_class_path, random_seed=665, radii=[8])
    knn_special_test_dataset_flat = OneVsRandomDataset([[original_radii]], test_set_size_for_knn, VALIDATION_HALF,
                                                       special_class_path, random_seed=665, radii=[8])
    knn_special_test_dataset_resnet = OneVsRandomDataset(original_radiis, test_set_size_for_knn, VALIDATION_HALF,
                                                         special_class_path, radii=RESNET_RADII, random_seed=665)
    knn_special_test_dataset_superresolution = OneVsRandomDataset(
        [[2 * (2 * original_radii + 1), 8 * (2 * original_radii + 1)]],
        test_set_size_for_knn, VALIDATION_HALF,
        special_class_path, radii=[34, 136], random_seed=665)
    print('made the datasets')
    on_latent_accuracy = []
    classic_1000_accuracy = []
    classic_1000_std = []
    classic_5000_accuracy = []
    classic_5000_std = []
    classic_20000_accuracy = []
    classic_20000_std = []
    pix2pix_accuracy = []
    pix2pix_std = []
    special_classic_accuracy = []
    special_classic_std = []
    unet_accuracy = []
    unet_std = []
    contrastive_on_latent_accuracy = []
    on_plain_accuracy = []
    resnet_accuracy = []
    resnet_transfer_accuracy = []
    amphib_ae_accuracy = []
    superresolution_accuracy = []
    on_latent_std = []
    contrastive_on_latent_std = []
    on_plain_std = []
    resnet_std = []
    resnet_transfer_std = []
    amphib_ae_std = []
    superresolution_std = []
    for train_set_size_for_knn in tqdm.tqdm(number_of_examples):
        knn_list = []
        pix2pix_list = []
        unet_list = []
        special_classic_list = []
        res_list = []
        classic_1000_list = []
        classic_5000_list = []
        classic_20000_list = []
        contrastive_list = []
        res_full_list = []
        amphib_ae_list = []
        superresolution_list = []
        for seed in tqdm.tqdm(seeds):
            print(f'making dataset for seed:{seed}')
            knn_special_train_dataset = OneVsRandomDataset(original_radiis, train_set_size_for_knn, TRAIN_HALF,
                                                           special_class_path, random_seed=seed, radii=[8])

            knn_special_train_dataset_flat = OneVsRandomDataset([[original_radii]], train_set_size_for_knn, TRAIN_HALF,
                                                                special_class_path, random_seed=seed, radii=[8])
            knn_special_train_dataset_resnet = OneVsRandomDataset(original_radiis, train_set_size_for_knn, TRAIN_HALF,
                                                                  special_class_path, radii=RESNET_RADII,
                                                                  random_seed=seed)
            knn_special_train_dataset_superresolution = OneVsRandomDataset(
                [[2 * (2 * original_radii + 1), 8 * (2 * original_radii + 1)]], train_set_size_for_knn,
                TRAIN_HALF,
                special_class_path, radii=[34, 136],
                random_seed=seed)
            # special_classic_list.append(
            #     get_accuracy_small_net(train_set_size_for_knn, special_class_name, special_class_path, seed,
            #                            knn_special_test_dataset_flat))
            with torch.no_grad():
                classifier_special_classes_test_log_dict_plain = \
                    classifier_test(None, knn_special_train_dataset_flat, knn_special_test_dataset_flat,
                                    knn_special_test_dataset_flat, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_resnet = \
                    classifier_test(topo_resnet_model, knn_special_train_dataset_resnet,
                                    knn_special_test_dataset_resnet,
                                    knn_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_resnet_transfer = \
                    classifier_test(topo_resnet_full, knn_special_train_dataset_resnet,
                                    knn_special_test_dataset_resnet,
                                    knn_special_test_dataset_resnet, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_amphib_autoencoder = \
                    classifier_test(amphib_autoencoder, knn_special_train_dataset, knn_special_test_dataset,
                                    knn_special_test_dataset, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_contrastive = \
                    classifier_test(contastive_model, knn_special_train_dataset_flat, knn_special_test_dataset_flat,
                                    knn_special_test_dataset_flat, 'special', use_gpu=True)

            with torch.no_grad():
                classifier_special_classes_test_log_dict_classic_best_1000 = \
                    classifier_test(classic_models_best_1000, knn_special_train_dataset_flat,
                                    knn_special_test_dataset_flat,
                                    knn_special_test_dataset_flat, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_classic_best_5000 = \
                    classifier_test(classic_models_best_5000, knn_special_train_dataset_flat,
                                    knn_special_test_dataset_flat,
                                    knn_special_test_dataset_flat, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_classic_best_20000 = \
                    classifier_test(classic_models_best_20000, knn_special_train_dataset_flat,
                                    knn_special_test_dataset_flat,
                                    knn_special_test_dataset_flat, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_superresolution = \
                    classifier_test(superresolution_model, knn_special_train_dataset_superresolution,
                                    knn_special_test_dataset_superresolution,
                                    knn_special_test_dataset_superresolution, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_pix2pix = \
                    classifier_test(pix2pix_model, knn_special_train_dataset_superresolution,
                                    knn_special_test_dataset_superresolution,
                                    knn_special_test_dataset_superresolution, 'special')

            with torch.no_grad():
                classifier_special_classes_test_log_dict_unet = \
                    classifier_test(unet_model, knn_special_train_dataset_superresolution,
                                    knn_special_test_dataset_superresolution,
                                    knn_special_test_dataset_superresolution, 'special')

            knn_list.append(classifier_special_classes_test_log_dict_plain[f'{classifier}_test_special_{what_to_plot}'])
            contrastive_list.append(
                classifier_special_classes_test_log_dict_contrastive[f'{classifier}_test_special_{what_to_plot}'])
            classic_1000_list.append(
                classifier_special_classes_test_log_dict_classic_best_1000[f'{classifier}_test_special_{what_to_plot}'])
            classic_5000_list.append(
                classifier_special_classes_test_log_dict_classic_best_5000[f'{classifier}_test_special_{what_to_plot}'])
            classic_20000_list.append(
                classifier_special_classes_test_log_dict_classic_best_20000[
                    f'{classifier}_test_special_{what_to_plot}'])
            res_list.append(
                classifier_special_classes_test_log_dict_resnet[f'{classifier}_test_special_{what_to_plot}'])
            res_full_list.append(classifier_special_classes_test_log_dict_resnet_transfer[
                                     f'{classifier}_test_special_{what_to_plot}'])
            amphib_ae_list.append(classifier_special_classes_test_log_dict_amphib_autoencoder[
                                      f'{classifier}_test_special_{what_to_plot}'])
            superresolution_list.append(
                classifier_special_classes_test_log_dict_superresolution[f'{classifier}_test_special_{what_to_plot}'])
            pix2pix_list.append(
                classifier_special_classes_test_log_dict_pix2pix[f'{classifier}_test_special_{what_to_plot}'])
            unet_list.append(
                classifier_special_classes_test_log_dict_unet[f'{classifier}_test_special_{what_to_plot}'])

        # logging.info(f'size of dataset = {len(knn_special_test_dataset)}')
        # logging.info('special classic_list on the classic 1000 latent space')
        # logging.info(special_classic_list)
        # special_classic_accuracy.append(np.mean(special_classic_list))
        # special_classic_std.append(np.std(special_classic_list))

        logging.info(f'size of dataset = {len(knn_special_test_dataset)}')
        logging.info('classic_1000_list on the classic 1000 latent space')
        logging.info(classic_1000_list)
        classic_1000_accuracy.append(np.mean(classic_1000_list))
        classic_1000_std.append(np.std(classic_1000_list))

        logging.info(f'size of dataset = {len(knn_special_test_dataset)}')
        logging.info('results on the classic 1000 latent space')
        logging.info(classic_5000_list)
        classic_5000_accuracy.append(np.mean(classic_5000_list))
        classic_5000_std.append(np.std(classic_5000_list))

        logging.info(f'size of dataset = {len(knn_special_test_dataset)}')
        logging.info('results on the classic 1000 latent space')
        logging.info(classic_20000_list)
        classic_20000_accuracy.append(np.mean(classic_20000_list))
        classic_20000_std.append(np.std(classic_20000_list))

        logging.info(f'size of dataset = {len(knn_special_test_dataset)}')
        logging.info('results on the latent space')
        logging.info(contrastive_list)
        contrastive_on_latent_accuracy.append(np.mean(contrastive_list))
        contrastive_on_latent_std.append(np.std(contrastive_list))

        logging.info('results on the resnet latent space')
        logging.info(classifier_special_classes_test_log_dict_resnet)
        resnet_accuracy.append(np.mean(res_list))
        resnet_std.append(np.std(res_list))

        logging.info(res_list)

        logging.info('results on the resnet transfer latent space')
        logging.info(classifier_special_classes_test_log_dict_resnet_transfer)
        resnet_transfer_accuracy.append(np.mean(res_full_list))
        resnet_transfer_std.append(np.std(res_full_list))

        logging.info(res_full_list)

        logging.info('results of knn trained')
        on_plain_accuracy.append(np.mean(knn_list))
        on_plain_std.append(np.std(knn_list))
        logging.info(knn_list)

        logging.info('results on the amphib ae latent space')
        logging.info(classifier_special_classes_test_log_dict_amphib_autoencoder)
        amphib_ae_accuracy.append(np.mean(amphib_ae_list))
        amphib_ae_std.append(np.std(amphib_ae_list))
        logging.info(amphib_ae_list)

        logging.info('results on the superresolution latent space')
        logging.info(classifier_special_classes_test_log_dict_superresolution)
        superresolution_accuracy.append(np.mean(superresolution_list))
        superresolution_std.append(np.std(superresolution_list))
        logging.info(superresolution_accuracy)

        logging.info('results on the pix2pix latent space')
        logging.info(classifier_special_classes_test_log_dict_superresolution)
        pix2pix_accuracy.append(np.mean(pix2pix_list))
        pix2pix_std.append(np.std(pix2pix_list))
        logging.info(pix2pix_accuracy)

        logging.info('results on the unet latent space')
        logging.info(classifier_special_classes_test_log_dict_superresolution)
        unet_accuracy.append(np.mean(unet_list))
        unet_std.append(np.std(unet_list))
        logging.info(unet_accuracy)

    import matplotlib.pyplot as plt
    import pandas as pd

    results_df = pd.DataFrame(columns=['name', 'number_of_examples', 'accuracies', 'stds'])

    # plt.plot(number_of_examples, special_classic_accuracy, label=f'special_classic on latent {what_to_plot}')
    results_df = results_df.append({'name': 'classicnet_1000 on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(special_classic_accuracy), 'stds': str(special_classic_std)},
                                   ignore_index=True)

    results_df = pd.DataFrame(columns=['name', 'number_of_examples', 'accuracies', 'stds'])
    # plt.plot(number_of_examples, classic_1000_accuracy, label=f'classicnet_1000 on latent {what_to_plot}')
    results_df = results_df.append({'name': 'classicnet_1000 on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(classic_1000_accuracy), 'stds': str(classic_1000_std)},
                                   ignore_index=True)

    # plt.plot(number_of_examples, classic_5000_accuracy, label=f'classicnet_5000 on latent {what_to_plot}')
    results_df = results_df.append({'name': 'classicnet_5000 on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(classic_5000_accuracy), 'stds': str(classic_5000_std)},
                                   ignore_index=True)
    # plt.plot(number_of_examples, classic_20000_accuracy, label=f'classicnet_20000 on latent {what_to_plot}')
    results_df = results_df.append({'name': 'classicnet_20000 on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(classic_20000_accuracy), 'stds': str(classic_20000_std)},
                                   ignore_index=True)
    # plt.plot(number_of_examples, contrastive_on_latent_accuracy, label=f'contrastive on latent {what_to_plot}')
    results_df = results_df.append({'name': 'contrastive on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(contrastive_on_latent_accuracy),
                                    'stds': str(contrastive_on_latent_std)}, ignore_index=True)
    # plt.plot(number_of_examples, resnet_accuracy, label=f'resnet on latent {what_to_plot}')
    results_df = results_df.append({'name': 'resnet on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(resnet_accuracy), 'stds': str(resnet_std)}, ignore_index=True)
    # plt.plot(number_of_examples, resnet_transfer_accuracy, label=f'resnet full transfer on latent {what_to_plot}')
    results_df = results_df.append(
        {'name': 'resnet full transfer on latent', 'number_of_examples': str(number_of_examples),
         'accuracies': str(resnet_transfer_accuracy), 'stds': str(resnet_transfer_std)}, ignore_index=True)
    # plt.plot(number_of_examples, amphib_ae_accuracy, label=f'amphib ae {what_to_plot}')
    results_df = results_df.append({'name': 'amphib ae on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(amphib_ae_accuracy), 'stds': str(amphib_ae_std)},
                                   ignore_index=True)
    # plt.plot(number_of_examples, on_plain_accuracy, label=f'on plain {what_to_plot}')
    results_df = results_df.append({'name': 'on_plain on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(on_plain_accuracy), 'stds': str(on_plain_std)}, ignore_index=True)
    # plt.plot(number_of_examples, superresolution_accuracy, label=f'superresolution {what_to_plot}')
    results_df = results_df.append({'name': 'superresolution on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(superresolution_accuracy), 'stds': str(superresolution_std)},
                                   ignore_index=True)

    # plt.plot(number_of_examples, pix2pix_accuracy, label=f'pix2pix {what_to_plot}')
    results_df = results_df.append({'name': 'pix2pix on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(pix2pix_accuracy), 'stds': str(pix2pix_std)},
                                   ignore_index=True)

    # plt.plot(number_of_examples, unet_accuracy, label=f'superresolution {what_to_plot}')
    results_df = results_df.append({'name': 'unet on latent', 'number_of_examples': str(number_of_examples),
                                    'accuracies': str(unet_accuracy), 'stds': str(unet_std)},
                                   ignore_index=True)

    # plt.legend()
    # plt.title(f'{classifier} - few shot: {special_class_name} vs. random')
    # plt.xlabel('number of poositive examples')
    # plt.ylabel(f'{what_to_plot}')
    Path(baes_path).mkdir(parents=True, exist_ok=True)

    # plt.savefig(f'{baes_path}/{special_class_name}_{what_to_plot}_{classifier}.png')
    # plt.clf()

    results_df.to_excel(os.path.join(baes_path,
                                     f'{what_to_plot}_{special_class_name}_type-{classifier}_seeds-{len(seeds)}{rescaling}lal.xlsx'))
