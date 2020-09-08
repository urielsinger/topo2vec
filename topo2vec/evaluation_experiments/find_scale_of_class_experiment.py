import random
import time

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.backends import cudnn

from topo2vec.background import VALIDATION_HALF
from topo2vec.constants import BASE_LOCATION
import pytorch_lightning as pl
import numpy as np
from topo2vec.modules import Classifier, OneVsRandomDataset
import pandas as pd

TEST_SIZE = 1000

RADII = 4

arch = 'BasicConvNetLatentSmallForScale' #'BasicConvNetLatent' #

def std_mean_accuracy_radius_class(train_set_size_for_scales_experiment, random_seeds, MAX_EPOCHS,
                                   original_radii_to_check, EXP_LOGS_PATH, class_name):
    class_json_location = BASE_LOCATION + f'data/overpass_classes_data/{class_name}_(45,5,50,15).json'
    if class_name == 'streams':
        class_json_location = BASE_LOCATION + f'data/overpass_classes_data_all/overpass_classes_data/{class_name}_(45,5,50,15).json'

    classifier_parser = Classifier.get_args_parser()
    parse_args_list = [
        '--save_model',
        '--learning_rate', '0.0003200196036593708',
        '--total_dataset_size', f'{train_set_size_for_scales_experiment}',
        '--arch', arch,
        '--name', 'classifier',
        '--latent_space_size', '35',
        '--scale_exp',
        '--scale_exp_class_name', class_name,
        '--scale_exp_class_path', class_json_location,
        '--num_classes', '2',
        '--radii', f'[{RADII}]'
    ]
    validation_accuracies_means = []
    validation_accuracies_stds = []
    for original_radii in original_radii_to_check:
        resulted_accuracies_this_original_radiis_list = []
        for random_seed in random_seeds:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            cudnn.deterministic = True
            np.random.seed(random_seed)
            logger = TensorBoardLogger(EXP_LOGS_PATH, name=f'{random_seed}_{original_radii}')
            trainer = pl.Trainer(max_epochs=MAX_EPOCHS, logger=logger)
            classifier_current_args = classifier_parser.parse_args(parse_args_list +
                                                                   ['--scale_exp_random_seed', str(random_seed),
                                                                    '--original_radii',
                                                                    f'[[{original_radii}]]'
                                                                    ])
            model = Classifier(classifier_current_args)
            trainer.fit(model)
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            cudnn.deterministic = True
            np.random.seed(random_seed)
            test_dataset = OneVsRandomDataset([[original_radii]], TEST_SIZE,
                                              VALIDATION_HALF,
                                              class_json_location,
                                              # f'scale_exp_{self.scale_exp_class_name}_vs_random_validation',
                                              radii=model.radii, random_seed=random_seed)
            test_accuracy = model.get_accuracy_for_small_dataset(test_dataset)

            resulted_accuracies_this_original_radiis_list.append(test_accuracy)
        validation_accuracies_means.append(np.mean(resulted_accuracies_this_original_radiis_list))
        validation_accuracies_stds.append(np.std(resulted_accuracies_this_original_radiis_list))
    return validation_accuracies_means, validation_accuracies_stds


train_set_size_for_scales_experiment = 1000
random_seeds = list(range(880, 900))
MAX_EPOCHS = 25
original_radii_to_check = list(range(4, 21, 1))
EXP_LOGS_PATH = BASE_LOCATION + 'tb_logs/scale_experiment'
class_names = ['streams', 'cliffs', 'rivers', 'peaks', 'saddles']

import matplotlib.pyplot as plt


def save_and_plot(original_radii_to_check, validation_accuracies_means, validation_accuracies_stds, class_name):
    plt.plot(original_radii_to_check, validation_accuracies_means, 'o')
    plt.xlabel('original size')
    plt.ylabel('accuracy')
    title = f'{class_name} vs random, {MAX_EPOCHS} epochs, {train_set_size_for_scales_experiment} train samples, {len(random_seeds)} seeds, {TEST_SIZE} test samples, radii {RADII}, arch {arch}, original radii {str(original_radii_to_check[0])}-{str(original_radii_to_check[-1])},{original_radii_to_check[1]-original_radii_to_check[0]} '
    plt.title(title)
    plt.errorbar(original_radii_to_check, validation_accuracies_means, yerr=validation_accuracies_stds)
    plt.show()
    dataframe = pd.DataFrame(
        list(zip(original_radii_to_check, validation_accuracies_means, validation_accuracies_stds)),
        columns=['original_radii', 'mean', 'std'])
    dataframe.to_excel('results_evaluation_experiments_scale/' + title + '.xlsx')


import time

start_time = time.time()

for class_name in class_names:
    validation_accuracies_means, validation_accuracies_stds = std_mean_accuracy_radius_class(
        train_set_size_for_scales_experiment, random_seeds, MAX_EPOCHS,
        original_radii_to_check, EXP_LOGS_PATH, class_name)
    save_and_plot(original_radii_to_check, validation_accuracies_means, validation_accuracies_stds, class_name)

print("--- %s seconds ---" % (time.time() - start_time))
