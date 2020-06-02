import os

import matplotlib
import torch
from torchvision import models

from topo2vec.background import CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL, VALIDATION_HALF, TRAIN_HALF
from topo2vec.constants import BASE_LOCATION
from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset
from topo2vec.evaluation_experiments.svm_on_plain_experiment import test_svm_on_plain
from topo2vec.helper_functions import get_paths_and_names_wanted
from topo2vec.modules import Classifier
from topo2vec.modules.svm_on_latent_tester import svm_classifier_test

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

def load_model_from_file(final_model_name = 'classifier_BasicConvNetLatent_[8, 16, 24]_lr_0.0009704376798307045_size_10000_num_classes_4_latent_size_35.pt'):
    load_path = os.path.join(FINAL_MODEL_DIR, final_model_name)
    final_model_classifier = Classifier(FINAL_HPARAMS)
    final_model_classifier.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    final_model_classifier.eval()
    # final_model_classifier.prepare_data()
    return final_model_classifier

model = load_model_from_file()

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
original_radiis = [[8,16,24]]
test_set_size_for_svm = 500
RESNET_RADII = [224, 224, 224]
special_classes_for_validation = ['alpine_huts', 'antenas', 'airialway_stations', 'waterfalls']
train_set_size_for_svm = 5 * len(special_classes_for_validation)

class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)

# svm_special_test_dataset = SeveralClassesDataset(original_radiis, VALIDATION_HALF,
#                                                  test_set_size_for_svm,
#                                                  class_paths_special, class_names_special,
#                                                  'test_svm_special', original_radiis[0])
#
# svm_special_train_dataset = SeveralClassesDataset(original_radiis, TRAIN_HALF,
#                                                  test_set_size_for_svm,
#                                                  class_paths_special, class_names_special,
#                                                  'train_svm_special', original_radiis[0])


train_set_size_for_svm_max = 15
special_classes_for_validation = ['peaks']
class_paths_special, class_names_special = get_paths_and_names_wanted(
    special_classes_for_validation, CLASS_PATHS_SPECIAL, CLASS_NAMES_SPECIAL)
svm_special_test_dataset = OneVsRandomDataset(original_radiis, test_set_size_for_svm, VALIDATION_HALF, class_paths_special[0])
svm_special_test_dataset_resnet = OneVsRandomDataset(original_radiis, test_set_size_for_svm, TRAIN_HALF, class_paths_special[0], radii=RESNET_RADII)

on_latent_accuracy = []
on_plain_accuracy = []
resnet_accuracy = []
number_of_examples = list(range(2, train_set_size_for_svm_max))
for train_set_size_for_svm in number_of_examples:
    svm_special_train_dataset = OneVsRandomDataset(original_radiis, train_set_size_for_svm, TRAIN_HALF, class_paths_special[0])
    svm_special_train_dataset_resnet = OneVsRandomDataset(original_radiis, train_set_size_for_svm, TRAIN_HALF,
                                                          class_paths_special[0], radii=RESNET_RADII)

    with torch.no_grad():
        svm_classifier_special_classes_test_log_dict = \
            svm_classifier_test(model, svm_special_train_dataset, svm_special_train_dataset, svm_special_test_dataset, 'special')

    with torch.no_grad():
        svm_classifier_special_classes_test_log_dict_resnet = \
            svm_classifier_test(topo_resnet_model, svm_special_train_dataset_resnet, svm_special_train_dataset_resnet,
                                svm_special_test_dataset_resnet, 'special')

    print(f'size of dataset = {svm_special_test_dataset}')
    print('results on the latent space')
    print(svm_classifier_special_classes_test_log_dict)
    on_latent_accuracy.append(svm_classifier_special_classes_test_log_dict['svm_test_special_accuracy'])

    print('results on the resnet latent space')
    print(svm_classifier_special_classes_test_log_dict_resnet)
    resnet_accuracy.append(svm_classifier_special_classes_test_log_dict_resnet['svm_test_special_accuracy'])

    print('results of SVM trained')
    svm_classifier_special_classes_test_log_dict = test_svm_on_plain(svm_special_train_dataset, svm_special_test_dataset)
    on_plain_accuracy.append(svm_classifier_special_classes_test_log_dict['svm_test_special_accuracy'])
    print(svm_classifier_special_classes_test_log_dict)

print(f'number_of_examples, on_latent_accuracy, on_plain_accuracy, resnet_accuracy = {number_of_examples}, {on_latent_accuracy}, {on_plain_accuracy}, {resnet_accuracy}')

# airialway_stations
# number_of_examples, on_latent_accuracy, on_plain_accuracy, resnet_accuracy = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0.5120967741935484, 0.45564516129032256, 0.5725806451612904, 0.6068548387096774, 0.5846774193548387, 0.6229838709677419, 0.5604838709677419, 0.5584677419354839, 0.5745967741935484, 0.6169354838709677, 0.6169354838709677, 0.594758064516129, 0.6048387096774194], [0.5786290322580645, 0.5625, 0.5564516129032258, 0.5080645161290323, 0.5342741935483871, 0.45161290322580644, 0.5, 0.48588709677419356, 0.4576612903225806, 0.5141129032258065, 0.5504032258064516, 0.5161290322580645, 0.5383064516129032], [0.6077235772357723, 0.6199186991869918, 0.43902439024390244, 0.44308943089430897, 0.6951219512195121, 0.6971544715447154, 0.5934959349593496, 0.6382113821138211, 0.6544715447154471, 0.6666666666666666, 0.6504065040650406, 0.6747967479674797, 0.5609756097560976]
# alpine huts
# number_of_examples, on_latent_accuracy, on_plain_accuracy, resnet_accuracy = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0.45564516129032256, 0.39314516129032256, 0.594758064516129, 0.5866935483870968, 0.42338709677419356, 0.5907258064516129, 0.6229838709677419, 0.6048387096774194, 0.5826612903225806, 0.6451612903225806, 0.6189516129032258, 0.6129032258064516, 0.4939516129032258], [0.45564516129032256, 0.4254032258064516, 0.4475806451612903, 0.48588709677419356, 0.43951612903225806, 0.5524193548387096, 0.5564516129032258, 0.5625, 0.5584677419354839, 0.5766129032258065, 0.592741935483871, 0.5685483870967742, 0.4838709677419355], [0.6483739837398373, 0.6219512195121951, 0.5711382113821138, 0.4898373983739837, 0.6646341463414634, 0.6646341463414634, 0.6585365853658537, 0.6890243902439024, 0.6849593495934959, 0.6504065040650406, 0.6808943089430894, 0.6971544715447154, 0.7012195121951219]

# waterfalls
# number_of_examples, on_latent_accuracy, on_plain_accuracy, resnet_accuracy = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0.6229838709677419, 0.4495967741935484, 0.5725806451612904, 0.6290322580645161, 0.657258064516129, 0.6713709677419355, 0.5846774193548387, 0.5584677419354839, 0.4737903225806452, 0.5100806451612904, 0.5967741935483871, 0.6612903225806451, 0.6754032258064516], [0.5423387096774194, 0.5040322580645161, 0.48185483870967744, 0.5161290322580645, 0.5383064516129032, 0.5362903225806451, 0.5383064516129032, 0.5342741935483871, 0.5141129032258065, 0.5040322580645161, 0.5504032258064516, 0.5685483870967742, 0.5483870967741935], [0.6016260162601627, 0.6239837398373984, 0.40853658536585363, 0.3882113821138211, 0.5975609756097561, 0.676829268292683, 0.5182926829268293, 0.5304878048780488, 0.6178861788617886, 0.5955284552845529, 0.6524390243902439, 0.6747967479674797, 0.6361788617886179]
# peaks
# number_of_examples, on_latent_accuracy, on_plain_accuracy, resnet_accuracy = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0.8629032258064516, 0.8810483870967742, 0.8225806451612904, 0.8911290322580645, 0.8810483870967742, 0.8951612903225806, 0.8830645161290323, 0.8891129032258065, 0.8891129032258065, 0.8810483870967742, 0.8850806451612904, 0.8810483870967742, 0.8830645161290323], [0.6673387096774194, 0.5786290322580645, 0.7016129032258065, 0.7560483870967742, 0.8326612903225806, 0.842741935483871, 0.8165322580645161, 0.8104838709677419, 0.8387096774193549, 0.8064516129032258, 0.844758064516129, 0.8387096774193549, 0.8346774193548387], [0.6788617886178862, 0.6626016260162602, 0.7357723577235772, 0.7215447154471545, 0.7560975609756098, 0.7032520325203252, 0.8211382113821138, 0.7764227642276422, 0.7845528455284553, 0.6971544715447154, 0.8008130081300813, 0.7439024390243902, 0.7601626016260162]
import matplotlib.pyplot as plt
plt.plot(number_of_examples, on_latent_accuracy, label='on latent accuracy')
plt.plot(number_of_examples, resnet_accuracy, label='resnet on latent accuracy')
plt.plot(number_of_examples, on_plain_accuracy, label='on plain accuracy')
plt.legend()
plt.title('latent vs. plain svm classification of peaks vs. random')
plt.xlabel('n')
plt.ylabel('accuracy')
plt.show()