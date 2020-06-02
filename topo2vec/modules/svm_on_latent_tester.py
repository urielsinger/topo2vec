import time

import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.background import TRAIN_HALF, VALIDATION_HALF
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset


def svm_accuracy_on_dataset_in_latent_space(SVMClassifier, dataset: Dataset, model, predict_probas = False) -> Tensor:
    '''

    Args:
        SVMClassifier: the svm classifier with which the classifying should be done
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
    predicted = SVMClassifier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    if predict_probas:
        probas = SVMClassifier.predict_proba(latent.numpy())
        y_np = y.numpy().squeeze()
        auc = sklearn.metrics.roc_auc_score(y_np, probas[:,1], multi_class='ovo')
        # sklearn.metrics.plot_roc_curve(SVMClassifier, latent.numpy(), y.numpy().squeeze())
        import matplotlib.pyplot as plt
        plt.savefig(f'{time.time()}.png')
        return accuracy, auc
    return accuracy



def svm_classifier_test_build_datasets(model, class_paths_to_test: str, class_names_to_test: str, type_of_svm_evaluation_name: str,
                                       test_dataset: Dataset, train_dataset_size: int):
    '''

    Args:
        class_paths_to_test: the path for the classes' to test on
        class_names_to_test: the names for the classes' to test on
        type_of_svm_evaluation_name: what kind of svm classifier test is it - for saving and logging
        test_dataset: the dataset to test on
        train_dataset_size: the size we want to train the svm on.

    Returns: nothing, just logs

    '''
    if model.hparams.svm_classify_latent_space:
        svm_train_dataset = SeveralClassesDataset(model.original_radiis, TRAIN_HALF, train_dataset_size,
                                                  class_paths_to_test, class_names_to_test,
                                                  'train_svm_' + type_of_svm_evaluation_name, model.radii)
        svm_validation_dataset = SeveralClassesDataset(model.original_radiis, VALIDATION_HALF, train_dataset_size,
                                                       class_paths_to_test, class_names_to_test,
                                                       'train_svm_' + type_of_svm_evaluation_name, model.radii)
        return svm_classifier_test(model, svm_train_dataset, svm_validation_dataset, test_dataset, type_of_svm_evaluation_name)
    return {}

def svm_classifier_test(model, svm_train_dataset, svm_validation_dataset, test_dataset, type_of_svm_evaluation_name):
    X_train, y_train = get_dataset_as_tensor(svm_train_dataset)
    if model.hparams.use_gpu:
        X_train = X_train.cuda()

    _, latent_train = model.forward(X_train)
    SVMClassifier = svm.SVC(probability=True)

    if model.hparams.use_gpu:
        latent_train = latent_train.cpu()
        y_train = y_train.cpu()

    latent_train_numpy = latent_train.numpy()
    y_train_numpy = y_train.numpy()

    SVMClassifier.fit(latent_train_numpy, y_train_numpy)

    train_accuracy = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                             svm_train_dataset, model)

    validation_accuracy = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                  svm_validation_dataset, model)

    test_accuracy, test_auc = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                            test_dataset, model, predict_probas=True)
    model.svm_validation_accuracy = validation_accuracy
    model.svm_test_accuracy = test_accuracy

    return {f'svm_train_{type_of_svm_evaluation_name}_accuracy': train_accuracy,
            f'svm_validation_{type_of_svm_evaluation_name}_accuracy': validation_accuracy,
            f'svm_test_{type_of_svm_evaluation_name}_accuracy': test_accuracy,
            f'svm_test_{type_of_svm_evaluation_name}_auc': test_auc}

