import time

import sklearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.background import TRAIN_HALF, VALIDATION_HALF
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset


def svm_accuracy_on_dataset_in_latent_space(SVMClassifier, dataset: Dataset, model, predict_probas=False,
                                            multi=False, use_gpu=False) -> Tensor:
    '''

    Args:
        SVMClassifier: the svm classifier with which the classifying should be done
        dataset: the dataset we want to classify on
        model: the model which produces the latent space

    Returns: the efficient accuracy

    '''
    X, y = get_dataset_as_tensor(dataset)
    if use_gpu or model.hparams.use_gpu:
        X = X.cuda()
    if use_gpu:
        latent = model.forward(X)
    else:
        _, latent = model.forward(X)

    if use_gpu or model.hparams.use_gpu:
        latent = latent.cpu()
    predicted = SVMClassifier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    if predict_probas:
        probas = SVMClassifier.predict_proba(latent.numpy())
        y_np = y.numpy().squeeze()
        if multi:
            auc = sklearn.metrics.roc_auc_score(y_np, probas, multi_class='ovo')
        else:
            auc = sklearn.metrics.roc_auc_score(y_np, probas[:, 1], multi_class='ovo')
        # sklearn.metrics.plot_roc_curve(SVMClassifier, latent.numpy(), y.numpy().squeeze())
        # import matplotlib.pyplot as plt
        # plt.savefig(f'{time.time()}.png')
        return accuracy, auc

    f1_macro = sklearn.metrics.f1_score(y.numpy(), predicted, average='macro')
    f1_micro = sklearn.metrics.f1_score(y.numpy(), predicted, average='micro')
    accuracy = sklearn.metrics.accuracy_score(y.numpy(), predicted)
    return accuracy, f1_micro, f1_macro


def svm_classifier_test_build_datasets(model, class_paths_to_test: str, class_names_to_test: str,
                                       type_of_svm_evaluation_name: str,
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
        return svm_classifier_test(model, svm_train_dataset, svm_validation_dataset, test_dataset,
                                   type_of_svm_evaluation_name)
    return {}


def svm_classifier_test(model, svm_train_dataset, svm_validation_dataset, test_dataset, type_of_svm_evaluation_name,
                        use_gpu=False):
    '''

    Args:
        model:
        svm_train_dataset:
        svm_validation_dataset:
        test_dataset:
        type_of_svm_evaluation_name:
        use_gpu: whether this model is contrastive or not

    Returns:

    '''
    X_train, y_train = get_dataset_as_tensor(svm_train_dataset)
    if use_gpu or model.hparams.use_gpu:
        X_train = X_train.cuda()

    if model is not None: # if None - we mean plain svm on the images
        if use_gpu:
            latent_train = model.forward(X_train)
        else:
            _, latent_train = model.forward(X_train)
    else:
        latent_train = X_train.flatten()
    SVMClassifier = svm.SVC(probability=True)

    if use_gpu or model.hparams.use_gpu:
        latent_train = latent_train.cpu()
        y_train = y_train.cpu()

    latent_train_numpy = latent_train.numpy()
    y_train_numpy = y_train.numpy()

    SVMClassifier.fit(latent_train_numpy, y_train_numpy)

    train_accuracy, f1_micro_train, f1_macro_train = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                                             svm_train_dataset, model,
                                                                                             use_gpu=use_gpu)

    validation_accuracy, f1_micro_validation, f1_macro_validation = svm_accuracy_on_dataset_in_latent_space(
        SVMClassifier,
        svm_validation_dataset, model, use_gpu=use_gpu)

    if len(test_dataset.class_names) == 2:
        test_accuracy, test_auc = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                          test_dataset, model, predict_probas=True,
                                                                          use_gpu=use_gpu)

    else:
        test_accuracy, test_auc = svm_accuracy_on_dataset_in_latent_space(SVMClassifier,
                                                                          test_dataset, model, predict_probas=True,
                                                                          multi=True, use_gpu=use_gpu)
    model.svm_validation_accuracy = validation_accuracy
    model.svm_test_accuracy = test_accuracy

    return {f'svm_train_{type_of_svm_evaluation_name}_accuracy': train_accuracy,
            f'svm_test_{type_of_svm_evaluation_name}_accuracy': test_accuracy,
            f'svm_test_{type_of_svm_evaluation_name}_auc': test_auc,
            f'svm_validation_{type_of_svm_evaluation_name}_accuracy': validation_accuracy,
            f'svm_validation_{type_of_svm_evaluation_name}_f1_micro': f1_micro_validation,
            f'svm_validation_{type_of_svm_evaluation_name}_f1_macro': f1_macro_validation}


def knn_accuracy_on_dataset_in_latent_space(knn_classfier, dataset: Dataset, model, use_gpu) -> Tensor:
    '''

    Args:
        SVMClassifier: the svm classifier with which the classifying should be done
        dataset: the dataset we want to classify on
        model: the model which produces the latent space

    Returns: the efficient accuracy

    '''
    X, y = get_dataset_as_tensor(dataset)
    if use_gpu or model.hparams.use_gpu:
        X = X.cuda()

    if use_gpu:
        latent = model.forward(X)
    else:
        _, latent = model.forward(X)

    if use_gpu or model.hparams.use_gpu:
        latent = latent.cpu()
    predicted = knn_classfier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    return accuracy


def knn_classifier_test(model, knn_train_dataset, knn_validation_dataset, test_dataset, type_of_knn_evaluation_name,
                        use_gpu=False):
    X_train, y_train = get_dataset_as_tensor(knn_train_dataset)
    try:
        if use_gpu or model.hparams.use_gpu:
            X_train = X_train.cuda()
    except:
        pass

    if use_gpu:  # means contrastive
        latent_train = model.forward(X_train)
    else:
        _, latent_train = model.forward(X_train)

    knn_classifier = KNeighborsClassifier()

    if use_gpu or model.hparams.use_gpu:
        latent_train = latent_train.cpu()
        y_train = y_train.cpu()

    latent_train_numpy = latent_train.numpy()
    y_train_numpy = y_train.numpy()

    knn_classifier.fit(latent_train_numpy, y_train_numpy)

    train_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                             knn_train_dataset, model, use_gpu)

    validation_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                                  knn_validation_dataset, model, use_gpu)

    if len(test_dataset.class_names) == 2:
        test_accuracy = knn_accuracy_on_dataset_in_latent_space(knn_classifier,
                                                                test_dataset, model, use_gpu)
        model.svm_validation_accuracy = validation_accuracy
        model.svm_test_accuracy = test_accuracy

        return {f'knn_train_{type_of_knn_evaluation_name}_accuracy': train_accuracy,
                f'knn_validation_{type_of_knn_evaluation_name}_accuracy': validation_accuracy,
                f'knn_test_{type_of_knn_evaluation_name}_accuracy': test_accuracy}
