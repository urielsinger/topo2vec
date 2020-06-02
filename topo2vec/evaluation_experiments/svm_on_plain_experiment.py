from sklearn import svm
from sklearn.metrics import accuracy_score

from topo2vec.background import TRAIN_HALF, CLASS_PATHS, CLASS_NAMES, VALIDATION_HALF, CLASS_PATHS_TEST
from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset

######################################################################
# an svm baseline - check if all this deep learning essential at all #
######################################################################

def svm_for_classifying():
    train_size = 8000
    test_size = 2000
    class_paths = CLASS_PATHS
    class_names = CLASS_NAMES
    radii = [[8, 16, 24]]
    svm_train_dataset = SeveralClassesDataset(radii, TRAIN_HALF, train_size,
                                              class_paths, class_names, 'svm_train')
    svm_test_dataset = SeveralClassesDataset(radii, VALIDATION_HALF, test_size,
                                         class_paths, class_names, 'svm_test')
    test_svm_on_plain(svm_train_dataset, svm_test_dataset, radii)

def test_svm_on_plain(svm_train_dataset, svm_test_database, radii = [[8, 16, 24]]):
    radius = min(radii[0])
    h_w = 2 * radius + 1
    im_size = h_w * h_w * len(radii[0])
    X_train, y_train = get_dataset_as_tensor(svm_train_dataset)
    latent_train = X_train.view(-1, im_size)
    SVMClassifier = svm.SVC()
    # print('starting svm fitting')
    SVMClassifier.fit(latent_train.numpy(), y_train.numpy())
    X_test, y_test = get_dataset_as_tensor(svm_test_database)
    latent_test = X_test.view(-1, im_size)
    predicted_train = SVMClassifier.predict(latent_train.numpy())
    predicted_test = SVMClassifier.predict(latent_test.numpy())

    train_accuracy = accuracy_score(y_train.numpy(), predicted_train)
    test_accuracy = accuracy_score(y_test.numpy(), predicted_test)
    return {'svm_train_special_accuracy':train_accuracy, 'svm_test_special_accuracy':test_accuracy}

# svm_for_classifying()