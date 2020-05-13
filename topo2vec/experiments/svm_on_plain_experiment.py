from sklearn import svm
from sklearn.metrics import accuracy_score

from topo2vec.background import TRAIN_HALF, CLASS_PATHS, CLASS_NAMES, VALIDATION_HALF, CLASS_PATHS_TEST
from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset

######################################################################
# an svm baseline - check if all this deep learning essential at all #
######################################################################

radii = [8, 16, 24]
radius = min(radii)
h_w = 2 * radius + 1
im_size = h_w * h_w * len(radii)

svm_train_dataset = SeveralClassesDataset(radii, TRAIN_HALF, 3000,
                                          CLASS_PATHS, CLASS_NAMES, 'svm_train')

X_train, y_train = get_dataset_as_tensor(svm_train_dataset)
latent_train = X_train.view(-1, im_size)
SVMClassifier = svm.SVC()
print('starting svm fitting')
SVMClassifier.fit(latent_train.numpy(), y_train.numpy())

size_test = 55
test_dataset = SeveralClassesDataset(radii, VALIDATION_HALF, size_test,
                                     CLASS_PATHS_TEST, CLASS_NAMES, 'svm_test')

X_test, y_test = get_dataset_as_tensor(test_dataset)
latent_test = X_test.view(-1, im_size)

predicted_train = SVMClassifier.predict(latent_train.numpy())
predicted_test = SVMClassifier.predict(latent_test.numpy())

train_accuracy = accuracy_score(y_train.numpy(), predicted_train)
test_accuracy = accuracy_score(y_test.numpy(), predicted_test)

print(f'svm train accuracy:{train_accuracy}')
print(f'svm test accuracy:{test_accuracy}')
