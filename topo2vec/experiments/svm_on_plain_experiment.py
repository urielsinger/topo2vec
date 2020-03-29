from sklearn import svm
from sklearn.metrics import accuracy_score

from topo2vec.background import TRAIN_HALF, class_paths, class_names, VALIDATION_HALF
from topo2vec.common import visualizations
from topo2vec.constants import class_paths_test
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset

radii = [8, 16, 24]
radius = min(radii)
h_w = 2 * radius + 1
im_size = h_w * h_w * len(radii)

svm_train_dataset = SeveralClassesDataset(radii, TRAIN_HALF, 1000,
                                          class_paths, class_names)

X_train, y_train = visualizations.get_dataset_as_tensor(svm_train_dataset)
latent_train = X_train.view(-1, im_size)
SVMClassifier = svm.SVC()
SVMClassifier.fit(latent_train.numpy(), y_train.numpy())

size_test = 55
test_dataset = SeveralClassesDataset(radii, VALIDATION_HALF, size_test,
                                     class_paths_test, class_names)

X_test, y_test = visualizations.get_dataset_as_tensor(test_dataset)
latent_test = X_test.view(-1, im_size)

predicted_train = SVMClassifier.predict(latent_train.numpy())
predicted_test = SVMClassifier.predict(latent_test.numpy())

train_accuracy = accuracy_score(y_train.numpy(), predicted_train)
test_accuracy = accuracy_score(y_test.numpy(), predicted_test)

print(f'train accuracy:{train_accuracy}')
print(f'test accuracy:{test_accuracy}')
