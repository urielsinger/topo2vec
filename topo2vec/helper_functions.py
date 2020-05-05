import os

from sklearn.metrics import accuracy_score

from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor


def full_path_name_to_full_path(full_path: str, name: str):
    full_path = os.path.join(full_path, name + '.npy')
    return full_path


def cache_path_name_to_full_path(cache_dir: str, file_path: str, name: str):
    file_name, _ = os.path.splitext(file_path)
    file_name_end = name + '_' + file_name.split('/')[-1]
    full_path = os.path.join(cache_dir, file_name_end + '.npy')
    return full_path


def get_paths_and_names_wanted(list_wanted, class_paths_list, class_names_list):
    index_names = list(enumerate(class_names_list))
    class_index_names_special = [index_name for index_name in
                                 index_names if index_name[1]
                                 in list_wanted]

    class_names_special = [index_name[1] for index_name in
                           class_index_names_special]
    class_paths_special = [class_paths_list[index_name[0]] for
                           index_name in class_index_names_special]

    return class_paths_special, class_names_special


def svm_accuracy_on_dataset_in_latent_space(SVMClassifier, dataset, model):
    X, y = get_dataset_as_tensor(dataset)
    if model.hparams.use_gpu:
        X = X.cuda()
    _, latent = model.forward(X)
    if model.hparams.use_gpu:
        latent = latent.cpu()
    predicted = SVMClassifier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    return accuracy