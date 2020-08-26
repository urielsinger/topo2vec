import os
from typing import List, Tuple


def full_path_name_of_dataset_data_to_full_path(full_path: str, name: str) -> str:
    '''
    Args:
        full_path:
        name:

    Returns: The full path of things from dataset( the points, the patchesa of the points, etc.)


    '''
    full_path = os.path.join(full_path, name + '.npy')
    return full_path


def cache_path_name_to_full_path(cache_dir: str, file_path: str, name: str) -> str:
    '''
    build a location to save a cache of a class file as npy file.
    saving as npy ius quicker than in pickle manner.
    Args:
        cache_dir:
        file_path:
        name:

    Returns:

    '''
    file_name, _ = os.path.splitext(file_path)
    file_name_end = name + '_' + file_name.split('\\')[-1]
    full_path = os.path.join(cache_dir, file_name_end + '.npy')
    return full_path


def get_paths_and_names_wanted(list_wanted: List[str], class_paths_list: List[str],
                               class_names_list: List[str]) -> Tuple[List[str], List[str]]:
    '''

    Args:
        list_wanted: a list of the names of the wanted classes
        class_paths_list: a list of all the class paths
        class_names_list: a list of all the class names in the same order

    Returns: the paths and names of the chosen classes (according to the "list wanted")

    '''
    index_names_tuples = list(enumerate(class_names_list))
    class_index_names_chosen = [index_name for index_name in
                                 index_names_tuples if index_name[1]
                                 in list_wanted]

    class_names_chosen = [index_name[1] for index_name in
                           class_index_names_chosen]
    class_paths_chosen = [class_paths_list[index_name[0]] for
                           index_name in class_index_names_chosen]

    return class_paths_chosen, class_names_chosen


