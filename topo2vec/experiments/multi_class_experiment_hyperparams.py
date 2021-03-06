from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.constants import MULTICLASS_LOGS_PATH
from topo2vec.task_handler import TaskHandler
from topo2vec.modules import Classifier

########################################################################################################################
# a hyperparams search ordinary classifier experiment - change the "classifier_regular_args" to make other experiments #
########################################################################################################################

classifier_parser = Classifier.get_args_parser()

classifier_check_args = classifier_parser.parse_args(['--save_model',
                                                      '--learning_rate', '1e-4',
                                                      '--max_epochs', '1',
                                                      '--total_dataset_size', '100',
                                                      '--arch', 'BasicConvNetLatent',
                                                      '--svm_classify_latent_space',
                                                      '--name', 'classifier',
                                                      '--pytorch_module', 'Classifier',
                                                      '--random_set_size_for_svm', '100',
                                                      '--random_set_size_for_knn', '100',
                                                      '--latent_space_size', '50',
                                                      '--knn_method_for_typical_choosing', 'group_from_file',
                                                      '--test_knn', ])

if not LOAD_CLASSES_LARGE:
    classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                            '--learning_rate', '1e-2',
                                                            '--max_epochs', '10',
                                                            '--total_dataset_size', '1000',
                                                            '--arch', 'BasicConvNetLatent',
                                                            '--svm_classify_latent_space',
                                                            '--name', 'classifier',
                                                            '--pytorch_module', 'Classifier',
                                                            '--latent_space_size', '20',
                                                            '--random_set_size_for_knn', '100'])
else:
    classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                            '--learning_rate', '1e-4',
                                                            '--max_epochs', '100',
                                                            '--total_dataset_size', '25000',
                                                            '--arch', 'BasicConvNetLatent',
                                                            '--svm_classify_latent_space',
                                                            '--name', 'classifier',
                                                            '--pytorch_module', 'Classifier',
                                                            '--random_set_size_for_svm', '100',
                                                            '--random_set_size_for_svm_special', '10',
                                                            '--test_set_size_for_svm', '100',
                                                            '--svm_classify_latent_space',
                                                            '--latent_space_size', '35',
                                                            '--special_classes_for_validation',
                                                            '["alpine_huts", "antenas"]',
                                                            '--logs_path', MULTICLASS_LOGS_PATH,
                                                            ])

lab = TaskHandler(classifier_regular_args)
lab.run_hparams_search()


# good for 2 classes streams and rivers
# classifier_regular_args = classifier_parser.parse_args(['--save_model',
#                                                         '--learning_rate', '1e-4',
#                                                         '--max_epochs', '100',
#                                                         '--total_dataset_size', '800',
#                                                         '--arch', 'BasicConvNetLatent',
#                                                         '--svm_classify_latent_space',
#                                                         '--name', 'classifier',
#                                                         '--pytorch_module', 'Classifier',
#                                                         '--random_set_size_for_svm', '4000',
#                                                         '--random_set_size_for_knn', '100',
#                                                         '--latent_space_size', '50',
#                                                         '--knn_method_for_typical_choosing', 'group_from_file',
#                                                         '--test_knn', ])
