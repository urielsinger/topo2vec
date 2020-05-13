from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.task_handler import TaskHandler
from topo2vec.modules import Classifier

classifier_parser = Classifier.get_args_parser()

######################################################################################################
# an ordinary classifier experiment - change the "classifier_regular_args" to make other experiments #
######################################################################################################

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
                                                            '--random_set_size_for_knn', '100',
                                                            '--save_to_final'])
else:
    classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                            '--learning_rate', '0.0003200196036593708',
                                                            '--max_epochs', '95',
                                                            '--total_dataset_size', '2500',
                                                            '--arch', 'BasicConvNetLatent',
                                                            '--svm_classify_latent_space',
                                                            '--name', 'classifier',
                                                            '--pytorch_module', 'Classifier',
                                                            '--random_set_size_for_svm', '4000',
                                                            '--random_set_size_for_svm_special', '1000',
                                                            '--random_set_size_for_knn', '1000',
                                                            '--latent_space_size', '35',
                                                            '--knn_method_for_typical_choosing', 'group_from_file',
                                                            '--test_knn',
                                                            '--test_set_size_for_svm', '100',
                                                            '--special_classes_for_validation',
                                                            '["cliffs", "peaks"]',
                                                            '--save_to_final',
                                                            ])
lab = TaskHandler(classifier_regular_args)
lab.run()

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

# good for classifying using BasicConvNet
# classifier_regular_args = classifier_parser.parse_args(['--save_model',
#                                                         '--learning_rate', '0.0003200196036593708',
#                                                         '--max_epochs', '100',
#                                                         '--total_dataset_size', '2500',
#                                                         '--arch', 'BasicConvNetLatent',
#                                                         '--svm_classify_latent_space',
#                                                         '--name', 'classifier',
#                                                         '--pytorch_module', 'Classifier',
#                                                         '--random_set_size_for_svm', '4000',
#                                                         '--random_set_size_for_svm_special', '1000',
#                                                         '--random_set_size_for_knn', '1000',
#                                                         '--latent_space_size', '35',
#                                                         '--knn_method_for_typical_choosing', 'group_from_file',
#                                                         '--test_knn',
#                                                         '--test_set_size_for_svm', '100',
#                                                         '--special_classes_for_validation',
#                                                         '["cliffs", "peaks"]',
#                                                         '--use_gpu'
#                                                         ])


# classifier_regular_args = classifier_parser.parse_args(['--save_model',
#                                                         '--learning_rate', '0.00005',
#                                                         '--max_epochs', '200',
#                                                         '--total_dataset_size', '10000',
#                                                         '--arch', 'AdvancedConvNetLatent',
#                                                         '--svm_classify_latent_space',
#                                                         '--name', 'classifier',
#                                                         '--pytorch_module', 'Classifier',
#                                                         '--random_set_size_for_svm', '4000',
#                                                         '--random_set_size_for_svm_special', '1000',
#                                                         '--random_set_size_for_knn', '1000',
#                                                         '--latent_space_size', '35',
#                                                         '--knn_method_for_typical_choosing', 'group_from_file',
#                                                         '--test_knn',
#                                                         '--test_set_size_for_svm', '100',
#                                                         '--special_classes_for_validation',
#                                                         '["cliffs", "peaks"]',
#                                                         ])