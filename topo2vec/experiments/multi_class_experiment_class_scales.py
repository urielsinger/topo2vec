from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.constants import MULTICLASS_LOGS_PATH
from topo2vec.task_handler import TaskHandler
from topo2vec.modules import Classifier

classifier_parser = Classifier.get_args_parser()

######################################################################################################
# an ordinary classifier experiment - change the "classifier_regular_args" to make other experiments #
######################################################################################################


classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                        '--learning_rate', '1e-4',
                                                        '--max_epochs', '100',
                                                        '--total_dataset_size', '1000',
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
                                                        '--original_radiis', '[[8]]',
                                                        '--radii', '[8]',
                                                        '--different_scales',
                                                        '--logs_path', MULTICLASS_LOGS_PATH,
                                                        '--use_gpu'
                                                        ])
lab = TaskHandler(classifier_regular_args)
lab.run()