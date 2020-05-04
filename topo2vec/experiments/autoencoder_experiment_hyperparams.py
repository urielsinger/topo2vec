from topo2vec.constants import AUTOENCODEsvm_classifier_ordinary_classes_test_log_dictR_LOGS_PATH
from topo2vec.experiments.task_handler import TaskHandler
from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.modules import Classifier

parser = Classifier.get_args_parser()
autoencoder_check_args = parser.parse_args(['--save_model',
                                            '--learning_rate', '1e-2',
                                            '--max_epochs', '1',
                                            '--total_dataset_size', '1000',
                                            '--arch', 'BasicAmphibAutoencoder',
                                            '--svm_classify_latent_space',
                                            '--name', 'autoencoder',
                                            '--pytorch_module', 'Autoencoder',
                                            '--random_set_size_for_svm', '10'])

if LOAD_CLASSES_LARGE:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '0.004',
                                                  '--max_epochs', '1000',
                                                  '--total_dataset_size', '10',
                                                  '--arch', 'BasicAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--name', 'autoencoder',
                                                  '--knn_method_for_typical_choosing', 'regular',
                                                  '--pytorch_module', 'Autoencoder',
                                                  '--random_set_size_for_svm', '10000',
                                                  '--latent_space_size', '30',
                                                  '--svm_classify_latent_space',
                                                  '--test_knn',
                                                  '--use_gpu',
                                                  '--logs_path', AUTOENCODER_LOGS_PATH
                                                  ])

lab = TaskHandler(autoencoder_regular_args)
lab.run_hparams_search()
