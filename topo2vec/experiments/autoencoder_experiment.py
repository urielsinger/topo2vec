from topo2vec.task_handler import TaskHandler
from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.modules import Classifier

######################################################################################################
# an ordinary autoencoder experiment - change the "classifier_regular_args" to make other experiments #
######################################################################################################

parser = Classifier.get_args_parser()
if LOAD_CLASSES_LARGE:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '0.0008',
                                                  '--max_epochs', '100',
                                                  '--total_dataset_size', '1000',
                                                  '--arch', 'AdvancedAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--knn_method_for_typical_choosing', 'regular',
                                                  '--name', 'autoencoder',
                                                  '--pytorch_module', 'Autoencoder',
                                                  '--random_set_size_for_svm', '2000',
                                                  '--latent_space_size', '30',
                                                  '--svm_classify_latent_space',
                                                  '--test_knn',
                                                  '--use_gpu',
                                                  ])

lab = TaskHandler(autoencoder_regular_args)
lab.run()


# autoencoder_regular_args = parser.parse_args(['--save_model',
#                                               '--learning_rate', '0.004',
#                                               '--max_epochs', '10',
#                                               '--total_dataset_size', '5000',
#                                               '--arch', 'BasicAmphibAutoencoder',
#                                               '--svm_classify_latent_space',
#                                               '--name', 'autoencoder',
#                                               '--pytorch_module', 'Autoencoder',
#                                               '--random_set_size_for_svm', '2000',
#                                               '--latent_space_size', '41',
#                                               '--svm_classify_latent_space',
#                                               '--test_knn'])


# if not LOAD_CLASSES_LARGE:
#     autoencoder_regular_args = parser.parse_args(['--save_model',
#                                                   '--learning_rate', '1e-2',
#                                                   '--max_epochs', '100',
#                                                   '--total_dataset_size', '5000',
#                                                   '--arch', 'BasicAmphibAutoencoder',
#                                                   '--svm_classify_latent_space',
#                                                   '--name', 'autoencoder',
#                                                   '--pytorch_module', 'Autoencoder',
#                                                   '--total_dataset_size', '10',
#                                                   '--random_set_size_for_knn', '100',
#                                                   '--random_set_size_for_svm', '2000'])