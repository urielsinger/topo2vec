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

if not LOAD_CLASSES_LARGE:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '1e-2',
                                                  '--max_epochs', '1',
                                                  '--total_dataset_size', '1000',
                                                  '--arch', 'BasicAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--name', 'autoencoder',
                                                  '--pytorch_module', 'Autoencoder',
                                                  '--total_dataset_size', '10',
                                                  '--random_set_size', '100',
                                                  '--random_set_size_for_svm', '1000'])
else:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '1e-2',
                                                  '--max_epochs', '40',
                                                  '--total_dataset_size', '1000',
                                                  '--arch', 'BasicAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--name', 'autoencoder',
                                                  '--pytorch_module', 'Autoencoder',
                                                  '--random_set_size_for_svm', '1000',
                                                  '--latent_space_size', '1'])

lab = TaskHandler(autoencoder_regular_args)
lab.run()
