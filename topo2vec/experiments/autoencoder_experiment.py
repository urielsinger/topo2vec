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
                                            '--pytorch_module', 'Autoencoder'])

if not LOAD_CLASSES_LARGE:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '1e-2',
                                                  '--max_epochs', '1',
                                                  '--total_dataset_size', '1000',
                                                  '--arch', 'BasicAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--name', 'autoencoder',
                                                  '--pytorch_module', 'Autoencoder',
                                                  '--total_dataset_size', '10', '--max_epochs', '10',
                                                  '--random_set_size', '100'])
else:
    autoencoder_regular_args = parser.parse_args(['--save_model',
                                                  '--learning_rate', '1e-2',
                                                  '--max_epochs', '40',
                                                  '--total_dataset_size', '1000',
                                                  '--arch', 'BasicAmphibAutoencoder',
                                                  '--svm_classify_latent_space',
                                                  '--name', 'autoencoder',
                                                  '--pytorch_module', 'Autoencoder'])

lab = TaskHandler(autoencoder_regular_args)
lab.run()
