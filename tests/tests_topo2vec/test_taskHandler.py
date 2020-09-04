from unittest import TestCase

from topo2vec.task_handler import TaskHandler
from topo2vec.modules import Classifier

class TestTaskHandler(TestCase):
    def test_run(self):
        parser = Classifier.get_args_parser()
        autoencoder_check_args = parser.parse_args(['--save_model',
                                                    '--learning_rate', '1e-2',
                                                    '--max_epochs', '1',
                                                    '--total_dataset_size', '1000',
                                                    '--arch', 'BasicAmphibAutoencoder',
                                                    '--svm_classify_latent_space',
                                                    '--name', 'autoencoder',
                                                    '--pytorch_module', 'Autoencoder',
                                                    '--test_knn',
                                                    '--random_set_size_for_svm', '10'])

        lab = TaskHandler(autoencoder_check_args)
        lab.run()

        parser = Classifier.get_args_parser()
        autoencoder_check_args = parser.parse_args(['--save_model',
                                                    '--learning_rate', '1e-2',
                                                    '--max_epochs', '1',
                                                    '--total_dataset_size', '1000',
                                                    '--arch', 'BasicAmphibAutoencoder',
                                                    '--svm_classify_latent_space',
                                                    '--name', 'autoencoder',
                                                    '--pytorch_module', 'Autoencoder',
                                                    '--random_set_size_for_svm', '10',
                                                    '--svm_classify_latent_space',
                                                    '--test_knn'
                                                    ])
        lab = TaskHandler(autoencoder_check_args)
        lab.run()

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
                                                              '--test_set_size_for_svm', '38',
                                                              '--random_set_size_for_knn', '100',
                                                              '--latent_space_size', '50',
                                                              '--knn_method_for_typical_choosing', 'group_from_file',
                                                              '--random_set_size_for_svm_special', '1000',
                                                              '--test_knn',
                                                              '--svm_classify_latent_space',
                                                              ])

        lab = TaskHandler(classifier_check_args)
        lab.run()

        #add use_gpu if needed.



