from topo2vec.background import LOAD_CLASSES_LARGE
from topo2vec.experiments.task_handler import TaskHandler
from topo2vec.modules import Classifier

classifier_parser = Classifier.get_args_parser()

classifier_check_args = classifier_parser.parse_args(['--save_model',
                                                      '--learning_rate', '1e-2',
                                                      '--max_epochs', '1',
                                                      '--total_dataset_size', '10',
                                                      '--arch', 'BasicConvNetLatent',
                                                      '--svm_classify_latent_space',
                                                      '--name', 'classifier',
                                                      '--pytorch_module', 'Classifier'])

if not LOAD_CLASSES_LARGE:
    classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                            '--learning_rate', '1e-2',
                                                            '--max_epochs', '1',
                                                            '--total_dataset_size', '10',
                                                            '--arch', 'BasicConvNetLatent',
                                                            '--svm_classify_latent_space',
                                                            '--name', 'classifier',
                                                            '--pytorch_module', 'Classifier',
                                                            '--random_set_size', '100'])
else:
    classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                            '--learning_rate', '1e-4',
                                                            '--max_epochs', '10',
                                                            '--total_dataset_size', '100000',
                                                            '--arch', 'BasicConvNetLatent',
                                                            '--svm_classify_latent_space',
                                                            '--name', 'classifier',
                                                            '--pytorch_module', 'Classifier',
                                                            '--random_set_size', '100'])

lab = TaskHandler(classifier_check_args)
lab.run()
