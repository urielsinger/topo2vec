from topo2vec.constants import MULTICLASS_LOGS_PATH, ON_TOP_LOGS_PATH
from topo2vec.evaluation_experiments.final_models import superresolution_model
from topo2vec.modules import Classifier
from topo2vec.modules.autoencoder_fine_tuner import AutoencoderFineTuner
from topo2vec.task_handler import TaskHandler

classifier_parser = Classifier.get_args_parser()

classifier_regular_args = classifier_parser.parse_args(['--save_model',
                                                        '--learning_rate', '1e-4',
                                                        '--max_epochs', '100',
                                                        '--total_dataset_size', '1000',
                                                        '--arch', 'LinearLayerOnTop',
                                                        '--svm_classify_latent_space',
                                                        '--name', 'classifier',
                                                        '--latent_space_size', '128',
                                                        '--pytorch_module', 'AutoencoderFineTuner',
                                                        '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
                                                        '--radii', '[34, 136]',
                                                        '--logs_path', ON_TOP_LOGS_PATH,
                                                        ])

fine_tuner = AutoencoderFineTuner(classifier_regular_args, superresolution_model)#, retrain=True)
lab = TaskHandler(classifier_regular_args, fine_tuner)
lab.run()
