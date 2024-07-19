import yaml
import os
current_directory = os.path.dirname(os.path.realpath(__file__))
class Config():
    def __init__(self, yaml_file=os.path.join(current_directory, 'parameters.yaml')):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)

            self.dataset_folder = config.get('dataset_folder')
            self.testing_cloudy_dir = self.dataset_folder + 'TestCloudy'
            self.testing_night_dir = self.dataset_folder + 'TestNight'
            self.testing_sunny_dir = self.dataset_folder + 'TestSunny'

            self.map_dir = self.dataset_folder + 'TrainingBaseline'

            self.training_noDA_dir = self.dataset_folder + 'TrainingBaseline'
            self.training_DA1_dir = self.dataset_folder + 'TrainingDA/DA1'
            self.training_DA2_dir = self.dataset_folder + 'TrainingDA/DA2'
            self.training_DA3_dir = self.dataset_folder + 'TrainingDA/DA3'
            self.training_DA4_dir = self.dataset_folder + 'TrainingDA/DA4'
            self.training_DA5_dir = self.dataset_folder + 'TrainingDA/DA5'
            self.training_DA6_dir = self.dataset_folder + 'TrainingDA/DA6'

            self.validation_dir = self.dataset_folder + 'Validation'

            self.train_batch_size = config.get('train_batch_size')
            self.validation_batch_size = config.get('validation_batch_size')
            self.num_classes = config.get('num_classes')
            self.epochs = config.get('epochs')

            self.models_to_train = config.get('models_to_train')
            self.DA_training_sequences = config.get('DA_training_sequences')

            self.models_to_test = config.get('models_to_test')
            self.DA_test_sequences = config.get('DA_testing_sequences')



PARAMS = Config()


