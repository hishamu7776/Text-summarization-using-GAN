import configparser
from prepare import PrepareData
from dataset import Dataset
from trainer import Trainer

config = configparser.ConfigParser()
config.read('config.cfg')

dataset = PrepareData(config)
summary_data = Dataset(dataset, config)



train_generator = Trainer(config, summary_data)