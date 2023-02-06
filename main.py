import configparser
from prepare import PrepareData
from dataset import Dataset

config = configparser.ConfigParser()
config.read('config.cfg')

dataset = PrepareData(config)
summary_data = Dataset(dataset, config)