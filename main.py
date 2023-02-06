import configparser
from prepare import PrepareData
from dataset import Dataset

config = configparser.ConfigParser()
config.read('config.cfg')

dataset = PrepareData(config)
summary_data = Dataset(dataset, config)
text_batch, label_batch = next(iter(summary_data.test_loader))