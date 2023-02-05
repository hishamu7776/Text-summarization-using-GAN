import configparser
from dataset import Dataset


config = configparser.ConfigParser()
config.read('config.cfg')

dataset = Dataset()
