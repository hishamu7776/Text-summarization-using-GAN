
import torch
import torch.nn as nn
import numpy as np
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

class Dataset:
    def __init__(self, dataset, config, generator=True):
        self.size = len(dataset.summary_data)
        self.dataset = dataset
        self.config = config
        self.generator = generator
        self.PADDING_TOKEN = self.config['DEFAULT']['PADDING_TOKEN']
        self.UNKNOWN_TOKEN = self.config['DEFAULT']['UNKNOWN_TOKEN']
        self.BOS_TOKEN = self.config['DEFAULT']['BOS_TOKEN']
        self.EOS_TOKEN = self.config['DEFAULT']['EOS_TOKEN']
        self.special_tokens = ['<PAD>', '<UNK>','<BOS>', '<EOS>']
        if generator:
            self.split_dataset(dataset.summary_data)
            self.vocab = self.get_vocab()
            self.vocab_size = len(self.vocab)
        else:
            self.split_dataset(self.generate_binary(dataset))
            self.vocab = dataset.vocab
            self.vocab_size = dataset.vocab_size
        self.text_transform = lambda x: [self.vocab[token] for token in tokenizer(x)]
        vocab_itos = self.vocab.get_itos()
        self.vec_vocab_itos = np.vectorize(lambda x: vocab_itos[x])

                            
    def split_dataset(self, dataset):
        _,test,val = [int(val)/100 for val in self.config['DEFAULT']['split_ratio'].split(":")]
        self.train_list, val_and_test_list = train_test_split(dataset, test_size=test+val)
        self.valid_list, self.test_list = train_test_split(val_and_test_list, test_size=test/(test+val))
        
    def get_vocab(self):
        min_freq = int(self.config['DEFAULT']['min_freq'])
        max_token = int(self.config['DEFAULT']['vocabulary_size'])
        vocab = build_vocab_from_iterator(self.yield_tokens(self.train_list), specials=self.special_tokens, max_tokens=max_token, min_freq = min_freq)
        vocab.set_default_index(vocab[self.UNKNOWN_TOKEN])
        return vocab

    @staticmethod
    def generate_binary(self):
        self.binary_set = []
        return
    
    @staticmethod
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)

    #text_max_len = 700
    #summary_max_len = 300
    def collate_batch(self, batch):
        text_list, label_list = [], []
        if self.generator:
            max_len = 1000
        else:
            max_len = 1000
        #print(type(batch)
        for (_text, _label) in batch:
            processed_text = torch.tensor(self.text_transform(_text)[:max_len])
            text_list.append(processed_text)
            if self.generator:
                processed_label = torch.tensor(self.text_transform(_label))
                label_list.append(processed_label)
            else:
                label_list.append(int(_label)) 

            #target_max_len
                
        text_list[0] = nn.ConstantPad1d((0, max_len - text_list[0].shape[0]), 0)(text_list[0])
        
        padded_text = pad_sequence(text_list, padding_value=self.PADDING_VALUE, batch_first=False)
        if self.generator:
            padded_label = pad_sequence(label_list, padding_value=self.PADDING_VALUE, batch_first=False)
            return padded_text, padded_label
        else:
            return padded_text, torch.tensor(label_list)
        
