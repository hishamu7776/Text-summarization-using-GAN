import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import evaluators as ev

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from generator import Generator, Encoder, Decoder



tokenizer = get_tokenizer("basic_english")

class BatchSamplerSimilarLength(Sampler):
    def __init__(self, dataset, batch_size,tokenizer ,indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indices and length
        self.indices = [(i, len(tokenizer(s[0]))) for i, s in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]
        
        # Comment in for validation
        #self.pooled_lengths = [x[1] for x in pooled_indices]
        #print(self.pooled_lengths)
        #print(self.pooled_indices)

        # yield indices for current batch
        batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


class Dataset:
    def __init__(self, dataset, config, generator=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.dataset = dataset
        self.config = config
        self.generator = generator
        self.PADDING_TOKEN = self.config['DEFAULT']['PADDING_TOKEN']
        self.UNKNOWN_TOKEN = self.config['DEFAULT']['UNKNOWN_TOKEN']
        self.BOS_TOKEN = self.config['DEFAULT']['BOS_TOKEN']
        self.EOS_TOKEN = self.config['DEFAULT']['EOS_TOKEN']
        self.special_tokens = ['<PAD>', '<UNK>','<BOS>', '<EOS>']
        if generator:
            self.size = len(dataset.summary_data)            
            self.split_dataset(dataset.summary_data)
            self.vocab = self.get_vocab()
            self.vocab_size = len(self.vocab)
        else:
            self.dataset = dataset
            self.split_dataset(self.generate_binary())
            self.vocab = dataset.vocab
            self.vocab_size = dataset.vocab_size
        self.PADDING_VALUE=self.vocab[self.PADDING_TOKEN]
        self.text_transform = lambda x: [self.vocab[token] for token in tokenizer(x)]
        vocab_itos = self.vocab.get_itos()
        self.vec_vocab_itos = np.vectorize(lambda x: vocab_itos[x])
        self.prepare_batch()

                            
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
    
    def prepare_batch(self):
        if self.generator:
            batch_size = int(self.config['GENERATOR']['batch_size'])
        else:
            batch_size = int(self.config['DISCRIMINATOR']['batch_size'])
        self.train_loader = DataLoader(self.train_list, 
                                batch_sampler=BatchSamplerSimilarLength(dataset = self.train_list,
                                                                        tokenizer=tokenizer,
                                                                        batch_size=batch_size),
                                collate_fn=self.collate_batch)
        self.valid_loader = DataLoader(self.train_list, 
                                batch_sampler=BatchSamplerSimilarLength(dataset = self.valid_list,
                                                                        tokenizer=tokenizer,
                                                                        batch_size=batch_size,
                                                                        shuffle=False),
                                collate_fn=self.collate_batch)
        self.test_loader = DataLoader(self.train_list, 
                                batch_sampler=BatchSamplerSimilarLength(dataset = self.test_list, 
                                                                        tokenizer=tokenizer,
                                                                        batch_size=batch_size,
                                                                        shuffle=False),
                                collate_fn=self.collate_batch)
    
    def collate_batch(self, batch):
        text_list, label_list = [], []
        if self.generator:
            max_len = int(self.config['GENERATOR']['max_len'])
        else:
            max_len = int(self.config['DISCRIMINATOR']['max_len'])
        #print(type(batch)
        for (_text, _label) in batch:
            
            
            if self.generator:
                processed_text = torch.tensor(self.text_transform(_text))
                text_list.append(processed_text)
                processed_label = torch.tensor(self.text_transform(_label))
                label_list.append(processed_label)
            else:
                processed_text = torch.tensor(self.text_transform(_text)[:max_len])
                text_list.append(processed_text)
                text_list[0] = nn.ConstantPad1d((0, max_len - text_list[0].shape[0]), 0)(text_list[0])
                label_list.append(int(_label))                 
        
        padded_text = pad_sequence(text_list, padding_value=self.PADDING_VALUE, batch_first=False)
        if self.generator:
            padded_label = pad_sequence(label_list, padding_value=self.PADDING_VALUE, batch_first=False)
            return padded_text, padded_label
        else:
            return padded_text, torch.tensor(label_list)

    def generate_binary(self):
        generator_path = Path(self.config['GENERATOR']['model_path'])
        self.gen_model = torch.load(generator_path)
        self.gen_model.eval()
        binary = []
        for text, summary in self.dataset.train_list:
            fake = self.generate_summary(text, summary)
            binary.append((summary, 1))
            binary.append((fake, 0))
        return binary


    def generate_summary(self, text, summary):
        input_tensor = torch.tensor(self.dataset.text_transform(text)).unsqueeze(0)
        input_tensor = input_tensor.permute(1,0).to(self.device).long()
        output_tensor = torch.tensor(self.dataset.text_transform(summary)).unsqueeze(0)
        output_tensor = output_tensor.permute(1,0).to(self.device).long()
        self.gen_model.eval()
        with torch.no_grad():
            output = self.gen_model(input_tensor, output_tensor)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        word_indexes = output.argmax(dim=1).squeeze(0).tolist()
        word_indexes = [word for word in word_indexes if word is not self.dataset.PADDING_VALUE]
        if len(word_indexes)>0:
            summary = " ".join(self.dataset.vec_vocab_itos(word_indexes))
        else:
            return "No summary generated"
        return summary

    
    @staticmethod
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)        