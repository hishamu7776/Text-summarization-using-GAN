import os
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

import evaluators as ev
from discriminator import Discriminator
from generator import Encoder, Decoder, Generator


class Trainer:
    def __init__(self, config, dataset, trainer='GENERATOR'):
        self.trainer = trainer
        self.config = config
        self.vocab_size= dataset.vocab_size
        self.pad_idx = dataset.PADDING_VALUE
        self.train_loader = dataset.train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.load_model()
        self.set_loss()
        self.set_optimzer()
        self.train()
        self.save_result()


    def load_model(self):
        if self.trainer == 'GENERATOR':
            if self.config['GENERATOR']['pretrained'] == 'True':
                path = Path(self.config['GENERATOR']['model_path'])
                self.model = self.load_pretrained(path)
            else:
                encoder_net = Encoder(self.vocab_size, 
                                      int(self.config['ENCODER']['embedding_dim']), 
                                      int(self.config['ENCODER']['hidden_dim']), 
                                      int(self.config['ENCODER']['num_layers'])).to(self.device)
                decoder_net = Decoder(self.vocab_size, 
                                      int(self.config['DECODER']['embedding_dim']), 
                                      int(self.config['DECODER']['hidden_dim']), 
                                      int(self.config['DECODER']['num_layers'])).to(self.device)
                self.model = Generator(encoder_net, decoder_net, self.device, self.vocab_size).to(self.device)            
        elif self.trainer == 'DISCRIMINATOR':
            if self.config['DISCRIMINATOR']['pretrained'] == 'True':
                path = Path(self.config['DISCRIMINATOR']['model_path'])
                self.model = self.load_pretrained(path)
            else:                
                self.model = Discriminator(vocab_size=self.vocab_size, 
                                              embedding_dim=int(self.config['DISCRIMINATOR']['embedding_dim']), 
                                              hidden_dim=int(self.config['DISCRIMINATOR']['hidden_dim']),
                                              num_layers=int(self.config['DISCRIMINATOR']['num_layers']),
                                              output_dim=int(self.config['DISCRIMINATOR']['output_dim']),
                                              dropout = float(self.config['DISCRIMINATOR']['dropout'])
                                              )
                self.model = self.model.to(self.device)
        else:
            self.generator = self.load_pretrained('generator_path')
            self.discriminator = self.load_pretrained('discriminator_path')
    
    def train(self):
        start_time = time.time()
        self.minibatch_loss_list, self.minibatch_accuracy_list = [],[]
        epochs = int(self.config[self.trainer]['epochs'])
        for epoch in range(epochs):
            total = 0
            accuracy = 0
            self.model.train()
            for batch_idx, (input, target) in tqdm(enumerate(self.train_loader)):
                input = input.to(self.device)
                target = target.to(self.device)
                with autocast():
                    loss, output = self.compute_loss(input, target)       
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # zero gradients again
                #Compute Accuracy
                if self.trainer == 'GENERATOR':
                    accuracy = self.compute_accuracy(target, output)
                elif self.trainer == 'DISCRIMINATOR':
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    accuracy += (predicted == target).sum().item()

                # ## LOGGING
                self.minibatch_loss_list.append(loss.item())
                self.minibatch_accuracy_list.append(accuracy)
                if not batch_idx % 50:
                    print(f'Epoch: {epoch+1:03d}/{epochs:03d} '
                        f'| Batch {batch_idx:04d}/{len(self.train_loader):04d} '
                        f'| Loss: {loss:.4f}'
                        f'| Accuracy: {accuracy:.4f} '
                        )
            if epoch%5 == 0:
                torch.save(self.model, os.path.join(self.config['DEFAULT']['target_folder'],f'{self.trainer}_{epoch}.pt'))
        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
        torch.save(self.model.state_dict(), os.path.join(self.config['DEFAULT']['target_folder'],f'final_{self.trainer}_state_dict.pt'))
        torch.save(self.model, os.path.join(self.config['DEFAULT']['target_folder'],f'final_{self.trainer}.pt'))
        
    
    def train_gan(self):
        return
    
    def set_loss(self):
        if self.trainer == 'GENERATOR':
            self.criterion = nn.CrossEntropyLoss(ignore_index = self.pad_idx)
        elif self.trainer == 'DISCRIMINATOR':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def set_optimzer(self):
        if self.trainer == 'GENERATOR':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['GENERATOR']['learning_rate']))
        elif self.trainer == 'DISCRIMINATOR':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['DISCRIMINATOR']['learning_rate']))
        else:
            self.optimizer = None

    def compute_loss(self, input, target):
        if self.trainer == 'GENERATOR':
            output = self.model(input, target)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)
            loss = self.criterion(output, target)
            return loss, output
        elif self.trainer == 'DISCRIMINATOR':
            logits = self.model(input)
            loss = self.criterion(logits, target)
            return loss, logits
        return
    
    def compute_accuracy(self, target, output):
        if self.trainer == 'GENERATOR':
            target = target[1:].view(-1)
            output_indexes = output.argmax(dim=1)
            accuracy = ev.compute_generator_accuracy(target, output_indexes)
            return accuracy
        elif self.trainer == 'DISCRIMINATOR':
            loss = 0
            return loss
    def save_result(self):
        df = pd.DataFrame({'Loss': self.minibatch_loss_list,
            'Accuracy': self.minibatch_accuracy_list})
        df.to_csv(os.path.join(self.config['DEFAULT']['target_folder'],f'{self.trainer}.csv'), index=False)

    @staticmethod
    def load_pretrained(path):
        model = torch.load(path)
        model.eval()
        return model