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
        self.minibatch_discriminator_loss = []
        self.minibatch_generator_loss = []
        self.minibatch_accuracy = []
        self.epochs = int(self.config[self.trainer]['epochs'])
        if self.trainer == 'GAN':
            self.train_gan()
            return
        start_time = time.time()
        for epoch in range(self.epochs):
            total = 0
            correct = 0
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
                    correct += (predicted == target).sum().item()
                    accuracy = 100*(correct/total)
                else:
                    break

                # ## LOGGING
                if self.trainer == 'GENERATOR':
                    self.minibatch_generator_loss.append(loss.item())
                    self.minibatch_discriminator_loss.append(0)
                elif self.trainer == 'DISCRIMINATOR':
                    self.minibatch_generator_loss.append(0)
                    self.minibatch_discriminator_loss.append(loss.item())
                self.minibatch_accuracy_list.append(accuracy)
                if not batch_idx % 50:
                    print(f'Epoch: {epoch+1:03d}/{self.epochs:03d} '
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
        start_time = time.time()
        for epoch in range(self.epochs):
            for batch_idx, (text, summaries) in tqdm(enumerate(self.train_loader)):
                self.generator.train()
                self.discriminator.train()
                real_summaries = summaries.to(self.device)
                real_text = text.to(self.device)

                # Train the discriminator
                self.discriminator.zero_grad()
                real_outputs = self.discriminator(real_text)
                real_labels = torch.ones(real_outputs.shape).to(self.device)
                self.discriminator.train()
                real_loss = self.criterion(real_outputs, real_labels)

                fake_summaries = self.generator(real_text, real_summaries).detach() 
                fake_summaries = torch.argmax(fake_summaries, dim=2)
                fake_outputs = self.discriminator(fake_summaries)
                fake_labels = torch.zeros(fake_outputs.shape).to(self.device)
                fake_loss = self.criterion(fake_outputs, fake_labels)
                
                dis_loss = 0.5 * (real_loss + fake_loss)
                dis_loss.backward()
                self.dis_optimizer.step()

                # Train the generator
                self.generator.zero_grad()
                fake_outputs = self.discriminator(fake_summaries)
                fake_labels = torch.ones(fake_outputs.shape).to(self.device)
                gen_loss = self.criterion(fake_outputs, fake_labels)
                gen_loss.backward()
                self.gen_optimizer.step()
                #Compute Accuracy
                output = fake_summaries.view(-1)
                target = summaries.view(-1)
                accuracy = self.compute_accuracy(target, output)

                # ## LOGGING
                self.minibatch_discriminator_loss.append(dis_loss.item())
                self.minibatch_generator_loss.append(gen_loss.item())
                self.minibatch_accuracy.append(accuracy)
                if not batch_idx % 50:
                    print(f'Epoch: {epoch+1:03d}/{self.epochs:03d} '
                        f'| Batch {batch_idx:04d}/{len(self.train_loader):04d} '
                        f'| Discriminator Loss: {dis_loss:.4f}'
                        f'| Generator Loss: {gen_loss:.4f}'
                        f'| Accuracy: {accuracy:.4f} '
                        )
            self.generator.eval()
            self.discriminator.eval()
            if epoch%2 == 0:
                torch.save(self.generator, os.path.join(os.path.join(self.config['DEFAULT']['target_folder'],f'{self.trainer}_GEN_EPOCH{epoch}.pt')))
                torch.save(self.discriminator, os.path.join(os.path.join(self.config['DEFAULT']['target_folder'],f'{self.trainer}_DIS_EPOCH{epoch}.pt')))
            elapsed = (time.time() - start_time)/60
            print(f'Time elapsed: {elapsed:.2f} min')            

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
            lr = float(self.config[self.trainer]['learning_rate'])
            self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
            self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

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
        else:
            return ev.compute_generator_accuracy(target, output_indexes)
    def save_result(self):
        df = pd.DataFrame({'Discriminator Loss': self.minibatch_discriminator_loss,
                           'Generator Loss': self.minibatch_generator_loss,
                           'Accuracy': self.minibatch_accuracy_list})
        df.to_csv(os.path.join(self.config['DEFAULT']['target_folder'],f'{self.trainer}.csv'), index=False)

    @staticmethod
    def load_pretrained(path):
        model = torch.load(path)
        model.eval()
        return model