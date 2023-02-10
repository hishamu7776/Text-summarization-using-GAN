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
                
                #Change device
                summary = summary.to(self.device)
                text = text.to(self.device)

                #Train the discriminator
                self.dis_optimizer.zero_grad()
                real_output = self.discriminator(summary)
                ones = torch.ones(real_output.shape).to(self.device)
                real_loss = self.criterion(real_output, ones)

                
                generated_summary = self.generator(text, summary)
                generated_summary = torch.argmax(generated_summary, dim=2)
                discriminator_output = self.discriminator(generated_summary)
                zeros = torch.zeros(discriminator_output.shape).to(self.device)
                generated_loss = self.criterion(discriminator_output, zeros)

                discriminator_loss = 0.5 * (real_loss + generated_loss)
                discriminator_loss.backward()
                self.dis_optimizer.step()

                # Train the generator
                # Zero the gradients
                self.gen_optimizer.zero_grad()
                
                # Calculate the loss
                generator_loss = self.compute_generator_loss(discriminator_output)

                # Calculate the gradients
                generator_loss.backward()
                
                # Update the parameters            
                self.gen_optimizer.step()

                #Compute Accuracy
                output = generated_summary.view(-1)
                target = summaries.view(-1)
                accuracy = self.compute_accuracy(target, output)

                # ## LOGGING
                self.minibatch_discriminator_loss.append(discriminator_loss.item())
                self.minibatch_generator_loss.append(generator_loss.item())
                self.minibatch_accuracy.append(accuracy)
                if not batch_idx % 50:
                    print(f'Epoch: {epoch+1:03d}/{self.epochs:03d} '
                        f'| Batch {batch_idx:04d}/{len(self.train_loader):04d} '
                        f'| Discriminator Loss: {discriminator_loss:.4f}'
                        f'| Generator Loss: {generator_loss:.4f}'
                        f'| Accuracy: {accuracy:.4f} '
                        )

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
    def compute_generator_loss(discriminator_output):
        #Convert discriminator output to [0,1] to avoid negative values.
        discriminator_output = torch.sigmoid(discriminator_output)
        discriminator_output_requires_grad = discriminator_output.detach().requires_grad_()

        #Compute reward, discriminator as reward
        reward = discriminator_output_requires_grad.squeeze()
        
        # Compute the log probability of the discriminator output being positive
        log_p_positive = torch.log(discriminator_output + 1e-8)
        log_p_positive = log_p_positive.detach().requires_grad_()
            
        # Compute the log probability of the discriminator output being negative
        log_p_negative = torch.log(1 - discriminator_output + 1e-8)
        log_p_negative = log_p_negative.detach().requires_grad_()
        
        R_G_D = log_p_positive * reward
        R_G_D = R_G_D.detach().requires_grad_()
        

        JML = log_p_negative * (1 - reward)
        JML = JML.detach().requires_grad_()
        
        # Compute the loss
        loss = -torch.mean(R_G_D + JML)
        loss = loss.detach().requires_grad_()
        
        return loss
    
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