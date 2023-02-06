import torch
from pathlib import Path
from generator import Encoder, Decoder, Generator
from discriminator import Discriminator

class Trainer:
    def __init__(self,config, dataset,trainer='Generator'):
        self.trainer = trainer
        self.config = config
        self.vocab_size= dataset.vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

    def load_model(self):
        if self.trainer == 'Generator':
            if bool(self.config['GENERATOR']['pretrain']):
                path = Path(self.config['GENERATOR']['model_path'])
                self.generator = torch.load(path)
                self.generator.eval()
            else:
                encoder_net = Encoder(self.vocab_size, 
                                      int(self.config['ENCODER']['embedding_dim']), 
                                      int(self.config['ENCODER']['hidden_dim']), 
                                      int(self.config['ENCODER']['num_layers'])).to(self.device)
                decoder_net = Decoder(self.vocab_size, 
                                      int(self.config['DECODER']['embedding_dim']), 
                                      int(self.config['DECODER']['hidden_dim']), 
                                      int(self.config['DECODER']['num_layers'])).to(self.device)
                self.generator = Generator(encoder_net, decoder_net, self.device, self.vocab_size).to(self.device)
            
        elif self.trainer == 'Discriminator':
            if bool(self.config['DISCRIMINATOR']['pretrain']):
                path = Path(self.config['DISCRIMINATOR']['model_path'])
                self.discriminator = torch.load(path)
                self.discriminator.eval()
            else:                
                self.discriminator = Discriminator(self.vocab_size, 
                                              int(self.config['DISCRIMINATOR']['embedding_dim']), 
                                              int(self.config['DISCRIMINATOR']['hidden_dim']),
                                              int(self.config['DISCRIMINATOR']['num_layers']),
                                              int(self.config['DISCRIMINATOR']['output_dim']),
                                              int(self.config['DISCRIMINATOR']['dropout'])
                                              )
                self.discriminator = self.discriminator.to(self.device)
        else:
            print("GAN trainer is not implemented")