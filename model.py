import torch
import torch.nn as nn

# GAN model
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        summary = self.generator(x)
        prediction = self.discriminator(summary)
        return prediction