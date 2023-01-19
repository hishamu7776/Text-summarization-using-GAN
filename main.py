
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


from discriminator import Discriminator
from generator import Generator
from model import GAN
import helper
print(torch.__version__)

'''
articles_path = '..\\BBC News Summary\\News Articles'
summaries_path = '..\\BBC News Summary\\Summaries'
#Read data
text_data = helper.read_articles(articles_path, summaries_path)
text_data = helper.clean_dataframe(text_data)
train_df, test_df = train_test_split(text_data, test_size=0.1)
print(train_df)

# Initialize model
input_size = 100
hidden_size = 50
output_size = 1
num_epochs = 3


generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(input_size, hidden_size, output_size)
model = GAN(generator, discriminator)

# Define loss function and optimizers
criterion = nn.BCEWithLogitsLoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)


# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        d_loss = criterion(outputs, labels)

        # Backward and optimize
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Generate fake summary
        fake_summary = generator(inputs)

        # Discriminator loss on fake summary
        fake_prediction = discriminator(fake_summary)
        g_loss = criterion(fake_prediction, labels)

        # Backward and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

'''