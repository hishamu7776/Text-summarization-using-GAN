import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)
        #self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        
    def forward(self, x):
        
        #x shape: (seq_length, N)
        #embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        
        #embedding shape: (seq_length, N, embedding_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)
        
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        #cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)

        return encoder_states, hidden, cell