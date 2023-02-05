import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,num_layers, p):
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.dropout = nn.Dropout(p)
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)
        
        self.energy = nn.Linear(hidden_size*3,1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, encoder_states, hidden, cell):
        
        #x shape: (N) required (1,N)
        x = x.unsqueeze(0)

        #embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        #embedding shape: (1, N, embedding_size)
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states),dim = 2)))
        attention = self.softmax(energy)#(seq_len, N, 1)
        
        attention = attention.permute(1, 2, 0)#(N, 1, seq_len)
        encoder_states = encoder_states.permute(1, 0, 2)#(N, seq_len, hidden_size*2)
        
        
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)
        #(N, 1, hidden_size*2) --> #(1, N, hidden_size*2)
        
        rnn_input = torch.cat((context_vector, embedding),dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        #output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)
        #output = self.fc(output.squeeze(0))
        #shape (1, N, length_of_vocabulary)
        #output = self.softmax(output)
        return predictions, hidden, cell