import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self, source, target, teacher_force_ratio=0.5):
        
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(vocab)
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        
        encoder_states, hidden, cell = self.encoder(source)
        #Grab start token
        x = target[0]
        
        for t in range(1,target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            #(N,vocab_size)
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs