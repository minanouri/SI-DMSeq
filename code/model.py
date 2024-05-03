import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
    
    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)       
        return x, (h_n, c_n)


class BahdanauAttention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, seq_len):
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        score = self.linear2(F.tanh(self.linear1(torch.cat((encoder_outputs, decoder_hidden), dim=2))))
        score = score.squeeze(2).unsqueeze(1)
        attention = F.softmax(score, dim=-1)
        context = torch.bmm(attention, encoder_outputs)
        return context, attention


class ReconstructionDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):   
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, encoder_outputs, seq_len):
        decoder_states = encoder_states
        decoder_outputs = []
        attentions = []
        for i in range(seq_len):
            x, decoder_states, attention = self.forward_step(x, decoder_states, encoder_outputs, seq_len)
            decoder_output = self.out(x)
            decoder_outputs.append(decoder_output)
            attentions.append(attention)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)       
        return decoder_outputs.flip(1)

    def forward_step(self, x, decoder_states, encoder_outputs, seq_len):
        decoder_hidden = decoder_states[0][-1]
        context, attention = self.attention(decoder_hidden, encoder_outputs, seq_len)
        x, decoder_states = self.lstm(torch.cat((context, x), dim=2), decoder_states)
        return x, decoder_states, attention


class PredictionDecoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0):
        super().__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, encoder_outputs, seq_len):
        decoder_hidden = encoder_states[0][-1]
        context, attention = self.attention(decoder_hidden, encoder_outputs, seq_len)
        x, _ = self.lstm(torch.cat((x, context), dim=2), encoder_states)
        decoder_output = self.out(x)
        return decoder_output


class MultitaskSequenceModel(nn.Module):
    
    def __init__(self, feature_size, embedding_size, num_layers, dropout=0):
        super().__init__()
        self.encoder = Encoder(feature_size, embedding_size, num_layers, dropout)
        self.recon_decoder = ReconstructionDecoder(2*embedding_size, embedding_size, feature_size, num_layers, dropout)
        self.pred_decoder = PredictionDecoder(2*embedding_size, embedding_size, feature_size, num_layers, dropout)
    
    def forward(self, x):
        seq_len = x.shape[1]
        encoder_outputs, encoder_states = self.encoder(x)
        recon = self.recon_decoder(encoder_outputs[:, -1, :].unsqueeze(1), encoder_states, encoder_outputs, seq_len)
        pred = self.pred_decoder(encoder_outputs[:, -1, :].unsqueeze(1), encoder_states, encoder_outputs, seq_len)
        return recon, pred

