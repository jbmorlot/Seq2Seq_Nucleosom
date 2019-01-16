import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    '''
        Sequence encoder using RNN
    '''
    def __init__(self, input_size, hidden_size,n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,
                          bidirectional=True,batch_first=True,
                          num_layers=n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input.long())
        output, hidden = self.gru(embedded, torch.transpose(hidden,1,0))

        hidden = torch.transpose(hidden,1,0)

        return output, hidden

    def initHidden(self,device,batch_size):
        return torch.zeros(batch_size,2*self.n_layers,self.hidden_size,dtype=torch.float,device=device)


class EncoderHidden2DecoderHidden(nn.Module):
    '''
        Concatenate hidden layers from encoder and decoder at the last step
        in order to initialise decoder hidden layer
    '''
    def __init__(self, hidden_size):
        super(EncoderHidden2DecoderHidden, self).__init__()
        self.hidden_size = hidden_size
        self.Linear = nn.Linear(hidden_size*3,hidden_size)
    def forward(self,encoder_hidden,decoder_hidden):

        encoder_hidden = self.cat_directions(encoder_hidden)

        hidden = torch.cat((encoder_hidden,decoder_hidden),2)

        hidden = self.Linear(hidden)

        return hidden

    def cat_directions(self,h):
        """
        If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        return torch.cat([h[:,0:h.size(1):2], h[:,1:h.size(1):2]], 2)



class AttnDecoderRNN(nn.Module):
    '''
        Sequence decoder using RNN and attention over encoder outputs
    '''
    def __init__(self, input_size,hidden_size, input_seq_len, output_size,n_layers=1,dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.input_seq_len = input_seq_len
        self.n_layers = n_layers

        # self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (n_layers+1), self.input_seq_len)
        self.attn_combine = nn.Linear(self.hidden_size *3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          batch_first=True,
                          num_layers=n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.Sig = nn.Sigmoid()
        self.Relu = nn.ReLU()

    def forward(self, input, hidden0, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        embedded = self.Relu(embedded)

        if hidden0.size(0)>1:
            hidden = torch.cat([h.contiguous().view(1,1,-1) for h in hidden0],0)
        else:
            hidden = hidden0.view(1,1,-1)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 2)), dim=2)

        attn_applied = torch.bmm(attn_weights,torch.transpose(encoder_outputs,1,2))


        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = self.Relu(output)
        hidden0 = torch.transpose(hidden0,0,1)
        output, hidden = self.gru(output, hidden0)
        hidden = torch.transpose(hidden,0,1)

        output = self.out(output)

        return output, hidden, attn_weights

    def initHidden(self,device,batch_size):
        return torch.zeros(batch_size,self.n_layers, self.hidden_size,dtype=torch.float,device=device)
