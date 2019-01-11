import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

from utils import *

#
# class Encoder(nn.Module):
#
#
#     def __init__(self, hidden_size,
#                  input_dropout_p=0,
#                  n_layers=1, bidirectional=True,
#                  embedding=None, update_embedding=True):
#
#         super(Encoder, self).__init__()
#
#         self.input_dropout = nn.Dropout(p=input_dropout_p)
#
#         self.embedding = nn.Embedding(4, hidden_size)
#
#         # if embedding is not None:
#         #     self.embedding.weight = nn.Parameter(embedding)
#         #
#         # self.embedding.weight.requires_grad = update_embedding
#
#         self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,bidirectional=bidirectional)
#
#
#     def forward(self, input_var):
#         # print(input_var.size())
#         embedded = self.embedding(input_var)
#         embedded = self.input_dropout(embedded)
#         embedded = torch.transpose(embedded,1,0)
#         print(embedded.size())
#         output, hidden = self.rnn(embedded)
#
#         return output, hidden
#
#
#
# class Decoder(nn.Module):
#
#
#     def __init__(self, output_size, hidden_size,n_layers=1, bidirectional=True,input_dropout_p=0):
#
#         super(Decoder, self).__init__()
#
#         self.bidirectional_encoder = bidirectional
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.init_input = None
#         self.input_dropout = nn.Dropout(p=input_dropout_p)
#
#         self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, n_layers)
#
#
#         self.embedding = nn.Linear(self.output_size, self.hidden_size)
#         self.attention = Attention(self.hidden_size)
#
#         self.Linear = nn.Linear(self.hidden_size, self.output_size)
#
#         self.list_attention = []
#
#     def forward_step(self, input_var, hidden, encoder_outputs):
#         # batch_size = input_var.size(1)
#         # output_size = input_var.size(0)
#
#         print(input_var.size())
#
#         embedded = self.embedding(input_var)
#         # embedded = self.embedding(input_var.squeeze(2).long())
#         embedded = self.input_dropout(embedded)
#
#         # embedded = torch.transpose(embedded,1,0)
#         print(embedded.size())
#
#         hidden = self._cat_directions(hidden) if self.bidirectional_encoder else encoder_hidden
#
#         print(embedded.size())
#         print(hidden.size())
#
#         output, hidden_t = self.rnn(embedded, hidden)
#         output = nn.Sigmoid(self.Linear(output))
#
#
#         #Prepare the next step input using attention
#         input_decoder_tplus1 , attn = self.attention(hidden_t, encoder_outputs)
#
#         # output_step = nn.Sigmoid(self.Linear(output_attn.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
#
#         return output, input_decoder_tplus1, attn
#
#     def forward(self, encoder_hidden=None, encoder_outputs=None):
#         list_attention = []
#
#         decoder_hidden = encoder_hidden
#
#         # print('inputs')
#         # print(encoder_outputs.size())
#         # print(decoder_hidden.size())
#
#         decoder_outputs = []
#         # sequence_symbols = []
#
#         batch_size = encoder_outputs.size(0)
#         decoder_input = numpy2var(np.zeros((batch_size,self.output_size))) #First entry of the decoder
#         for di in range(len(encoder_outputs)):
#             decoder_output, decoder_input, attn_vect = self.forward_step(input_var=decoder_input,
#                                                                           hidden=decoder_hidden,
#                                                                           encoder_outputs=encoder_outputs)
#             # decoder_input = decoder_hidden.squeeze(1)
#             list_attention.append(attn_vect)
#             decoder_outputs.append(decoder_output)
#
#         decoder_output = torch.cat(decoder_output)
#         list_attention = torch.cat(list_attention)
#
#         return decoder_outputs, list_attention
#
#     # def _init_state(self, encoder_hidden):
#     #     """ Initialize the encoder hidden state to send it to decoder. """
#     #     if encoder_hidden is None:
#     #         return None
#     #     if isinstance(encoder_hidden, tuple):
#     #         encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
#     #     else:
#     #         encoder_hidden = self._cat_directions(encoder_hidden)
#     #     return encoder_hidden
#
#     def _cat_directions(self, encoder_hidden):
#         """ If the encoder is bidirectional, do the following transformation.
#             (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
#         """
#         h = torch.cat(encoder_hidden,0)
#         return h
#
#
# class Attention(nn.Module):
#     r"""
#     Applies an attention mechanism on the output features from the decoder.
#
#     .. math::
#             \begin{array}{ll}
#             x = context*output \\
#             attn = exp(x_i) / sum_j exp(x_j) \\
#             output = \tanh(w * (attn * context) + b * output)
#             \end{array}
#
#     Args:
#         dim(int): The number of expected features in the output
#
#     Inputs: output, context
#         - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
#         - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
#
#     Outputs: output, attn
#         - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
#         - **attn** (batch, output_len, input_len): tensor containing attention weights.
#
#     Attributes:
#         linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
#         mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
#
#     Examples::
#
#          >>> attention = seq2seq.models.Attention(256)
#          >>> context = Variable(torch.randn(5, 3, 256))
#          >>> output = Variable(torch.randn(5, 5, 256))
#          >>> output, attn = attention(output, context)
#
#     """
#     def __init__(self, dim):
#         super(Attention, self).__init__()
#         self.Linear = nn.Linear(dim*2, dim)
#         self.mask = None
#
#     def set_mask(self, mask):
#         """
#         Sets indices to be masked
#
#         Args:
#             mask (torch.Tensor): tensor containing indices to be masked
#         """
#         self.mask = mask
#
#     def forward(self, hidden, context):
#
#         attn = torch.bmm(hidden, context.transpose(1, 2))
#         attn = Linear
#         # batch_size = hidden.size(0)
#         # hidden_size = output.size(2)
#         # input_size = context.size(1)
#         # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
#         # if self.mask is not None:
#         #     attn.data.masked_fill_(self.mask, -float('inf'))
#
#         attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
#
#         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
#         mix = torch.bmm(attn, context)
#
#         # concat -> (batch, out_len, 2*dim)
#         combined = torch.cat((mix, output), dim=2)
#         # output -> (batch, out_len, dim)
#         output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
#
#         return output, attn
#
#
class Seq2Seq(nn.Module):

    def __init__(self,output_size, hidden_size,
                      input_dropout_p=0,
                      n_layers=1):
        super(Seq2Seq, self).__init__()

        #Encoder Layers
        self.Embedding = nn.Embedding(4, hidden_size)
        self.Dropout = nn.Dropout(p=input_dropout_p)
        self.Encoder = nn.GRU(hidden_size,hidden_size, n_layers,bidirectional=True,batch_first=True)


        #Attention layer
        self.Tanh = nn.Tanh()
        self.LinearAttn = nn.Linear(hidden_size*3,1)
        self.Softmax = nn.Softmax(dim=0)

        #Decoder layers
        self.EH2DH = nn.Linear(hidden_size*2,hidden_size) #Encoder hidden to decoder hidden
        self.Decoder = nn.GRU(hidden_size*2,hidden_size, n_layers,bidirectional=False,batch_first=True)
        self.LinearOut = nn.Linear(hidden_size,output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):
        #Encoder
        embedded = self.Embedding(x)
        embedded =  self.Dropout(embedded)
        encoder_outputs, last_encoder_hidden = self.Encoder(embedded)


        #Decoder-Attention
        last_encoder_hidden = self._cat_directions(last_encoder_hidden)
        decoder_hidden = self.EH2DH(last_encoder_hidden)

        decoder_outputs=[]
        attn_list = []
        # for k in tqdm(range(encoder_outputs.size(1)),desc='FeedForeward'):
        for k in range(encoder_outputs.size(1)):
            #Attention layer
            mix = torch.cat((decoder_hidden.squeeze(0).repeat(encoder_outputs.size(1),1),encoder_outputs.squeeze(0)),1)
            attn = self.Tanh(self.LinearAttn(mix))
            attn = self.Softmax(attn)
            input_decoder = torch.mm(torch.transpose(attn,1,0),encoder_outputs.squeeze(0))

            #Decoder
            decoder_output, decoder_hidden = self.Decoder(input_decoder.unsqueeze(0),decoder_hidden)

            decoder_output = self.Sigmoid(self.LinearOut(decoder_output)).squeeze(1)

            decoder_outputs.append(decoder_output)
            attn_list.append(var2numpy(attn))

        decoder_outputs = torch.cat(decoder_outputs,1)

        return decoder_outputs,attn_list


    def _cat_directions(self,h):
        """
        If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
