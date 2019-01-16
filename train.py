import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import Dataset,random_split,DataLoader

from tensorboardX import SummaryWriter

from tqdm import tqdm

from model import *
from utils import *


class Seq2Seq:
    '''
        Seq2seq architecture with attention which aim to predict the nucleosom frequency
        from DNA sequence

        inputs:
             fasta_path: Path of the FASTA file
             histone_path: Path of the histone frequency in BEDGRAPH format
             input_size=5 : Input vocabulary size: 4 DNA nucletotides and 1 "out of range" value
             output_size=1: size of the output: 1-d vector
             Nepochs: Number of epochs
             batch_size: batch size
             seq_len_hist: Lenght of the predicted histone sequence
             seq_len_DNA: Lenght of the DNA sequence used to predict 'seq_len_hist' histone frequency.
                          The DNA sequence is centered around histones' sequence.
             hidden_size: size of the RNN hidden vectors
             dropout_p: Dropout of the attention over the  previous decoder output
             n_layers: Number of layers per RNN
    '''

    def __init__(self,fasta_path,histone_path,
                 input_size,
                 output_size,
                 Nepochs=10,
                 batch_size=1,
                 seq_len_DNA=50000,
                 seq_len_hist=5000,
                 hidden_size=128,
                 dropout_p=0,
                 n_layers=1):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size

        self.n_layers = n_layers

        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(input_size, self.hidden_size,n_layers=n_layers)
        self.decoder = AttnDecoderRNN(input_size,self.hidden_size, seq_len_DNA,output_size,
                                      dropout_p=dropout_p,n_layers=n_layers)
        self.EH2DH = EncoderHidden2DecoderHidden(self.hidden_size)

        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        if self.device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()
            self.EH2DH.cuda()
            self.criterion.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),lr=0.0001,betas=(0.9,0.999))
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),lr=0.0001,betas=(0.9,0.999))
        self.EH2DH_optimizer = optim.Adam(self.EH2DH.parameters(),lr=0.0001,betas=(0.9,0.999))

        # self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=0.01)
        # self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=0.01)
        # self.EH2DH_optimizer = optim.SGD(self.EH2DH.parameters(), lr=0.01)



        self.seq_len_DNA = seq_len_DNA
        self.seq_len_hist = seq_len_hist

        self.train_loader,self.val_loader = get_dataset(histone_path,
                                                fasta_path,
                                                self.seq_len_DNA,
                                                self.seq_len_hist,
                                                self.device,
                                                self.batch_size)

        self.StartEpochs=0
        self.Nepochs = Nepochs

        self.teacher_forcing_ratio=0.5

        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)

        self.loss_record = {key:[] for key in ['train','val']}
        self.writer = SummaryWriter('./logs/') #Writter for image saving

    def train(self):
        '''
            Train the model on training set
        '''

        for self.epoch in tqdm(range(self.StartEpochs,self.Nepochs)):

            self.encoder.train()
            self.decoder.train()
            self.EH2DH.train()
            self.phase = 'train'

            self.clear_loss_records()
            self.encoder_hidden = self.encoder.initHidden(self.device,self.batch_size)
            self.decoder_hidden = self.decoder.initHidden(self.device,self.batch_size)
            self.decoder_input = torch.zeros(self.batch_size,1,1,dtype=torch.float,device=self.device)


            for self.counter,(self.input_tensor,self.target_tensor) in enumerate(tqdm(self.train_loader,total=self.train_len,desc='train')):

                self.forward()
                self.backward()

            self.validate()
            self.write2tensorboard()

    def forward(self):
        '''
            Model forward pass: Encoder -> Decoder with attention
        '''
        #Gradient initialization
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.EH2DH_optimizer.zero_grad()

        encoder_outputs = torch.zeros(self.batch_size,self.hidden_size*2,self.seq_len_DNA, dtype=torch.float, device=self.device)
        self.decoder_outputs = torch.zeros_like(self.target_tensor, dtype=torch.float, device=self.device)

        #Record attention
        if self.phase=='pred':
            decoder_attentions = torch.zeros(self.seq_len_hist, self.seq_len_DNA, dtype=torch.float, device=self.device)

        # Encoder
        for ei in range(self.seq_len_DNA):
            encoder_output, self.encoder_hidden = self.encoder(self.input_tensor[:,:,ei], self.encoder_hidden)
            encoder_outputs[:,:,ei] = encoder_output[0, 0]
            if ei==self.seq_len_hist-1:
                self.encoder_hidden_out = self.encoder_hidden.detach()

        # Initialize decoder hidden using encoder hidden and decoder hidden at previous step
        self.decoder_hidden = self.EH2DH(self.encoder_hidden,self.decoder_hidden)


        # Teacher forcing: Use output given by the network at previous step or the ground truth value
        use_teacher_forcing = True if torch.rand(1) < self.teacher_forcing_ratio else False

        # Teacher forcing: Feed the target as the next input
        if use_teacher_forcing and self.phase=='train':
            for di in range(self.seq_len_hist):
                decoder_output, self.decoder_hidden, decoder_attention = self.decoder(self.decoder_input,
                                                                             self.decoder_hidden, encoder_outputs)
                self.decoder_input = self.target_tensor[:,:,di].unsqueeze(2)  # Teacher forcing
                self.decoder_outputs[:,0,di] = decoder_output.squeeze()


        # Without teacher forcing: use its own predictions as the next input
        else:
            for di in range(self.seq_len_hist):
                decoder_output, self.decoder_hidden, decoder_attention = self.decoder(
                    self.decoder_input, self.decoder_hidden, encoder_outputs)

                self.decoder_input = decoder_output.detach()  # detach from history as input for next step
                self.decoder_outputs[:,0,di] = decoder_output.squeeze()

        if self.phase=='train' or self.phase=='val':
            self.loss = self.criterion(self.decoder_outputs.float(), self.target_tensor.float())
            self.record_loss()

        #Detach hidden states
        self.decoder_hidden = self.decoder_hidden.detach()
        self.encoder_hidden = self.encoder_hidden_out #Already detached


        if self.phase=='pred':
            decoder_attentions = torch.cat([decoder_attention.detach().data for decoder_attention in decoder_attentions],0)
            return self.decoder_outputs.detach().data,decoder_attentions



    def backward(self):
        '''
            Model backward
        '''
        self.loss.backward()
        self.decoder_optimizer.step()
        self.EH2DH_optimizer.step()
        self.encoder_optimizer.step()


    def record_loss(self):
        self.loss_record[self.phase].append(trch2npy(self.loss.mean(),self.device))

    def clear_loss_records(self):
        for p in ['train','val']:
            self.loss_record[p] = []

    def validate(self):
        '''
            Compute loss score on validation set
        '''

        self.encoder.eval()
        self.decoder.eval()
        self.EH2DH.eval()
        self.phase = 'val'

        self.encoder_hidden = self.encoder.initHidden(self.device,self.batch_size)
        self.decoder_hidden = self.decoder.initHidden(self.device,self.batch_size)
        self.decoder_input = torch.zeros(self.batch_size,1,1,dtype=torch.float,device=self.device)


        for self.counter,(self.input_tensor,self.target_tensor) in enumerate(tqdm(self.val_loader,total=self.val_len,desc='validation')):

            with torch.no_grad():
                self.forward()
                self.record_loss()
                self.tensorboard()


    def predict(self,fasta_path):

        '''
            Compute decoder output and attention score on the input FASTA file
        '''

        # sequences = get_fasta(fasta_path,self.seq_len_DNA,self.seq_len_hist,split=False,batch_size=self.batch_size)
        #
        # sequences = torch.from_numpy(sequences).long().to(self.device)

        self.encoder.eval()
        self.decoder.eval()
        self.EH2DH.eval()
        self.phase = 'pred'
        self.encoder_hidden = self.encoder.initHidden(self.device,self.batch_size)
        self.decoder_hidden = self.decoder.initHidden(self.device,self.batch_size)
        self.decoder_input = torch.zeros(self.batch_size,1,1,dtype=torch.float,device=self.device)

        total_iter = np.ceil(len(sequences))

        decoder_outputs = []
        decoder_attentions = []
        # for self.input_tensor in tqdm(sequences,total=total_iter,desc='prediction'):
        for self.counter,(self.input_tensor,self.target_tensor) in enumerate(tqdm(self.train_loader,total=self.val_len,desc='prediction on training set')):

            with torch.no_grad():
                decoder_output,decoder_attention = self.forward()

            decoder_outputs.append(trch2npy(decoder_output,self.device))
            decoder_attentions.append(trch2npy(decoder_attention,self.device))

        decoder_outputs = np.concatenate(decoder_outputs,axis=0)

        return decoder_outputs,decoder_attentions


    def write2tensorboard(self):
        '''
            Write losses in order to follow the network evolution using
            tensorboard
        '''
        for p in ['train','val']:
            prefix = p+'/'
            info = {}
            info[prefix] =  np.array(self.loss_record[p]).mean()

            for tag, value in info.items():
                self.writer.add_scalars(tag, {tag:value}, self.epoch + self.counter)
