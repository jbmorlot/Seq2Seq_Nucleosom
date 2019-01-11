import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim

from torch.autograd import Variable
from torch.utils.data import Dataset,random_split,DataLoader

import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchsample as ts

from tensorboardX import SummaryWriter

from sklearn.metrics import adjusted_mutual_info_score

from tqdm import tqdm

from model import Seq2Seq
from loss import *
from utils import *



class Seq2Seq_model:
    '''
        Seq2seq architecture with attention which aim to predict the nucleosom frequency
        from DNA sequence
    '''

    def __init__(self,fasta_path,histone_path,
                 output_size,Nepochs=10,
                 seq_len=50000,
                 batchsize=1,
                 hidden_size=128,
                 input_dropout_p=0,
                 n_layers=1,
                 use_cuda=True):


        self.use_cuda = use_cuda

        self.batchsize = batchsize

        self.Seq2Seq = Seq2Seq(output_size, hidden_size,
                              input_dropout_p=input_dropout_p,
                              n_layers=n_layers)
        if use_cuda:
            self.Seq2Seq.cuda()



        # self.loss = Perplexity()
        self.loss = nn.CosineSimilarity()
        if use_cuda:
            self.loss.cuda()

        self.optim = optim.Adam(self.Seq2Seq.parameters(),lr=0.0001,betas=(0.9,0.999))

        self.seq_len = seq_len

        train_dataset,val_dataset = get_dataset(histone_path,fasta_path,self.seq_len)

        self.train_loader = DataLoader(train_dataset,
                                        batch_size=self.batchsize,
                                        shuffle=False,
                                        num_workers=30,
                                        pin_memory=True)

        self.val_loader = DataLoader(val_dataset,
                                        batch_size=self.batchsize,
                                        shuffle=False,
                                        num_workers=30,
                                        pin_memory=True)

        self.StartEpochs=0
        self.Nepochs = Nepochs

        self.train_len = len(self.train_loader)
        self.val_len = len(self.val_loader)

        self.train_iter = int(np.ceil(self.train_len/self.batchsize))
        self.val_iter = int(np.ceil(self.val_len/self.batchsize))


        self.loss_record = {key:[] for key in ['train','val']}
        self.writer = SummaryWriter('./logs/') #Writter for image saving

    def fit(self):

        for self.epoch in tqdm(range(self.StartEpochs,self.Nepochs)):

            #In training phase weight are optimized
            self.Seq2Seq.train()
            self.phase = 'train'

            self.clear_loss_records()

            for self.counter,(self.X,self.Y) in enumerate(tqdm(self.train_loader,total=self.train_iter,desc='train')):

                # self.X = torch.transpose(self.X,1,0)
                # self.Y = torch.transpose(self.Y,1,0)

                self.X = Variable(self.X,requires_grad=True)#.cuda()
                self.Y = Variable(self.Y)#.cuda()

                if self.use_cuda:
                    self.X = self.X.long().cuda()
                    self.Y = self.Y.cuda()

                self.foreward()
                self.backward()
                self.record_loss()
                self.tensorboard()

    def foreward(self):
        self.decoder_outputs, self.attention_list = self.Seq2Seq(self.X)

    def backward(self):
        self.compute_loss()
        self.S2S_loss.backward()
        self.optim.step()


    def compute_loss(self):
        self.S2S_loss = self.loss(self.decoder_outputs,self.Y)


    def record_loss(self):
        self.loss_record[self.phase].append(var2numpy(self.S2S_loss.mean(),use_cuda=self.use_cuda))

    def clear_loss_records(self):
        for p in ['train','val']:
            self.loss_record[p] = []

    def validate(self):

        #In training phase weight are optimized
        self.Seq2Seq.eval()
        self.phase = 'val'
        for self.counter,(self.X,self.Y) in enumerate(tqdm(self.test_loader,total=self.val_iter,desc='validation')):

            self.X = Variable(self.X,requires_grad=True)#.cuda()
            self.Y = Variable(self.Y).long()#.cuda()

            if self.use_cuda:
                self.X = self.X.cuda()
                self.Y = self.Y.cuda()

            with torch.no_grad():
                self.foreward()
                self.compute_loss()
                self.record_loss()



    def predict(self,fasta_path,batchsize=1):

        #Loading FASTA file
        f = open(fasta_path,'r')
        f.readline() #Remove header
        sequence=''
        for l in f:
            sequence  = sequence + l.replace('\n','').lower()

        #Split the sequence in chunks of size seq_len
        Num_chunck = np.floor(len(sequence)/self.seq_len)
        sequences = np.empty((Num_chunck,self.seq_len))
        for i in range(Num_chunck):
            sequences[i,:] = sequence[i*self.seq_len:(i+1)*self.seq_len]

        self.Seq2Seq.eval()
        total_iter = np.ceil(len(sequences)/batchsize)

        decoder_outputs=[]
        attention_list=[]

        for counter,(X,lt) in enumerate(tqdm(sequences,total=total_iter,desc='prediction')):

            X = Variable(X)
            if self.use_cuda:
                X = X.cuda()

            with torch.no_grad():
                self.foreward()
                decoder_outputs.append(self.decoder_outputs)
                attention_list.append(self.attention_list)

        decoder_outputs = np.hstack(decoder_outputs)
        attention_list = np.hstack(attention_list)

        return decoder_outputs, attention_list

    def tensorboard(self):
        # ===Add scalar losses===
        for p in ['train','val']:
            prefix = p+'/'
            info = {}
            info[prefix] =  np.array(self.loss_record[p]).mean()

            for tag, value in info.items():
                self.writer.add_scalars(tag, {tag:value}, self.epoch)
