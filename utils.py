import numpy as np
import pickle
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset
import datetime

from sklearn.preprocessing import StandardScaler,OneHotEncoder

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class dataset_h5(Dataset):
    def __init__(self,hist_occ,sequence):
        super(dataset_h5, self).__init__()
        self.hist_occ = hist_occ.T
        self.sequence = sequence.T

        if self.hist_occ.shape[0]!=self.sequence.shape[0]:
            print('The two list needs to have the same length')
            print('(Fasta len, Nucleosom len) =  ({0},{1})'.format(self.sequence.shape[0],self.hist_occ.shape[0]))

            raise AttributeError

        # self.hist_occ = np.transpose(hist_occ,(1,2,0))
        # self.sequence = np.transpose(sequence,(1,2,0))

        # self.M = self.hist_occ.shape[1]
        self.M = self.hist_occ.shape[0]

    def __getitem__(self, index):

        # return self.sequence[:,index,:],self.hist_occ[:,index,:]
        return self.sequence[index],self.hist_occ[index]

    def __len__(self):
        return self.M


def get_dataset(histone_path,fasta_path,seq_len):

    Xtrain,Xval = get_histones(histone_path,seq_len)
    Ytrain,Yval = get_fasta(fasta_path,seq_len)
    return dataset_h5(Xtrain,Ytrain),dataset_h5(Xval,Yval)

def get_histones(histone_path,seq_len):
    '''
        Load and split DNA and histone sequences before to format it into
        pytorch dataset
    '''

    f = open(histone_path,'r')
    f.readline() #Remove header
    hist_occ_list = [[float(n) for n in l[1:].replace('\n','').split('\t') ] for l in f]
    f.close()

    M = int(hist_occ_list[-1][1])
    hist_occ = np.zeros(M)
    for h in hist_occ_list:
        hist_occ[int(h[0]):int(h[1])] = h[2]


    #Split the sequence in chunks of size seq_len
    Num_chunck = int(len(hist_occ)/seq_len)
    hist_occs = np.zeros((seq_len,Num_chunck),dtype=np.float32)
    for i in range(Num_chunck):
        hist_occs[:,1] = hist_occ[i*seq_len:(i+1)*seq_len]

    train_len = int(Num_chunck*0.8)

    return hist_occs[:,:train_len],hist_occs[:,train_len:]


def get_fasta(fasta_path,seq_len):
    '''
        Load and split DNA and histone sequences before to format it into
        pytorch dataset
    '''
    #Loading FASTA file
    f = open(fasta_path,'r')
    f.readline() #Remove header
    sequence=''
    for l in f:
        sequence  = sequence + l.replace('\n','').lower()

    def OHE(seq):
        nuc = np.array(['a','c','t','g'])
        return np.where(seq==nuc)[0]
        # ohe = np.zeros(4)
        # ohe[seq==nuc] = 1
        # return ohe

    #Split the sequence in chunks of size seq_len
    Num_chunck = int(len(sequence)/seq_len)
    sequences = np.zeros((seq_len,Num_chunck),dtype=np.float32)
    for i in range(Num_chunck):
        for k in range(seq_len):
            sequences[k,i] = OHE(sequence[i*seq_len+k])

    # print(sequences.shape)
    train_len = int(Num_chunck*0.8)

    return sequences[:,:train_len],sequences[:,train_len:]



def var2numpy(var,use_cuda=True):
    if use_cuda:
        return var.cpu().data.numpy()
    return var.data.numpy()

def numpy2var(nmpy,use_cuda=True):
    if type(nmpy) is not np.ndarray or type(nmpy) is not np.array:
        nmpy = np.array([nmpy],dtype=np.float32)
    var = Variable(torch.from_numpy(nmpy))
    if use_cuda:
        return var.cuda()
    return var

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def data_parallel(module, input, device_ids, output_device=None):
    '''
        Allow to launch the model over multiple GPUs in parallel
    '''
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)
