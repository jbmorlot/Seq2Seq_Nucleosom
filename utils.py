import numpy as np
import pickle
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class dataset_h5(Dataset):
    def __init__(self,sequence,hist_occ,device,batch_size):
        super(dataset_h5, self).__init__()
        self.sequence = torch.from_numpy(sequence).long()
        self.hist_occ = torch.from_numpy(hist_occ).float()
        self.device = device
        self.batch_size=batch_size

        if self.hist_occ.shape[0]!=self.sequence.shape[0]:
            print('The two list needs to have the same length')
            print('(Fasta len, Nucleosom len) =  ({0},{1})'.format(self.sequence.shape[0],self.hist_occ.shape[0]))

            raise AttributeError

        self.sequence = self.sequence.to(self.device)
        self.hist_occ = self.hist_occ.to(self.device)

        self.M = self.hist_occ.shape[2]#//self.batch_size
        self.index=0
    def __iter__(self):
       return self
    def __next__(self):

        while True:
            if self.index==self.M:
                self.index=0
                raise StopIteration()

            seq = self.sequence[:,:,self.index].unsqueeze(1)
            hist = self.hist_occ[:,:,self.index].unsqueeze(1)

            self.index+=1

            return seq,hist

    def __len__(self):
        return self.M


def get_dataset(histone_path,fasta_path,seq_len_DNA,seq_len_hist,device,batch_size):

    Xtrain,Xval = get_fasta(fasta_path,seq_len_DNA,seq_len_hist,batch_size)
    Ytrain,Yval = get_histones(histone_path,seq_len_DNA,seq_len_hist,batch_size)
    return dataset_h5(Xtrain,Znorm(Ytrain),device,batch_size),dataset_h5(Xval,Znorm(Yval),device,batch_size)


def get_histones(histone_path,seq_len_DNA,seq_len_hist,batch_size,split=True):
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

    hist_occ[hist_occ>3]=3 # Removing far outliers

    #Split the sequence in chunks of size seq_len
    len_per_batch = int(np.floor(len(hist_occ)/batch_size))
    Num_chunck = int(np.ceil(len_per_batch/seq_len_hist))
    hist_occs = np.zeros((batch_size,seq_len_hist,Num_chunck),dtype=np.float32)
    for b in range(batch_size):
        for i in range(Num_chunck):
            if b*len_per_batch+(i+1)*seq_len_hist>=len(hist_occ):
                break
            hist_occs[b,:,i] = hist_occ[b*len_per_batch+i*seq_len_hist:b*len_per_batch+(i+1)*seq_len_hist]

    if split:
        train_len = int(Num_chunck*0.8)
        return hist_occs[:,:,:train_len],hist_occs[:,:,train_len:]
    else:
        return hist_occs


def get_fasta(fasta_path,seq_len_DNA,seq_len_hist,batch_size,split=True):
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

    #Split the sequence in chunks of size seq_len
    len_per_batch = int(np.floor(len(sequence)/batch_size))
    Num_chunck = int(np.ceil(len_per_batch/seq_len_DNA))
    sequences = np.zeros((batch_size,seq_len_DNA,Num_chunck),dtype=np.float32)
    for b in range(batch_size):
        for i in range(Num_chunck):
            start = int(b*len_per_batch + i*seq_len_hist - seq_len_DNA/2)
            stop =  int(b*len_per_batch + i*seq_len_hist + seq_len_DNA/2)
            for k,pos in enumerate(range(start,stop)):
                if pos>=0 and pos<len(sequence):
                    sequences[b,k,i] = OHE(sequence[pos])
                else:
                    sequences[b,k,i] = 4

    if split:
        train_len = int(Num_chunck*0.8)

        return sequences[:,:,:train_len],sequences[:,:,train_len:]
    else:
        return sequences


def Znorm(mat):
    '''
        Z-score normalization
    '''
    std = mat.std(axis=1)[:,None,:]
    std[std==0]=1
    mean = mat.mean(axis=1)[:,None,:]
    return (mat-mean)/std

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def trch2npy(trch,device):
    if device.type=="cpu":
        return trch.data.numpy()
    else:
        return trch.cpu().data.numpy()

# def data_parallel(module, input, device_ids, output_device=None):
#     '''
#         Allow to launch the model over multiple GPUs in parallel
#     '''
#     if not device_ids:
#         return module(input)
#
#     if output_device is None:
#         output_device = device_ids[0]
#
#     replicas = nn.parallel.replicate(module, device_ids)
#     inputs = nn.parallel.scatter(input, device_ids)
#     replicas = replicas[:len(inputs)]
#     outputs = nn.parallel.parallel_apply(replicas, inputs)
#     return nn.parallel.gather(outputs, output_device)
