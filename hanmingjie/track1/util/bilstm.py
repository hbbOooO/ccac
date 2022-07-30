#!/usr/bin/env python
# coding:utf8
# code from https://github.com/albertwy/BiLSTM.git 

from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class BiLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,biFlag,dropout=0.5):
        super(BiLSTM,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag

        self.layer1=nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=True))
        if(biFlag):
        # 如果是双向，额外加入逆向层
                self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=True))


        # self.layer2=nn.Sequential(
        #     nn.Linear(hidden_dim*self.bi_num,output_dim),
        #     nn.LogSoftmax(dim=2)
        # )

        # self.to(self.device)

    def init_hidden(self, batch_size, x):
        return (torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(device=x.device),
                torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(device=x.device))
    

    def forward(self, x, length):
        batch_size=x.size(0)
        max_length=torch.max(length)
        x=x[:,0:max_length,:]
        # x,length=sort_batch(x,length)
        # x,y=x.to(self.device),y.to(self.device)
        hidden=[self.init_hidden(batch_size, x) for l in range(self.bi_num)]

        out=[x,reverse_padded_sequence(x,length,batch_first=True)]
        for l in range(self.bi_num):
            # pack sequence
            out[l]=pack_padded_sequence(out[l],length.cpu(),batch_first=True, enforce_sorted=False)
            out[l],hidden[l]=self.layer1[l](out[l],hidden[l])
            # unpack
            out[l],_=pad_packed_sequence(out[l],batch_first=True)
            # 如果是逆向层，需要额外将输出翻过来
            if(l==1):out[l]=reverse_padded_sequence(out[l],length.cpu(),batch_first=True)
    
        if(self.bi_num==1):out=out[0]
        else:out=torch.cat(out,2)
        # out=self.layer2(out)
        # out=torch.squeeze(out)
        return out


def sort_batch(data,length):
    batch_size=data.size(0)
    # 先将数据转化为numpy()，再得到排序的index
    inx=torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data=data[inx]
    length=length[inx]
    # length转化为了list格式，不再使用torch.Tensor格式
    length=list(length.numpy())
    return (data,length)


def reverse_padded_sequence(inputs, lengths, batch_first=True):
    '''这个函数输入是Variable，在Pytorch0.4.0中取消了Variable，输入tensor即可
    '''
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = Variable(ind.expand_as(inputs))
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


