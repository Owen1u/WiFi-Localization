'''
Descripttion: Resnet1d + RNN + MLP
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-02 22:50:18
'''
import math
import torch
from torch import nn as nn
from einops.layers.torch import Rearrange

class Basiclayer(nn.Module):
    def __init__(self,in_channels,hid_channels,out_channels,downsample) -> None:
        super().__init__()
        stride = 1
        if downsample == True:
            stride = 2
        self.layer = nn.Sequential(
            torch.nn.Conv1d(in_channels, hid_channels, 1, stride),
            torch.nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hid_channels, hid_channels, 3, padding=1),
            torch.nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(hid_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU()
        )
        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels,1,stride)
        else:
            self.res_layer = None
            
    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual

class Resnet1d(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels,32,kernel_size=15,stride=2,padding=7),
            nn.MaxPool1d(kernel_size=3,stride=3),
            
            Basiclayer(32,32,128,downsample=False),
            Basiclayer(128,32,128,downsample=False),
            Basiclayer(128,32,128,downsample=False),
            
            Basiclayer(128,64,256,downsample=True),
            Basiclayer(256,64,256,downsample=False),
            Basiclayer(256,64,256,downsample=False),
            
            Basiclayer(256,128,512,downsample=True),
            Basiclayer(512,128,512,downsample=False),
            Basiclayer(512,128,512,downsample=False),
            
            # Basiclayer(256,128,512,downsample=True),
            # Basiclayer(512,128,512,downsample=False),
            
            # torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512,out_channels)
        )
        
    def forward(self,x):
        x = self.features(x)
        x = x.permute(0,2,1)
        x = self.classifer(x)
        return x

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Model(nn.Module):
    def __init__(self,*args,**kw) -> None:
        super().__init__()
        
        self.resnet1d = Resnet1d(in_channels=8,out_channels=256)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(input_size=256,hidden_size=512,output_size=512,bidirectional=True),
            BidirectionalLSTM(input_size=512,hidden_size=512,output_size=256,bidirectional=True)
        )
        self.mlp1 = MLP(in_features=256,hidden_features=16,out_features=2,drop=0.)
        self.mlp2 = MLP(in_features=256,hidden_features=16,out_features=1,drop=0.)
        
    def forward(self,x:torch.Tensor):
        # print(x.size())
        # x = x.reshape(0,2,1)
        x = self.resnet1d(x)
        x = self.rnn(x)
        manned = torch.softmax(self.mlp1(x), dim=2).permute(0,2,1)
        num_human = self.mlp2(x).squeeze(-1)
        return manned,num_human
        