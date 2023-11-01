'''
Descripttion: 早期写的，已弃用
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-11-01 19:07:29
'''
import math
import torch
from torch import nn as nn
from einops.layers.torch import Rearrange
from model.resnet1d import ResNet1D

class Mlp(nn.Module):
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
    
class BidirectionalGRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Basemodel(nn.Module):
    def __init__(self,window_size,n_sample,V=2,C=64,L=4) -> None:
        super().__init__()
        kernel_size = 10
        stride=5

        self.resnet1d = ResNet1D(in_channels=V*C*L,
                                 base_filters=V*C*L,
                                 kernel_size_first=n_sample//2,
                                 stride_first=n_sample//10,
                                 kernel_size=3,
                                 stride=1,
                                 groups=8,
                                 n_block=12,
                                 n_classes=1*V*C*L,
                                 increasefilter_gap=3,
                                 downsample_gap=2)
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(input_size=1*V*C*L,hidden_size=2*V*C*L,output_size=2*V*C*L,bidirectional=True),
            BidirectionalLSTM(input_size=2*V*C*L,hidden_size=2*V*C*L,output_size=1*V*C*L,bidirectional=True)
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=V*C*L,num_heads=4,batch_first=True)
        self.pool1 = nn.Conv1d(in_channels=1*V*C*L,out_channels=2*V*C*L,kernel_size=n_sample,stride=n_sample)
        
        self.relu   = nn.GELU()
        
        self.mlp = Mlp(in_features=2*V*C*L,hidden_features=V*C*L,out_features=C*L,drop=0.)
        
        self.fc_out = nn.Linear(C*L,1)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=2*L*C,num_heads=4,batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=2*L*C,num_heads=4,batch_first=True)
        self.tanh = nn.Tanh()
        # self.apply(self.weight_init)
        self.pe = PositionalEncoding(d_model=1*V*C*L,max_len=window_size*n_sample)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1*V*C*L, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self,input:torch.Tensor)->torch.Tensor:
        N,T,V,C = input.size()
        x = input.reshape(N,T,-1)
        # x = self.resnet1d(x.permute(0,2,1))

        # N,T,2*V*C
        x = self.rnn(x)
        # N,T/4,2*V*C
        # x = x.reshape(N,T,V,2*C)
        # amplitude = x[:,:,0,:]
        # phase = x[:,:,1,:]
        # # N,T,2*C
        # y,_ = self.cross_attn(query=amplitude,key=phase,value=phase)
        # y = self.relu(y)
        # y,_ = self.self_attn(query=y,key=y,value=y)

        # print(x.size())
        # x = self.transformer_encoder(self.pe(x))
        y = self.pool1(x.permute(0,2,1)).permute(0,2,1)
        # N,2*C
        z = self.mlp(y)
        z = self.fc_out(z).flatten(1)
        # print(z.size())
        return z
    
    # def weight_init(self,m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     # 也可以判断是否为conv2d，使用相应的初始化方式 
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     # 是否为批归一化层
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
