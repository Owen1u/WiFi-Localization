'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-24 02:16:30
'''
import torch
from torch import nn as nn
from einops.layers.torch import Rearrange

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

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
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
        
class Basemodel(nn.Module):
    def __init__(self,T,V=2,C=64,L=4) -> None:
        super().__init__()
        kernel_size = 10
        stride=5
        self.cnn = nn.Conv1d(in_channels=V*C*L,out_channels=2*C*L,kernel_size=kernel_size,stride=stride)
        self.bn0 = nn.BatchNorm1d(num_features=2*C*L)
        self.pool0 = nn.Linear((T-kernel_size)//stride+1,1)
        
        self.rnn = BidirectionalGRU(input_size=2*V*C*L,hidden_size=4*V*C*L,output_size=2*V*C*L)
        self.self_attn = nn.MultiheadAttention(embed_dim=V*C*L,num_heads=4,batch_first=True)
        self.pool1 = nn.Linear(T,1)
        
        self.relu   = nn.GELU()
        
        self.fc1 = nn.Linear(V*C*L,2*V*C*L)
        self.bn1 = nn.BatchNorm1d(num_features=2*V*C*L)
        self.ln1 = nn.LayerNorm([2*V*C*L])
        self.fc2  = nn.Linear(2*C*L,4*C*L)
        self.bn2 = nn.BatchNorm1d(num_features=4*C*L)
        self.fc3  = nn.Linear(4*C*L,1*C*L)
        self.bn3 = nn.BatchNorm1d(num_features=1*C*L)
        self.fc_out = nn.Linear(C*L,5)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=2*L*C,num_heads=4,batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=2*L*C,num_heads=4,batch_first=True)
        
        self.apply(self.weight_init)
        
    def forward(self,input:torch.Tensor):
        N,T,V,C = input.size()
        input = input.reshape(N,T,-1)
        input_ = self.bn0(self.cnn(input.permute(0,2,1)))
        input_ = self.relu(input_)
        input_ = self.pool0(input_).squeeze(-1)

        # N,T,V*C
        x = self.fc1(input)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu(x)

        # N,T,2*V*C
        x = self.rnn(x)
        # N,T,2*V*C
        x = x.reshape(N,T,V,2*C)
        amplitude = x[:,:,0,:]
        phase = x[:,:,1,:]
        # N,T,2*C
        y,_ = self.cross_attn(query=amplitude,key=phase,value=phase)
        y = self.relu(y)
        y,_ = self.self_attn(query=y,key=y,value=y)
        y = self.pool1(y.permute(0,2,1)).squeeze(-1)
        y = self.relu(y)
        # N,2*C
        y = self.fc2(y+input_)
        y = self.bn2(y)
        y = self.relu(y)
        
        z = self.fc3(y)
        z = self.bn3(z)
        z = self.relu(z)
        
        return self.fc_out(z).softmax(dim=1)
    
    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
