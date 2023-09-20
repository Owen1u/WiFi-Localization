'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-09-20 19:28:05
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

class Model(nn.Module):
    def __init__(self) -> None:
        ROOM=3
        T=25
        L=4
        V=2
        C=64
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=L*C,num_heads=4,batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=L*C,num_heads=4,batch_first=True)
        # self.rearrange = Rearrange('b p1 p2 t c -> b (p1 p2) t c', p1 = 3, p2 = 4)
        self.rnn = BidirectionalLSTM(input_size=2*ROOM*L*V*C,hidden_size=2*ROOM*L*V*C,output_size=ROOM*L*V*C)
        # self.adaptiveavgpool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.adaptiveavgpool2 = nn.AdaptiveAvgPool2d((1, None))
        self.mlp = Mlp(in_features=L*C,hidden_features=L*C,out_features=C)
        self.cls= Mlp(in_features=C,hidden_features=2*C,out_features=4)
        self.bn = nn.BatchNorm1d(num_features=C)
        self.ln = nn.LayerNorm([ROOM,L*C])
        self.cnn = nn.Conv1d(in_channels=ROOM*L*V*C,out_channels=2*ROOM*L*V*C,padding=1,stride=1,kernel_size=3)
        self.bn2 = nn.BatchNorm1d(num_features=2*ROOM*L*V*C)
        
    def forward(self,input:torch.Tensor):
        N,ROOM,T,L,v,C = input.size()
        x = input.permute(0,2,1,3,4,5).reshape(N,T,-1)
        
        x = self.bn2(self.cnn(x.permute(0,2,1))).permute(0,2,1)
        
        x = self.rnn(x)
        x = self.adaptiveavgpool2(x).squeeze(1)
        x = x.reshape(N,ROOM,L,v,C)
        
        real = x[:,:,:,0,:].reshape(N,ROOM,-1)
        imag = x[:,:,:,1,:].reshape(N,ROOM,-1)
        y,_ = self.cross_attn(query=real, key=imag, value=imag)
        y = self.ln(y)
        y,_ = self.self_attn(query=y, key=y, value=y)
        # [N,ROOM,L*C]
        
        z = self.mlp(y)
        z = self.bn(z.permute(0,2,1)).permute(0,2,1)
        z = self.adaptiveavgpool2(z).squeeze(1)
        z = self.cls(z)

        return z.log_softmax(1)
        
class Basemodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu   = nn.GELU()
        self.bn = nn.BatchNorm1d(num_features=512)
        self.fc1 = nn.Linear(512,1024)
        self.ln1 = nn.LayerNorm([1024])
        self.fc2  = nn.Linear(1024,2048)
        self.ln2 = nn.LayerNorm([2048])
        self.fc3  = nn.Linear(2048,512)
        self.ln3 = nn.LayerNorm([512])
        self.fc_out = nn.Linear(512,4)

        self.dropout = nn.Dropout(0.05)
        
        # self.pool = nn.AvgPool1d(25,stride=25)
        self.pool = nn.Linear(25,1)
        self.insnorm = nn.InstanceNorm1d(25,affine=True)
        self.rnn = BidirectionalGRU(input_size=4*2*64,hidden_size=2*4*2*64,output_size=4*2*64)
        
        self.apply(self.weight_init)
        
    def forward(self,x:torch.Tensor):
        N,T,L,C,V = x.size()
        x = x.reshape(N,T,-1)
        # x = self.insnorm(x)
        # print(x[0,0])
        # x = self.rnn(x)
        # x = self.pool(x.permute(0,2,1)).squeeze(-1)
        x = self.pool(x.permute(0,2,1)).squeeze(-1)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        # x = self.ln2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        # x = self.ln3(x)
        x = self.relu(x)
        
        x = self.fc_out(x)
        print(x)
        
        return x.softmax(dim=1)
    
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
