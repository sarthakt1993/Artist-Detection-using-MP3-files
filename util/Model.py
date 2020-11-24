import torch 
import torch.nn  as nn
from torch.autograd import Variable
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(conv_block,self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))
    
class Inception_Block(nn.Module):
    def __init__(self,input_channels,out_1x1,reduction_3x3,out_3x3,reduction_5x5,out_5x5,out_1x1pool):
        super(Inception_Block,self).__init__()
        self.branch1 = conv_block(input_channels,out_1x1,kernel_size=(1,1))
        
        self.branch2 = nn.Sequential(
                           conv_block(input_channels,reduction_3x3,kernel_size=(1,1)),
                           conv_block(reduction_3x3,out_3x3,kernel_size=(3,3),padding=(1,1)),)
        self.branch3 = nn.Sequential(
                        conv_block(input_channels,reduction_5x5,kernel_size=(1,1)),
                        conv_block(reduction_5x5,out_5x5,kernel_size=(5,5),padding=(2,2)),)
        self.branch4 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                        conv_block(input_channels,out_1x1pool,kernel_size=(1,1)))
    def forward(self,x):
        return torch.cat(
            [self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)

class MusicArtistClassificationModel(nn.Module):
    def __init__(self,input_channels,total_artists,use_cuda):
        super(MusicArtistClassificationModel,self).__init__()          
        self.layer1 = conv_block(input_channels,32,kernel_size=(3,3),padding=(1,1))
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2  = nn.Dropout(0.3)
        self.layer2 = Inception_Block(32,64,96,128,16,32,32)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.layer3 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        self.layer4  = conv_block(256,128,kernel_size=(3,3),padding=(1,1))
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.layer5  = conv_block(128,32,kernel_size=(3,3),padding=(1,1))
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.layer7 = nn.MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        self.rnn = nn.LSTM(
            input_size=1024, 
            hidden_size=32, 
            num_layers=2,
            batch_first=True,
            dropout=0.3)
        self.linear = nn.Linear(32,total_artists)
        self.use_cuda = use_cuda
        
        
    def forward(self,x):
        x = self.dropout1(F.relu(self.batchnorm1(self.layer1(x))))
        x = F.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout1(self.layer3(x))
        x = self.dropout1(F.relu(self.batchnorm3(self.layer4(x))))
        x = F.relu(self.batchnorm4(self.layer5(x)))
        x = self.dropout1(self.layer7(x))
        
        batch_size, channel,timesteps, freq = x.size()
        x = x.permute(0,3,2,1)
        reshape_size = channel*freq
        x = x.reshape(batch_size, timesteps, -1)
        
        
        h0 = Variable(torch.zeros(2, x.size(0), 32).requires_grad_())
        c0 = Variable(torch.zeros(2, x.size(0), 32).requires_grad_())
        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        out, (hn, cn) = self.rnn(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out