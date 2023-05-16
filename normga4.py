# 22-7-28
# kli
# G D

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# MNF
class MF(nn.Module):
    def __init__(self,in_channel,out_channel,N) -> None:
        super(MF,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,(1,N),bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.delin = nn.Parameter(t.randn(out_channel,1,1))
        self.relin = nn.Parameter(t.randn(out_channel,1,1))

    def forward(self,x):
        x = self.relu(self.conv1(x))
        # 乘权重 这里是pair mul,自动广播
        x = x.mul(self.delin)
        # 构造r1矩阵
        xt = x.permute([0,1,3,2])
        x = (x * xt).mul(self.relin)

        return x

# AFF
class AFF(nn.Module):
    '''
    AFF
    '''
    def __init__(self, channels=16, r=8,up = False):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.up = up

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.upconv = nn.Conv2d(r,channels,1,bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        if(self.up):
            residual = self.upconv(residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class Fusion(nn.Module):
    def __init__(self,c,ks,p) -> None:
        super(Fusion,self).__init__()
        self.mf1 = MF(1,c,150)
        self.res = ResLayer(1,c,ks,p)
        self.aff = AFF(c,c)


    def forward(self,x):
        x1 = self.mf1(x)
        x2 = self.res(x)
        return self.aff(x2,x1)

class MultFusion(nn.Module):
    def __init__(self,big = 64,small = 32) -> None:
        super(MultFusion,self).__init__()
        self.Fusion1 = Fusion(small,5,2)
        self.Fusion2 = Fusion(big,3,1)
        self.aff = AFF(big,small,up=True)
        self.avi = nn.ReLU()

    def forward(self,x):
        x1 = self.Fusion1(x)
        x1 = self.avi(x1 + x)
        x2 = self.Fusion2(x)
        x2 = self.avi(x2 + x)

        fffo = self.aff(x2,x1)

        return fffo + x

# Den
class DenseLayer(nn.Sequential):
    def __init__(self,num_in_features,growth_rate,bn_size,drop_rate):
        super(DenseLayer,self).__init__()
        self.add_module("norm1",nn.BatchNorm2d(num_in_features))
        self.add_module("relu1",nn.ReLU(inplace=False))
        self.add_module("conv1",nn.Conv2d(num_in_features,bn_size*growth_rate,kernel_size=1,stride=1,bias=False))
        self.add_module("norm2",nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2",nn.ReLU(inplace=False))
        self.add_module("conv2",nn.Conv2d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False))
        self.drop_rate = drop_rate
    def forward(self,x):
        new_features = super(DenseLayer,self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,p = self.drop_rate)
        # dense
        return t.cat([x,new_features],1)

# 构造Dense块
class DenseBlock(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
        super(DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features+i*growth_rate,
                               growth_rate,
                               bn_size,drop_rate)
            self.add_module("denselayer%d" % (i+1),layer)

# 构造DenseNet
class DenseNet(nn.Module):
    def __init__(self,input_channel = 64,growth_rate = 8,block_config = (2,4,6),
    num_init_feature = 128,bn_size = 4,compression_rate = 0.5,
    drop_rate = 0):
        super(DenseNet,self).__init__()
        # first conv2d /4倍数
        self.features = nn.Sequential(OrderedDict([
            ("conv0",nn.Conv2d(input_channel,num_init_feature,kernel_size=5,stride=1,padding=2,bias=False)),
            ("norm0",nn.BatchNorm2d(num_init_feature)),
            ("relu0",nn.ReLU(inplace=False))
        ]))

        # Dense Block
        num_features = num_init_feature
        for i,num_layers in enumerate(block_config):
            block = DenseBlock(num_layers,num_features,bn_size,growth_rate,drop_rate)
            self.features.add_module("denseblock%d"%(i+1),block)
            num_features += num_layers*growth_rate

        # # final bn+Relu
        self.features.add_module("norm5",nn.BatchNorm2d(num_features))
        self.features.add_module("relu5",nn.ReLU(inplace=False))

    def forward(self,x):
        fea = self.features(x)

        # tfea = fea.permute([0,1,3,2])
    
        return fea


# Res
class ResLayer(nn.Module):
    def __init__(self,in_channel,out_channel,ks = 1,p = 0,downsample = False,s = 1):
        super(ResLayer,self).__init__()
        self.feature = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(in_channel,out_channel,kernel_size=ks,padding=p,bias=False,stride=s)),
            ("norm1",nn.BatchNorm2d(out_channel)),
            ("relu1",nn.LeakyReLU()),
            ("conv2",nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False,stride=1)),
            ("norm2",nn.BatchNorm2d(out_channel))
        ]))

        self.downsample = downsample
        if(downsample):
            self.ds = nn.Conv2d(in_channel,out_channel,1)

        self.relu = nn.LeakyReLU()

    def forward(self,x):
        residual = x
        new_features = self.feature(x)
        if self.downsample:
            residual = self.ds(residual)
            
        new_features += residual

        # relu
        return self.relu(new_features)

class ResBlock(nn.Module):
    def __init__(self,num_layers,in_channel,out_channel,ks = 1,p = 0,s = 1):
        super(ResBlock,self).__init__()
        simple_block = []
        for i in range(num_layers):
            if(i == 0):
                self.upfeature = ResLayer(in_channel,out_channel,ks,p,True,s)
            else:
                simple_block.append(('res'+str(i+1),ResLayer(out_channel,out_channel,ks,p,False,s)))
                
        self.downfeature = nn.Sequential(OrderedDict(simple_block))

    def forward(self,x):

        x = self.upfeature(x)
        x = self.downfeature(x)

        return x

class ResNet(nn.Module):
    def __init__(self,in_c = 224) -> None:
        super(ResNet,self).__init__()
        self.block1 = ResBlock(6,in_c,128)
        # self.pool1 = nn.AvgPool2d(3)
        self.block2 = ResBlock(4,128,64)
        self.block3 = ResBlock(2,64,1)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        tx = x.permute([0,1,3,2])
    
        return (x+tx)/2

# G
class Construct(nn.Module):
    def __init__(self) -> None:
        super(Construct,self).__init__()
        # self.gen = nn.Sequential(OrderedDict([
        #     ("Mix Layer",MixLayer()),
        #     ("Dense Net",DenseNet()), # 160 150 150
        #     ("ResNet",ResNet())
        # ]))
        self.mix = MultFusion()
        self.den = DenseNet()
        self.res = ResNet()

    def forward(self,x):
        dx = self.den(self.mix(x))
        # print(dx.shape)
        dx = dx + x

        return self.res(dx)



# G = Generator()
# D = Discriminator()
# inp = t.randn(2,1,150,150)
# fake = G(inp)
# print(D(fake),D(inp))
