import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBnReLU, ConvBn, DepthWiseConv2D

class BMF(nn.Module):
    #Bitemporal Image Multi-level Fusion Module
    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        self.cbr1 = ConvBnReLU(in_channels, 32, 3, 1, 1)
        self.cbr2 = ConvBnReLU(in_channels, 32, 3, 1, 1)

        self.cond1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.cond3 = nn.Conv2d(64, 64, 3, 1, padding=3, dilation=3)
        self.cond5 = nn.Conv2d(64, 64, 3, 1, padding=5, dilation=5)

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.shift = ConvBnReLU(64, out_channels, 3, 2, 1)

    def forward(self, x1, x2):
        y1 = self.cbr1(x1)
        y2 = self.cbr2(x2)

        y = torch.concat([y1, y2], 1)

        y10 = self.cond1(y)
        y11 = self.cond3(y)
        y12 = self.cond5(y)
       
        yc = self.relu(self.bn(y10 + y11 + y12))

        return self.shift(yc)


class LRFE(nn.Module):
    #Large receptive field features extraction
    def __init__(self, in_channels, dw_channels, block_lk_size, stride=1):
        super().__init__()
        self.cbr1 = ConvBnReLU(in_channels, dw_channels, 3, stride=1, padding=1)
        
        self.dec = DepthWiseConv2D(dw_channels, block_lk_size, stride=stride)
        self.gelu = nn.GELU()
        self.c2 = nn.Conv2d(dw_channels, in_channels, 1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.cbr1(x)
        y = self.dec(y)
        y = self.gelu(y)
        y = self.c2(y)

        return self.relu(self.bn(x + y))

class MSIF(nn.Module):
    #multi-scale information fusion
    def __init__(self, in_channels, internal_channels):
        super().__init__()
        self.cbr1 = ConvBnReLU(in_channels, internal_channels, 1)

        self.cond1 = nn.Conv2d(internal_channels, internal_channels, 1,1)
        self.cond3 = nn.Conv2d(internal_channels, internal_channels, 3, 1, padding=3, dilation=3, groups=internal_channels)
        self.cond5 = nn.Conv2d(internal_channels, internal_channels, 3, 1, padding=5, dilation=5, groups=internal_channels)

        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.relu1 = nn.ReLU()

        self.cbr2 = ConvBnReLU(internal_channels, in_channels, 1)
        
        self.lastbn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        y = self.cbr1(x)
        y1 = self.cond1(y)
        y2 = self.cond3(y)
        y3 = self.cond5(y)
        y = self.relu1(self.bn1(y1 + y2 + y3))
        y = self.cbr2(y)
        return self.relu(self.lastbn(x + y))

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out