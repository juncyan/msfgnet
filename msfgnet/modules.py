import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBnReLU, ConvBn, DepthWiseConv2D, SeparableConvBNReLU
from .blocks import MSIF, LRFE

class CDFSF(nn.Module):
    #cross dimension features sifting fusion
    def __init__(self, in_c1, in_c2):
        super().__init__()
        dims = in_c1 + in_c2
        self.zip_channels = ConvBnReLU(dims, in_c2, 1)
        self.lfc = ConvBnReLU(in_c2, in_c2, 3, padding=1)
    
        self.sa = nn.Sequential(ConvBn(2, 1, 3,padding=1), nn.Sigmoid())

        self.outcbr = ConvBnReLU(in_c2, in_c2, 3, padding=1)
        
    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = torch.concat([x1, x2], 1)
        x = self.zip_channels(x)
        y = self.lfc(x)
        
        max_feature = torch.max(y, dim=1, keepdim=True).values
        mean_feature = torch.mean(y, dim=1, keepdim=True)
        
        att_feature = torch.concat([max_feature, mean_feature], dim=1)
        y = self.sa(att_feature)
        y = y * x
        y = self.outcbr(y)
        return y

class MRFE(nn.Module):
    def __init__(self):
        super().__init__()

        down_channels = [[64, 128], [128, 256], [256, 512]]
        lksizes = [7,7,7]

        self.down_sample_list = nn.ModuleList([
            self.down_sampling(channel[0], channel[1], lksizes[i])
            for i, channel in enumerate(down_channels)
        ])

    def down_sampling(self, in_channels, out_channels, lksize):
        modules = []
        modules.append(ConvBnReLU(in_channels, in_channels, 3, stride=2, padding=1))
        modules.append(LRFE(in_channels, in_channels, lksize))
        modules.append(MSIF(in_channels, 4*in_channels))
        modules.append(ConvBnReLU(in_channels, out_channels, 3, padding=1))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        for down_sample in self.down_sample_list:
            x = down_sample(x)
            short_cuts.append(x)
        return short_cuts