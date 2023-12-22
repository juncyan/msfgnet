import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ConvBn, ConvBnReLU
from .modules import CDFSF, MRFE
from .blocks import BMF, _PositionAttentionModule


class MSFGNet(nn.Module):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = MRFE()

        self.pam = _PositionAttentionModule(512)

        self.up1 = CDFSF(512, 512)
        self.up2 = CDFSF(512, 256)
        self.up3 = CDFSF(256, 128)
        self.up4 = CDFSF(128, 64)
        
        self.classier = nn.Sequential(ConvBn(64, num_classes, 7, 1,3), nn.Sigmoid())

    def forward(self, x1, x2):
        # x1 ,x2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.bmf(x1,x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)

