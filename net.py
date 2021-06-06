import torchvision
import torch.nn as nn
import torch
import os
import sys
sys.path.append('..')
import numpy as np
from model.Dilated_Unet import UNet
from model.RDN import resnet50_
from torch.autograd import Variable


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class MWFA_net(nn.Module):
    def __init__(self, num_slice, inplanes, axial_encoder, coronal_encoder, sagittal_encoder, norm_layer=None):
        super(MWFA_net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.num_slice = num_slice # CT slice number
        self.inplanes = inplanes # 输入提取的特征通道数
        self._norm_layer = norm_layer
        self.axial_net = axial_encoder # 横截面encoder
        self.coronal_net = coronal_encoder # 冠状面encoder
        self.sagittal_net = sagittal_encoder # 矢状面encoder
        self.conv1 = nn.Conv2d(self.num_slice, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_channeL_align = nn.Conv2d(in_channels=256, out_channels=self.inplanes, kernel_size=3, padding=1) # 使横截面输出的通道数与输入的通道数相同
        self.upconv_last = nn.Sequential(upconv2x2(in_channels=inplanes, out_channels=self.inplanes),
                                         upconv2x2(in_channels=inplanes, out_channels=self.num_slice)) # 使加权后的特征进行上采样恢复到之前的size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        input_of_MWFA = self.maxpool(x) # 生成MWFA的输入

        x_axial = input_of_MWFA
        x_coronal = input_of_MWFA.permute(0, 3, 1, 2) # 转换方向
        x_sagittal = input_of_MWFA.permute(0, 2, 1, 3) # 转换方向
        x_before_att = self.conv_channeL_align(self.axial_net(x_axial))
        att_coronal = self.coronal_net(x_coronal) # 冠状面encoder
        att_sagittal = self.sagittal_net(x_sagittal) # 矢状面encoder
        mwfa_out = x_before_att + torch.sigmoid(att_coronal.permute(0, 2, 3, 1))*x_before_att + torch.sigmoid(att_sagittal.permute(0, 2, 1, 3))*x_before_att # 注意力机制加权
        x_out = self.upconv_last(mwfa_out) # 上采样恢复

        return torch.sigmoid(x_out)


# if __name__ == '__main__':
#     x = Variable(torch.FloatTensor(np.random.randint(1, 10, (2, 110, 512, 512))).cuda())
#     y = Variable(torch.FloatTensor(np.zeros((2, 110, 512, 512))).cuda())
#     axial_encoder = resnet50_(replace_stride_with_dilation=np.array([0, 0, 1])).cuda()
#     coronal_encoder = UNet(num_classes=128, in_channels=128, depth=5, dilation_rate=2).cuda()
#     sagittal_encoder = UNet(num_classes=128, in_channels=128, depth=5, dilation_rate=2).cuda()
#     Net = MWFA_net(num_slice=110, inplanes=64, axial_encoder=axial_encoder, coronal_encoder=coronal_encoder,
#                    sagittal_encoder=sagittal_encoder)
#
#     Net = Net.cuda()
#     loss = nn.BCELoss()
#     optimizer = torch.optim.Adam(Net.parameters(), lr=0.0001, weight_decay=0)
#     # for i in range(1000):
#     #     out = Net(x)
#     #     l = loss(out, y)
#     #     optimizer.zero_grad()
#     #     l.backward()
#     #     optimizer.step()
#     #     print('epoch {} loss {}'.format(i+1, l.item()))
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
#                                                            eta_min=0.0000001)
#     for epoch in range(10):
#         print()
#
#         scheduler.step()