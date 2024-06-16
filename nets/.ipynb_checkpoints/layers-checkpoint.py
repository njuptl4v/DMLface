import torch
import torch.nn as nn
import torch.nn.functional as F



#---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#


def conv_bn(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
       # 在sequential里面可以直接磊神经层
def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

#---------------------------------------------------#
#   卷积块
#   Conv2D + BatchNormalization
#---------------------------------------------------#
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

#---------------------------------------------------#
#   多尺度加强感受野
#---------------------------------------------------#
class SPPF(nn.Module):
    def __init__(self):
        super(SPPF, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        y1 = self.maxpool1(x)
        y2 = self.maxpool2(y1)
        y3 = self.maxpool3(y2)
        out = torch.cat([x, y1, y2, y3], dim=1)
        return out

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1

        # 3x3卷积
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        # 利用两个3x3卷积替代5x5卷积
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        # 利用三个3x3卷积替代7x7卷积
        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        
        # 所有结果堆叠起来
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out



