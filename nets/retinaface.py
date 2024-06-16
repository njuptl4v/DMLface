import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models


from nets.layers import SSH, SPPF, Dynamic_conv2d, MLCA, ADown, DCNv3_pytorch
from nets.mobilenet025 import MobileNetV1

def conv_bn(inp, oup, stride, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
       # 在sequential里面可以直接磊神经层
def conv_bn1X1(inp, oup, stride, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv_bn1X1(in_filters, filters_list[0], stride=1),
        conv_bn(filters_list[0], filters_list[1], stride=1),
        conv_bn1X1(filters_list[1], filters_list[0], stride=1),
    )
    return m

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv_bn1X1(in_channels, out_channels, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x
#---------------------------------------------------#
#   种类预测（是否包含人脸）
#---------------------------------------------------#
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

#---------------------------------------------------#
#   预测框预测
#---------------------------------------------------#
class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

#---------------------------------------------------#
#   人脸关键点预测
#---------------------------------------------------#
class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg=None, pretrained=False, mode='train'):
        super(RetinaFace, self).__init__()
        backbone = None
        #-------------------------------------------#
        #   选择使用mobilenet0.25、resnet50作为主干
        #-------------------------------------------#
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pretrained)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        #-------------------------------------------#
        #   两次卷积模块加SPPF模块,backbone第三层输出是256通道,64,128,256
        #-------------------------------------------#
        self.conv1 = make_three_conv([128, 256], 256)   #输出128
        self.sppf = SPPF()  #输出512
        self.conv2 = make_three_conv([128, 256], 512)   #第三层输出为20*20*128
        #-------------------------------------------#
        #     获得每个初步有效特征层的通道数
        #-------------------------------------------#
        #-------------------------------------------#
        #   利用初步有效特征层构建特征金字塔
        #-------------------------------------------#
        self.upsample1 = Upsample(128, 64)  #fpn的最高层上采样，40*40*64
        self.conv_for_P4 = conv_bn1X1(128, 64, stride=1,)    #fpn中间层的改变通道数，40*40*64
        self.conv3 = make_three_conv([64, 128], 64)  #两层特征融合，变128通道，做3此卷积加深变40*40*64
        
        self.upsample2 = Upsample(64, 32)    #fpn中间层上采样，80*80*32
        self.conv_for_P3 = conv_bn1X1(64, 32, stride=1)   #最底层改变通道数，80*80*32
        self.conv4 = make_three_conv([64, 128], 32)   #最底层和中间层融合，变64通道，做3此卷积加深变80*80*32
        self.dynamic1 = conv_bn(64, 32, stride=1)# 
        
        self.fpn_head1 = conv_bn1X1(32, 64, stride=1)  #输出最底层80*80*64
        
        self.adown1 = ADown(64, 64)   #conv4做步长为2的卷积相当于下采样，40*40*64
        self.conv5 = make_three_conv([64, 128], 128)     #64和conv3的64融合，40*40*128，接着3此卷积加深，变40*40*64
        self.dynamic2 = conv_bn(64, 64, stride=1)
        
        self.fpn_head2 = conv_bn1X1(64, 64, stride=1)  #输出中间层40*40*64
        
        self.adown2 = ADown(128, 128)    #conv5上采样，20*20*128
        self.conv6 = make_three_conv([128, 256], 256)    #conv5和cpnv2融合变256通道，3此卷积融合变128通道20*20*128
        self.dynamic3 = conv_bn(128, 128, stride=1)
        
        self.fpn_head3 = conv_bn1X1(128, 64, stride=1)  #输出最高层20*20*64
        #-------------------------------------------#
        #   利用ssh模块提高模型感受野
        #-------------------------------------------#
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.mlca = MLCA(80)
        self.dcnv3 = make_three_conv([64, 128], 64)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

        self.mode = mode

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  80, 80, 64
        #         C4  40, 40, 128
        #         C5  20, 20, 256
        #-------------------------------------------#
        out = self.body.forward(inputs)
        out = list(out.values())
        x0, x1, x2 = out[0], out[1], out[2]
        p2 = self.conv1(x2)  #最高层x2卷积
        p2 = self.sppf(p2)   #然后spp池化
        p2 = self.conv2(p2)  # 接着卷积，输出20*20*128
        p2_upsample = self.upsample1(p2)    # 20*20*128/40*40*64
        p1 = self.conv_for_P4(x1)    #40*40*64
        p1 = p1 + p2_upsample   #40*40*64，第一层和第二层特征图相加
        p_1 = self.conv3(p1)     #40*40*64,fpn第二层输出
        p1_upsample = self.upsample2(p1)   #80*80*32
        p0 = self.conv_for_P3(x0)   #80*80*32
        p0 = p0 + p1_upsample   #80*80*32，第二层和第三层特征图相加
        c0 = self.conv4(p0)   #80*80*64,一会输出用
        p0_downsample = self.adown1(c0)  #40*40*64，pan最下面一层
        p3 = torch.cat([p0_downsample, p_1], axis=1)  #40*40*128，pan最下面一层和中间层的cat
        c1 = self.conv5(p3)   #40*40*64,一会输出用
        p1_downsample = self.adown2(p3)   #20*20*128
        p4 = torch.cat([p1_downsample, p2], axis=1)  #20*20*256
        c2 = self.conv6(p4)    #20*20*128,一会输出用
        c2_ = self.dynamic3(c2)   #20*20*128经过动态3*3变为20*20*128
        c1_ = self.dynamic2(c1)   #40*40*64经过动态3*3变成40*40*64
        c0_ = self.dynamic1(c0)   #80*80*64经过动态3*3变成80*80*32
        #-------------------------------------------#
        #  最高层特征层20*20*128
        #   中间特征曾40*40*64
        #   最底层特征80*80*32
        #-------------------------------------------#
        out2 = self.fpn_head3(c2_)  #128变64，20*20*64
        out1 = self.fpn_head2(c1_)  #64变64，40*40*64
        out0 = self.fpn_head1(c0_)   #32变64，80*80*64
        #输出20*20*64  40*40*64  80*80*64(c0)

        outmlca = self.mlca(out0)
        f1 = self.ssh1(outmlca)
        f2 = self.ssh2(out1)
        f3 = self.ssh3(out2)
        feature1 = self.dcnv3(f1)
        feature2 = self.dcnv3(f2)
        feature3 = self.dcnv3(f3)
        features = [feature1, feature2, feature3]

        #-------------------------------------------#
        #   将所有结果进行堆叠
        #-------------------------------------------#
        bbox_regressions    = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications     = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions     = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.mode == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
