#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from torchsummary import summary
from thop import clever_format, profile

from nets.retinaface import RetinaFace
from utils.config import cfg_mnet

if __name__ == '__main__':
    #--------------------------------------------#
    #   需要使用device来指定网络在GPU还是CPU运行
    #--------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaFace(cfg_mnet).to(device)
    summary(model, input_size=(3, 640, 640))
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    #.......................................................#
    #flops * 2是因为profile没有将卷积作为两个operations
    #有些论文将卷积算乘法、加法两个operations。此时乘2
    #有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #--------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
