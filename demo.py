import os,sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from ssd import build_ssd

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    #build the net, load the pretrained weights
    net = build_ssd('test', 300, 21)
    net.load_weights('/data/houyaozu/ssd.pytorch/demo/weights/ssd300_mAP_77.43_v2.pth')
    image = cv2.imread('/data/houyaozu/ssd.pytorch/data/example.png', cv2.IMREAD_COLOR)
    #print(image)

    x = cv2.resize(image, (300, 300)).astype(np.float32) #调整大小
    x -= (104.0, 117.0, 123.0)#减去均值
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    #print(x)

    xx = Variable(x.unsqueeze(0))
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    #print(y)
