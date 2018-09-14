import os,sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(os.path.expanduser("~"))

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt
from ssd import build_ssd

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#build the net, load the pretrained weights
net = build_ssd('test', 300, 21)
net.load_weights('./demo/weights/ssd300_mAP_77.43_v2.pth')
image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(rgb_image)
plt.show()

