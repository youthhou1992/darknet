from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from tb import build_tb
from data import text as cfg
import cv2
import numpy as np
import draw_boxes

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/tb_80000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./result/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
#parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

#将边界框保存到对应的txt文件中
def test_net(save_folder, net, cuda, testset, thresh):
    drawer = draw_boxes.Drawer()
    for image in os.listdir(testset):
        img_path = os.path.join(testset, image)
        print(img_path)
        img = cv2.imread(img_path)
        # to rgb
        img = img[:, :, (2, 1, 0)]
        img_resized = cv2.resize(img, (300, 300))
        img = np.array(img_resized, np.float32)
        x = torch.from_numpy(img).permute(2,0,1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        y = net(x)
        #print(y)
        detections = y.data
        #print(detections.size())
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        # print(img.shape)
        # print('scale:', scale)
        boxes = []
        scores = []
        for i in range(detections.size(1)):
            j = 0
            while(detections[0,i,j,0] >= thresh):
                score = detections[0, i, j, 0]
                #label_name = labelmap[i-1]
                pt =(detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = [pt[0], pt[1], pt[2], pt[3]]
                #print(score.data, coords)
                boxes.append(coords)
                scores.append(score)
                j += 1
                #break
        drawer.draw_boxes(img_path, save_folder, scores, boxes)
        break


def test_voc():
    # load net
    num_classes = cfg['num_class'] # +1 background
    net = build_tb('test') # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    #testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    #testset = '/data/samples/ICDAR/Challenge1_Test_Task12_Images/'
    testset = '/data/samples/ICDAR/Challenge1_Training_Task12_Images/'
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
