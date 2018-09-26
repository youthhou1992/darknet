"""ICDAR Dataset Classes

Original author: youthhou
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

"""

import os
from PIL import Image
import cv2
import torch.utils.data as data
import torch
import numpy as np

def default_loader(path):
    '''加载图像
       input: image path
       output: image in numpy
    '''
    img = cv2.imread(path)
    img = np.array(img, np.float32)
    return img


def data_transform(img):
    '''图像处理
       resize: 300*300
    '''
    mean = [104, 117, 123]
    img = img - mean
    return cv2.resize(img, (300,300))

class ICDARData(data.Dataset):
    """ICDAR Detection Dataset Object

    input is image, target is gt_box

    Arguments:
        img_path
        txt_path
    """
    def __init__(self, root, img_path, txt_path, dataset = '', data_transforms = data_transform, loader = default_loader):
        img_dir = os.path.join(root, img_path)
        txt_dir = os.path.join(root, txt_path)
        self.img_list = []
        self.target_list = []
        #self.ids = list()
        for im in os.listdir(img_dir):
            self.img_list.append(os.path.join(img_dir, im))
            txt_name = 'gt_' + im[:-4] + '.txt'
            gt_boxs  = []
            #self.ids.append(os.path.join(txt_dir, txt_name))
            with open(os.path.join(txt_dir, txt_name), 'r') as f:
                for line in f.readlines():
                    x_min, y_min, x_max, y_max = line.strip().split(',')[:4]
                    gt_box = [int(x_min), int(y_min), int(x_max), int(y_max), 1]
                    gt_boxs.append(gt_box)
            self.target_list.append(gt_boxs)
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img, target, h, w = self.pull_item(item)
        return img, target

    def pull_item(self, item):
        img_name = self.img_list[item]
        # print(img_name)
        target = self.target_list[item]
        img = self.loader(img_name)
        if self.data_transforms:
            img = self.data_transforms(img)
        # to rgb
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.type(torch.FloatTensor)
        c, h, w = img.size()
        return img, target, h, w

if __name__ == '__main__':

    DATAROOT = '/data/samples/ICDAR'
    IMG_PATH = 'Challenge1_Training_Task12_Images'
    TXT_PATH = 'Challenge1_Training_Task1_GT'
    data = ICDARData(root = DATAROOT ,img_path = IMG_PATH, txt_path = TXT_PATH)
    print(data.pull_item(0))
