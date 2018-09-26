import argparse
import torch
import os, cv2
import tb
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from utils import create_xml

'''
    use DetEval calculated recall, precision, F1 score
    https://perso.liris.cnrs.fr/christian.wolf/software/deteval/index.html
'''

parser = argparse.ArgumentParser(description= 'TextBoxes detection')
parser.add_argument('--trained_model', default= 'weights/tb_80000.pth',
                    type = str, help='trained model to use')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#if not os.path.exists(args.sa)
def eval_net(net, dataset, cuda, thresh):
    det_bbox = []
    det_img = []
    for image in os.listdir(dataset):
        det_img.append(image)
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        img = img[:, :, (2, 1, 0)]
        img_resized = cv2.resize(img, (300, 300))
        img = np.array(img_resized, np.float32)
        x = torch.from_numpy(img).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        y = net(x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        i = 0
        single_box = []
        while(detections[0, 1, i, 0] >= thresh):
            pt = (detections[0, 1, i, 1:]*scale).cpu().numpy()
            box = [int(p) for p in pt]
            box = create_xml.point2center(box)
            #box = [int(x) for x in box]
            box.append('0')
            box = create_xml.box2dict(box, 'modelType')
            single_box.append(box)
            i += 1
        det_bbox.append(single_box)
        #break
    return det_img, det_bbox

def deal_gt(img_list, txt_root):
    gt_bbox = []
    for img in img_list:
        img_id = img.split('.')[0]
        img_id = img_id.split('_')[1]
        txt_path = 'gt_img_' + img_id + '.txt'
        txt_path = os.path.join(txt_root, txt_path)
        single_box = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                box = line.strip().split(',')
                box = box[:4]
                box = [eval(x) for x in box]
                box = create_xml.point2center(box)
                box.append('1')
                box = create_xml.box2dict(box, 'offset')
                single_box.append(box)
            #print (gt_bbox)
            #break
        gt_bbox.append(single_box)
        #break
    return img_list, gt_bbox


def eval_model():
    #加载网络
    net = tb.build_tb('test')
    #加载模型
    net.load_state_dict(torch.load(args.trained_model))
    #eval 模式
    net.eval()
    #cuda, cudnn
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    #eval dataset
    img_root = '/data/samples/ICDAR/Challenge1_Test_Task12_Images'
    txt_root = '/data/samples/ICDAR/Challenge1_Test_Task1_GT'
    det_path = 'result/det.xml'
    gt_path = 'result/gt.xml'
    det_img, det_bbox = eval_net(net, img_root, args.cuda, thresh = args.visual_threshold)
    #gt_img = det_img
    gt_img, gt_bbox = deal_gt(det_img, txt_root)
    print(gt_bbox)
    print(det_bbox)
    xml_creator = create_xml.xmlCreator()
    xml_creator.create_xml(det_path, det_img, det_bbox)
    xml_creator.create_xml(gt_path, gt_img, gt_bbox)

if __name__ == '__main__':
    eval_model()
    ##############