from xml.etree import ElementTree as ET
from xml.dom import minidom as md
import os

class xmlCreator(object):
    def __init__(self):
        pass

    def write_xml(self, tagset, save_path):
        rough_string = ET.tostring(tagset, 'utf-8')
        reared_content = md.parseString(rough_string)
        with open(save_path, 'w') as f:
            reared_content.writexml(f, addindent='\t', newl='\n', encoding='UTF-8')

    def create_xml(self, save_path, image_list, bbox_list):
        tagset = ET.Element('tagset')
        for i, img in enumerate(image_list):
            image = ET.SubElement(tagset, 'image')
            imageName = ET.SubElement(image, 'imageName')
            imageName.text = img
            taggedRectangles = ET.SubElement(image, 'taggedRectangles')
            for bbox in bbox_list:
                taggedRectangle = ET.SubElement(taggedRectangles, 'taggedRectangle')
                taggedRectangle.attrib = bbox
        self.write_xml(tagset, save_path)

def point2center(box):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    x = (xmax - xmin)/2 + xmin
    y = (ymax - ymin)/2 + ymin
    width = xmax - xmin
    height = ymax - ymin
    return [str(x), str(y), str(width), str(height)]


def box2dict(box, ext):
    bbox = {}
    bbox['x'] = box[0]
    bbox['y'] = box[1]
    bbox['width'] = box[2]
    bbox['height'] = box[3]
    bbox[ext] = box[4]
    return bbox


if __name__ == '__main__':
    gt_path = './gt.xml'
    det_path = './det.xml'
    xml_creator = xmlCreator()
    gt_txt = '/data/samples/ICDAR/Challenge1_Test_Task1_GT'
    gt_img = []
    gt_bbox = []
    for txt in os.listdir(gt_txt):
        img_id = txt.split('.')[0]
        img_id = img_id.split('_')[-1]
        gt_img.append(img_id)
        txt_path = os.path.join(gt_txt, txt)
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                box = line.strip().split(',')
                box = box[:4]
                box = [eval(x) for x in box]
                box = point2center(box)
                box.append('1')
                box = box2dict(box, 'offset')
                gt_bbox.append(box)
            print (gt_bbox)
            #break
        break

    #gt_img, gt_bbox =
    #det_img, det_bbox =
    #生成gt.xml
    xml_creator.create_xml(gt_path, gt_img, gt_bbox)
    #生成det.xml
    #xml_creator.create_xml(det_path, det_img, det_bbox)
