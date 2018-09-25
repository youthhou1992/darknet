import os
from PIL import Image

def change_format(img_name):
    img = Image.open(img_name)
    #_, img_format = os.path.split(img_name)
    #print(img_format)
    if  '.gif' in img_name:
        print(img_name.replace('.gif', '.png'))
        #img.save(os.path.join(_, '.png'))
        img.save(img_name.replace('.gif', '.png'))
    else:
        pass

if __name__ == '__main__':
    img_path = '/data/samples/ICDAR/Challenge1_Test_Task12_Images'
    for image in os.listdir(img_path):
        #print(image)
        img_name = os.path.join(img_path, image)
        change_format(img_name)