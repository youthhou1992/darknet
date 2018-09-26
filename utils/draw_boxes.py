import cv2
import os

class Drawer(object):
    def __init__(self):
        pass

    def draw_boxes(self, img_path, save_path, scores, coords):

        img = cv2.imread(img_path)
        color = [0, 255, 0]
        for i, box in enumerate(coords):
            score = scores
            print(box)
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.line(img, (xmin, ymin), (xmin, ymax), color, 1)
            cv2.line(img, (xmin, ymin), (xmax, ymin), color, 1)
            cv2.line(img, (xmax, ymax), (xmin, ymax), color, 1)
            cv2.line(img, (xmax, ymax), (xmax, ymin), color, 1)
        _, img_name = os.path.split(img_path)
        result_path = os.path.join(save_path, img_name)
        cv2.imwrite(result_path, img)

if __name__ == '__main__':
    import numpy as np
    img_path = '/data/samples/ICDAR/Challenge1_Test_Task12_Images/img_47.jpg'
    save_path = './result/'
    scores = 0.98
    coords = [[70.72731, 265.03317, 112.81486, 277.40427],
                [26.18301, 264.98373, 63.19695, 276.98276],
                [20.986385, 218.75409, 76.966705, 229.84178],
                [70.98318, 217.4993, 109.74308, 229.6011],
                [134.5121, 26.944166, 161.07018, 51.54236],
                [112.36632, 201.34152, 180.11157, 214.22758],
                [136.13445, 200.35399, 163.24348, 211.60428],
                [171.31247, 251.9995, 185.41518, 258.83182],
                [25.187336, 246.76761, 48.547813, 255.9104],
                [35.196575, 217.65854, 69.52378, 226.93332],
                [112.7631, 205.61964, 115.46682, 206.48766]]
    coords = np.array(coords)
    drawer = Drawer()
    drawer.draw_boxes(img_path, save_path, scores, coords)