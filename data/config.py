# config.py
import os.path

# gets home dir cross platform
#HOME = os.path.expanduser("~")
HOME = '/home/houyaozu/repo/'
#HOME = '/data/houyaozu'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

#textboxes
text = {
    'num_class':2,
    'min_dim':300,  #暂定输入图像大小为固定的300*300，同ssd一样
    'lr_steps': (40000, 80000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes':[30.0, 60.0, 114.0, 168.0, 222.0, 276.0],
    'max_sizes':[60.0, 114.0, 168.0, 222.0, 276.0, 330.0],
    'aspect_ratios':[2, 3, 5, 7, 10],
    'variance':[0.1, 0.1, 0.2, 0.2],
    'clip': True,
    'name': 'TEXT'
}
