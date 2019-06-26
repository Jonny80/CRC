from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
from os import listdir
import matplotlib.pyplot as plt

# ===== test =====




mapping = {
            (0, 0, 0, 255): 0,  # background,
            (0, 255, 0, 255): 1,  # instrument-shaft
            (0, 255, 255, 255): 2,  # instrument-clasper
            (125, 255, 12, 255): 3,  # instrument-wrist
            (255, 55, 0, 255): 4,  # kidney-parenchyma
            (24, 55, 125, 255): 5,  # covered-kidney
            (187, 155, 25, 255): 6,  # thread
            (0, 255, 125, 255): 7,  # clamps
            (255, 255, 125, 255): 8,  # suturing-needle
            (123, 15, 175, 255): 9,  # suction-instrument
            (124, 155, 5, 255): 10  # small-intestine
        }

job2 = "train"

image_list = []

label_list = []

job = job2 + "/"

for ordner in listdir('data/' + job):
    files = listdir('data/' + job + ordner + '/left_frames')
    for bild in files:
        if bild.find('.png') != -1:
            image_list.append('data/' + job + ordner + '/left_frames/' + bild)
            label_list.append('data/' + job + ordner + '/labels/' + bild)

image = Image.open(image_list[0])


label = Image.open(label_list[1])


image = image.resize((256,256),resample=Image.BILINEAR)

label = label.resize((256,256),resample=Image.BILINEAR)


image = transforms.ToTensor()(image)
image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)



height , width = label.size

print(height)
print(width)

label_tensor = np.zeros((height,width,1), dtype = np.int)

errorList = []

for i in range(height):
    for j in range(width):
        value = label.getpixel((i,j))
        try:
            value = mapping[value]
        except:
            value = 0
            errorList.append(value)
        finally:
            label_tensor[i][j] = value