import matplotlib.pyplot as plt
from crc_dataset import *
from torch.utils.data import DataLoader
import torch
import sys
import numpy as np

label = Image.open("data/train/labels/labels/frame000.png")


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

reversed_mapping = dict(map(reversed,mapping.items()))

mapping_labels = {
            "background": 0, "instrument-shaft": 1, "instrument-clasper": 2, "instrument-wrist": 3,
            "kidney-parenchyma": 4, "covered-kidney": 5, "thread": 6, "clamps": 7, "suturing-needle": 8,
            "suction-instrument": 9, "small intestines": 10
        }
def showImage(pic):
    pic = np.asarray(pic)
    print(pic.shape)
    plt.imshow(pic)
    plt.show()



data_train = crc_Dataset(img_size=128,job="validate")
data_loader= DataLoader(dataset=data_train,batch_size=1,pin_memory=True,shuffle=False)


label = label.resize((128,128))
