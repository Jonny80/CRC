from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import os.path as path
import os
from PIL import Image
import PIL


current = os.getcwd()

data_dir = "/data/train"
left_frames = "/left_frames"
labels = "/valid/labels"

frames = []
labeled = []
dataset = []

for i in os.listdir(current + data_dir + left_frames):
    frames.append(i)

frames = sorted(frames)

for j in os.listdir(current + data_dir + labels):
    labeled.append(j)

labeled = sorted(labeled)

for a in range(0,len(frames)):
    dataset.append((Image.open(current+data_dir+left_frames+"/"+frames[a]),Image.open(current+data_dir+labels+"/"+labeled[a])))

