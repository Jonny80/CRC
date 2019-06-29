from UNet import *
from crc_dataset import *
from label_to_img import *
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
import random
import time
import numpy

mapping = {
    0: (0, 0, 0, 255),  # background,
    1: (0, 255, 0, 255),  # instrument-shaft
    2: (0, 255, 255, 255),  # instrument-clasper
    3: (125, 255, 12, 255),  # instrument-wrist
    4: (255, 55, 0, 255),  # kidney-parenchyma
    5: (24, 55, 125, 255),  # covered-kidney
    6: (187, 155, 25, 255),  # thread
    7: (0, 255, 125, 255),  # clamps
    8: (255, 255, 125, 255),  # suturing-needle
    9: (123, 15, 175, 255),  # suction-instrument
    10: (124, 155, 5, 255)  # small-intestine
}


def label_to_img(batch, img_size):  # [4, 11, 128, 128]
    height = img_size
    width = img_size
    element = batch[0].numpy()
    out_img = Image.new('RGBA', (width, height), (0, 0, 0, 255))
    for row in range(height):
        for col in range(width):
            max_channel = 0
            for channel in range(11):
                if element[channel, row, col] > element[max_channel, row, col]:
                    max_channel = channel
            if max_channel == 0:
                pixel = (0, 0, 0, 255)
            elif max_channel == 1:
                pixel = (0, 255, 0, 255)
            elif max_channel == 2:
                pixel = (0, 255, 255, 255)
            elif max_channel == 3:
                pixel = (125, 255, 12, 255)
            elif max_channel == 4:
                pixel = (255, 55, 0, 255)
            elif max_channel == 5:
                pixel = (24, 55, 125, 255)
            elif max_channel == 6:
                pixel = (187, 155, 25, 255)
            elif max_channel == 7:
                pixel = (0, 255, 125, 255)
            elif max_channel == 8:
                pixel = (255, 255, 125, 255)
            elif max_channel == 9:
                pixel = (123, 15, 175, 255)
            elif max_channel == 10:
                pixel = (124, 155, 5, 255)
            out_img.putpixel((col, row), pixel)
    # todo out_img centercrop to original shape
    # todo scale up and paste on original image
    '''centercrop
            width, height = label_as_img.size
            left = (width - self.img_size) / 2
            top = (height - new_height) / 2
            right = (width + self.img_size) / 2
            bottom = (height + new_height) / 2
            label_as_img = label_as_img.crop((left, top, right, bottom))'''
    return out_img


def tensor_to_img(batch, img_size):
    element = batch[0].numpy()  # Get Tensor
    out_img = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 255))  # Initialize Image

    for i in range(img_size):
        for j in range(img_size):
            value = mapping[element[i][j]]  # Class to RGBA
            out_img.putpixel((i, j), value)  # Set Pixel Value

    return out_img
