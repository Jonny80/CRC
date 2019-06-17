from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import numpy as np


class crc_Dataset(Dataset):
    def __init__(self, img_size, job):
        self.img_size = img_size
        self.job = job
        self.transforms = transforms.ToTensor()
        job = job + "/"
        self.image_list = []
        self.label_list = []
        for ordner in listdir('data/' + job):
            files = listdir('data/' + job + ordner + '/left_frames')
            for bild in files:
                if bild.find('.png') != -1:
                    self.image_list.append('data/' + job + ordner + '/left_frames/' + bild)
                    self.label_list.append('data/' + job + ordner + '/labels/' + bild)
        self.data_len = len(self.image_list)

        self.mapping = {
            0:(0, 0, 0),           # background,
            1:(0, 255, 0),         # instrument-shaft
            2:(0, 255, 255),       # instrument-clasper
            3:(125, 255, 12),      # instrument-wrist
            4:(255, 55, 0),        # kidney-parenchyma
            5:(24, 55, 125),       # covered-kidney
            6:(187, 155, 25),      # thread
            7: (0, 255, 125),       # clamps
            8: (255, 255, 125),     # suturing-needle
            9: (123, 15, 175),      # suction-instrument
            10: (124, 155, 5)       # small-intestine
        }

        self.mapping_labels = {
            0: "background", 1: "instrument-shaft", 2: "instrument-clasper", 3: "instrument-wrist",
            4: "kidney-parenchyma", 5: "covered-kidney", 6: "thread", 7: "clamps", 8: "suturing-needle",
            9: "suction-instrument", 10: "small intestines"
        }

    def __getitem__(self, index):
        single_image_name = self.image_list[index]
        single_label_name = self.label_list[index]
        img_as_img = Image.open(single_image_name)
        label_as_img = Image.open(single_label_name)
        old_size = img_as_img.size
        new_size = max(old_size)
        boarder_img_as_img = Image.new("RGB", [new_size, new_size])
        boarder_label_as_img = Image.new("RGBA", [new_size, new_size], (0, 0, 0, 255))
        boarder_img_as_img.paste(img_as_img, (int((new_size - old_size[0]) / 2),
                                              int((new_size - old_size[1]) / 2)))
        boarder_label_as_img.paste(label_as_img, (int((new_size - old_size[0]) / 2),
                                                  int((new_size - old_size[1]) / 2)))
        img_as_img = boarder_img_as_img
        label_as_img = boarder_label_as_img
        if self.job == "train":
            if random.choice([True, False]):
                img_as_img = img_as_img.transpose(Image.FLIP_LEFT_RIGHT)
                label_as_img = label_as_img.transpose(Image.FLIP_LEFT_RIGHT)
            wi, hei = img_as_img.size
            wizoom = random.randint(-0.05 * wi, 0.05 * wi)
            heizoom = int(wizoom / wi * hei)
            newwi = wi - wizoom
            newhei = hei - heizoom
            left = (wi - newwi) / 2
            top = (hei - newhei) / 2
            right = (wi + newwi) / 2
            bottom = (hei + newhei) / 2
            img_as_img = img_as_img.crop((left, top, right, bottom))
            label_as_img = label_as_img.crop((left, top, right, bottom))
            degree = random.randint(0, 359)
            img_as_img = img_as_img.rotate(degree, expand=False)
            label_as_img = label_as_img.rotate(degree, expand=False)
            rotate_size = label_as_img.size[0]
            background_label_as_img = Image.new("RGBA", [rotate_size, rotate_size], (0, 0, 0, 255))
            background_label_as_img.paste(label_as_img, (0, 0), label_as_img)
            label_as_img = background_label_as_img
        label_as_img = label_as_img.resize((self.img_size, self.img_size))
        img_as_img = img_as_img.resize((self.img_size, self.img_size))
        img_tensor = self.transforms(img_as_img)
        width, height = label_as_img.size
        label_channel = np.zeros((height, width, 11), dtype=np.int)
        label_tensor = self.transforms(label_channel)
        return (img_tensor, label_tensor)

    def __len__(self):
        return self.data_len
