from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import numpy as np
import torch


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

        self.mapping2 = {
            torch.tensor([0, 0, 0, 255],dtype=torch.uint8): 0,  # background,
            torch.tensor([0, 255, 0, 255],dtype=torch.uint8): 1,  # instrument-shaft
            torch.tensor([0, 255, 255, 255],dtype=torch.uint8): 2,  # instrument-clasper
            torch.tensor([125, 255, 12, 255],dtype=torch.uint8): 3,  # instrument-wrist
            torch.tensor([255, 55, 0, 255],dtype=torch.uint8): 4,  # kidney-parenchyma
            torch.tensor([24, 55, 125, 255],dtype=torch.uint8): 5,  # covered-kidney
            torch.tensor([187, 155, 25, 255],dtype=torch.uint8): 6,  # thread
            torch.tensor([0, 255, 125, 255],dtype=torch.uint8): 7,  # clamps
            torch.tensor([255, 255, 125, 255],dtype=torch.uint8): 8,  # suturing-needle
            torch.tensor([123, 15, 175, 255],dtype=torch.uint8): 9,  # suction-instrument
            torch.tensor([124, 155, 5, 255],dtype=torch.uint8): 10  # small-intestine
        }

        self.mapping_labels = {
            0: "background", 1: "instrument-shaft", 2: "instrument-clasper", 3: "instrument-wrist",
            4: "kidney-parenchyma", 5: "covered-kidney", 6: "thread", 7: "clamps", 8: "suturing-needle",
            9: "suction-instrument", 10: "small intestines"
        }

    def __getitem__(self, index):

        # Get Image Files
        single_image_name = self.image_list[index]
        single_label_name = self.label_list[index]
        img_as_img = Image.open(single_image_name)
        label_as_img = Image.open(single_label_name)

        # Adjust Image Size
        old_size = img_as_img.size
        new_size = max(old_size)
        boarder_img_as_img = Image.new("RGB", [new_size, new_size])
        boarder_label_as_img = Image.new("RGBA", [new_size, new_size], (0, 0, 0))
        boarder_img_as_img.paste(img_as_img, (int((new_size - old_size[0]) / 2),
                                              int((new_size - old_size[1]) / 2)))
        boarder_label_as_img.paste(label_as_img, (int((new_size - old_size[0]) / 2),
                                                  int((new_size - old_size[1]) / 2)))
        img_as_img = boarder_img_as_img
        label_as_img = boarder_label_as_img

        '''
        if self.job == "train":

            #####################################Data Augmentation##################################
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
            #####################################Data Augmentation##################################
        
        
        '''
        label_as_img = label_as_img.resize((self.img_size, self.img_size))  # resize (PIL.Image)
        img_as_img = img_as_img.resize((self.img_size, self.img_size))  # resize (PIL.Image)
        width, height = label_as_img.size

        img_tensor = torch.from_numpy(np.array(img_as_img))        # Image to Tensor
        img_tensor = img_tensor.permute(2,0,1)

        label_tensor = torch.from_numpy(np.array(label_as_img))    # Label to Tensor
        label_tensor = label_tensor.permute(2,0,1)
        label_tensor = label_tensor.view(label_tensor.size(0), -1).permute(1,0)


        label_tensor = self.mask_to_class(label_tensor)


        return img_tensor, label_tensor

    def __len__(self):
        return self.data_len

    def mask_to_class(self,mask):
        for k in self.mapping2:
            mask[mask == k] = self.mapping2[k]
        print(mask)
        print(mask.shape)
        return mask
