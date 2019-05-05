import torch
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

import torch.nn as nn

# Activates GPU for learning if available
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

# Equalizes Images and Transforms them to Tensors
transformation = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

# Import our Data (path has to be adjusted) probably wrong
train_data = datasets.ImageFolder("/home/jonny/Dokumente/Uni/Chirugie/Data/"
                                  "miccai_challenge_2018_release_1/seq_1/train/", transform=transformation)

val_data = datasets.ImageFolder("/home/jonny/Dokumente/Uni/Chirugie/Data/"
                                "miccai_challenge_2018_release_1/seq_1/valid/", transform=transformation)

# Import to the Dataloader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

val_loader = DataLoader(val_data, batch_size=16, shuffle=True)


# building Unet copied from https://towardsdatascience.com/u-net-b229b32b4a71
class unet(nn.Module):

    # Upwards
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )

        return block

    # Downwards
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    # Final block
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    # Connecting block

    def bottleneck_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                     stride=2, padding=1, output_padding=1)
        )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def __init__(self, in_channel, out_channel):
        super(unet, self).__init__()
        # contracting

        self.conv_encode = self.contracting_block(in_channel, out_channels=64)
        self.convmaxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        # connection
        self.bottleneck2 = self.bottleneck_block(128, 256)

        # expansive
        self.conv_decode2 = self.expansive_block(256, 128, 64)

        # final layer
        self.final = self.final_block(128, 64, out_channel)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer


# Initialize Unet
model = unet(in_channel=1, out_channel=11)

# Check for GPU
if torch.cuda.is_available():
    model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

# Criterion
criterion = nn.CrossEntropyLoss


def train(epoch):
    model.train()
    for (data, target) in enumerate(train_loader):
        # For GPU use
        # data = data.cuda()
        # target = target.cuda()

        data = Variable(data)
        target = Variable(data)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()


def go(times):
    for epoch in range(1, times):
        train(epoch)

