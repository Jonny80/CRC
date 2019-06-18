from UNet import *
from crc_dataset import *
from label_to_img import *
from dice import *
import sys
import torch
from loss import *

img_size = 128
batch_size = 1
learning_rate = 0.1
momentum = 0.9



train_data = crc_Dataset(img_size=img_size, job="train")
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, pin_memory=True, shuffle=True)


def test():
    train_data = crc_Dataset(img_size=img_size, job="train")
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, pin_memory=True, shuffle=True)
    generator = UnetGenerator(3, 11, 64).cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True)
    dice_sum = 0
    for batch_number, (input_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        input_batch = Variable(input_batch).cuda(0)
        label_batch = Variable(label_batch).cuda(0)
        generated_batch = generator.forward(input_batch)
        print(input_batch.size(), label_batch.size())

        loss = loss_function(generated_batch, label_batch)

        dice = dice_loss(generated_batch, label_batch.cuda()).item()
        dice_sum += dice
        loss.backward()
        optimizer.step()
    avg_dice = dice_sum / train_loader.__len__()
    print(avg_dice)
    print("done.")

test()