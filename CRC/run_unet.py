from UNet import *
from crc_dataset import *
from label_to_img import *
from dice import *
import sys
import torch
from torch import nn
import losses

use_gpu = torch.cuda.is_available()

def main():
    img_size = 128
    epoch = 101
    batch_size = 1
    learning_rate = 0.1
    momentum = 0.9

    job = sys.argv[1]
    model = sys.argv[2]

    if job == 'train':
        train_data = crc_Dataset(img_size=img_size, job=job)
        train_loader = data.DataLoader(dataset = train_data, batch_size = batch_size, pin_memory=True, shuffle=True)

        try:
            generator = torch.load('model/'+ model)
            try:
                learning_rate = float(sys.argv[3])
            except:
                pass
        except:
            generator = UnetGenerator(3, 11, 64).cuda()# (3,3,64)#in_dim,out_dim,num_filter out dim = 4 oder 11
            print("new model generated")
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=momentum)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True)
        for ep in range(epoch):
            confusion_matrix = np.zeros(
                (11, 11), dtype=np.uint32)
            for batch_number,(input_batch, label_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                input_batch = Variable(input_batch).cuda(0)
                label_batch = Variable(label_batch).cuda(0)
                generated_batch = generator.forward(input_batch)
                loss = loss_function(generated_batch,label_batch)
                output_classes = generated_batch.data.cpu().numpy().argmax(axis=1)
                target_classes = label_batch.data.cpu().numpy()
                confusion_matrix += calculate_confusion_matrix_from_arrays(
                 output_classes, target_classes, 11)
                loss.backward()
                optimizer.step()
            confusion_matrix = confusion_matrix[1:, 1:]
            dices = {'dice_{}'.format(cls + 1): dice
                     for cls, dice in enumerate(calculate_dice(confusion_matrix))}
            print(dices)
            average_dices = np.mean(list(dices.values()))
            print(average_dices)
            #scheduler.step(0.41234)
            if ep % 10 == 0:
                torch.save(generator, 'model/'+model)
                print("model saved")
        torch.save(generator, 'model/'+model)

    if job == 'validate':
        batch_size = 1
        validate_data = crc_Dataset(img_size=img_size, job=job)
        validate_loader = data.DataLoader(dataset=validate_data, batch_size=batch_size, pin_memory=True, shuffle=False)
        try:
            generator = torch.load('model/'+model)
        except:
            print("Error: Model doesn't exist")
            exit()
        dice_sum = 0
        for batch_number, (input_batch, label_batch) in enumerate(validate_loader):
            input_batch = Variable(input_batch).cuda(0)
            generated_batch= generator.forward(input_batch)
            generated_out_img = label_to_img(generated_batch.cpu().data, img_size)
            generated_out_img.save("data/validate-result/img_{}_generated.png".format(batch_number))
        avg_dice = dice_sum / validate_loader.__len__()
        print("Avgerage dice distance, 0 means perfect:", avg_dice)

if __name__ == "__main__":
    main()
    pass
