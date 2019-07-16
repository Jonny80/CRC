from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import label_to_img
from crc_dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import label_to_img

image = Image.open("data/train/labels/labels/frame000.png")


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

def getSingleLabel(label, image):
    label = mapping_labels[label]  # Get Label Number
    label_value = reversed_mapping[label]  # Get Label Pixelvalue

    height, width = image.size  # Get Imagesize

    for i in range(height):  # Change everthing exept Label to background
        for j in range(width):
            value = image.getpixel((i, j))
            if value == label_value:
                pass
            else:
                image.putpixel((i,j),(0,0,0,255))

    return image


def showImage(pic):
    im_array = np.asarray(pic)
    plt.imshow(pic)
    plt.show()



def getMultipleLabels(labels,image):

    labellist = []

    for i in labels:
        labellist.append(reversed_mapping[mapping_labels[i]])


    height,width = image.size

    set_pixel = True

    for i in range(height):
        for j in range(width):
            value = image.getpixel((i,j))

            for k in labellist:

                if k == value:
                    set_pixel = False


            if set_pixel == False:
                set_pixel = True
                pass
            else:
                image.putpixel((i,j),(0,0,0,255))


    return image

#image2 = getMultipleLabels(("instrument-shaft","instrument-clasper"),image)
#showImage(image2)



data_train = crc_Dataset(img_size=128,job="validate")
data_loader= DataLoader(dataset=data_train,batch_size=1,pin_memory=True,shuffle=False)
for i in range(10):
    try:
        generator = torch.load('model/' + "myNewModel.pkl")
    except:
        print("Error: Model doesn't exist")
        exit()

    for batch_number, (input_batch,label_batch) in enumerate(data_loader):
        input_batch = Variable(input_batch).cuda(0)
        generated_batch = generator.forward(input_batch)
        picture = label_to_img.tensor_to_img(generated_batch.cpu().data,128)
        picture.save("data/validate-result/img_{}_generated.png".format(batch_number))
        print("done.")


old_size = image.size