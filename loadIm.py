import random
from os import walk
from os import path
from PIL import Image
from torchvision import datasets, transforms
import torch
from base_funcs import *
import params

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    normalize
])

def LoadData(dir, trainPercentage):
    train = []
    trainLabels = []
    validation = []
    validLabels = []
    i = 0
    for (dirpath, dirnames, filenames) in walk(dir):
        total = len(filenames)
        random.shuffle(filenames)
        for filename in filenames:
            filepath = path.join(dirpath, filename)
            img = Image.open(filepath)
            if img.layers != 3:
                continue
            img_tensor = preprocess(img)
            if(i <= int(trainPercentage * total)):
                train.append(img_tensor)
                trainLabels.append(filepath.split('/')[1])
            else:
                validation.append(img_tensor)
                validLabels.append(filepath.split('/')[1])
            i = i + 1
    return train, trainLabels, validation, validLabels


trainImgs = []
trainNames = []
validImgs =[]
validNames =[]
(trainImgs,trainNames,validImgs,validNames) = LoadData('train',0.7)

initialize()

####################### train
attrList = []
for trainName in trainNames:
   attr = train_pics_to_attr_dict[trainName]
   attrList.append(attr)
target = torch.FloatTensor(attrList)
print("kheili alaki")
def load_train_images(image_size=64, batch_size=64, root="./img/"):

    train_set = trainImgs

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    #train_loader.dataset.target = torch.FloatTensor([0,1])
    return train_loader

####################### validation
attrListValid = []
for validName in validNames:
   attr = train_pics_to_attr_dict[validName]
   attrListValid.append(attr)
targetValid = torch.FloatTensor(attrListValid)
print("kheili alaki")
def load_train_images_validation(image_size=64, batch_size=64, root="./img/"):
    train_set = validImgs
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    #train_loader.dataset.target = torch.FloatTensor([0,1])
    return train_loader

