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
    transforms.Resize(64),
    transforms.ToTensor(),
    normalize
])

def LoadData(dir, trainPercentage, testSize):
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
            if i <= int(trainPercentage * total):
                train.append(img_tensor)
                trainLabels.append(filepath.split('/')[1])
            elif len(validation) <= testSize:
                validation.append(img_tensor)
                validLabels.append(filepath.split('/')[1])
            else:
                break
            i = i + 1
    return train, trainLabels, validation, validLabels


trainImgs = []
trainNames = []
validImgs =[]
validNames =[]
(trainImgs,trainNames,validImgs,validNames) = LoadData('train',0,10000)

initialize()

####################### train

def sagTosag(target):
    npTarget = target.detach().numpy()
    ind_dis = np.array([0, 2, 3, 4, 5, 15, 17])
    ind_con = np.array([1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25])
    targetDis = npTarget[:,ind_dis]
    targetDis = torch.FloatTensor(targetDis)
    targetCont = npTarget[:, ind_con]
    targetCont = torch.FloatTensor(targetCont)
    return targetCont, targetDis

def sagTosag1(dataCon, dataDis):
    # dataCon = dataCon.detach().numpy()
    # dataDis = dataDis.detach().numpy()
    npTarget = np.zeros([len(dataCon), 30])
    ind_dis = np.array([0, 2, 3, 4, 5, 15, 17])
    ind_con = np.array([1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25])

    npTarget[:, ind_con] = dataCon
    npTarget[:, ind_dis] = dataDis

    return npTarget

attrList = []
for trainName in trainNames:
   attr = train_pics_to_attr_dict[trainName]
   attrList.append(attr)
target = torch.FloatTensor(attrList)

targetCont, targetDis = sagTosag(target)

targetnew = sagTosag1(targetCont, targetDis)
print(sum(sum(target.detach().numpy()[:,:26] - targetnew[:,:26])))

print("targetTest")


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

