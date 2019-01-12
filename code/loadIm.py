import random
from os import walk
from os import path
from PIL import Image
from torchvision import datasets, transforms
import torch
from base_funcs import *
import cv2 as cv
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
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
            img = cv.imread(filepath)
            imgCV = cv.resize(img,(64,64))
            img = Image.fromarray(imgCV)
            # img = img.resize((64, 64))
            if imgCV.shape[2] != 3:
                img = img.convert('RGB')
            img_tensor = preprocess(img)
            if i <= int(trainPercentage * total):
                train.append(img_tensor)
                trainLabels.append(filename)
            elif len(validation) <= testSize:
                validation.append(img_tensor)
                validLabels.append(filename)
            else:
                break
            i = i + 1
    return train, trainLabels, validation, validLabels
trainImgs = []
trainNames = []
validImgs = []
validNames = []
(trainImgs, trainNames, validImgs, validNames) = LoadData("../data/DatasetB/train/",1, 0)

testImgs = []
testNames = []
dummy1 = []
dummy2 = []
(testImgs, testNames, dummy1, dummy2) = LoadData("../data/DatasetB/test/", 1, 0)
#
#
# trainImgs = torch.load('trainDataSet.pt')
#
# trainNames = torch.load('trianNames.pt')
print("done")


print(trainImgs[0].detach().numpy().shape)
initialize()
labels_to_atrs_dict = np.load('labels_to_atrs_dict.npy')[()]
tr = np.load('train_pics_dict.npy')[()]
train_pics_dict = tr
train_pics_to_attr_dict = np.load('train_pics_to_attr_dict.npy')[()]
readable_to_label_dict = np.load('readable_to_label_dict.npy')[()]
seen_readable_to_label_dict = np.load('seen_readable_to_label_dict.npy')[()]
unseen_readable_to_label_dict = np.load('unseen_readable_to_label_dict.npy')[()]
create_test_pics_list()

####################### train
attrList = []
final_label_list = []
for trainName in trainNames:
    attr = train_pics_to_attr_dict[trainName]
    labell = train_pics_dict[trainName]
    label_list = list(readable_to_label_dict.values())
    final_label = (-1 * np.ones(225)).tolist()
    final_label[label_list.index(labell)] = 1
    final_label_list.append(final_label)
    attrList.append(attr)

target = torch.FloatTensor(final_label_list)
print("kheili alaki")


def load_train_images(image_size=64, batch_size=16, root="./img/"):
    train_set = trainImgs

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    # train_loader.dataset.target = torch.FloatTensor([0,1])
    return train_loader


####################### validation
attrListValid = []
for validName in validNames:
    attr = train_pics_to_attr_dict[validName]
    attrListValid.append(attr)
targetValid = torch.FloatTensor(attrListValid)
print("kheili alaki")


def load_train_images_validation(image_size=64, batch_size=8, root="./img/"):
    train_set = validImgs
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    # train_loader.dataset.target = torch.FloatTensor([0,1])
    return train_loader


def test_loader(image_size=64, batch_size=16, root="./img/"):
    train_set = testImgs

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    # train_loader.dataset.target = torch.FloatTensor([0,1])
    return train_loader
