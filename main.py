from loadIm import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import math



parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-5)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

import numpy as np

# print(attr)

train_loader = load_train_images()
train_loader_valid = load_train_images_validation()
####### CNN

targetLearnSize = 1


#
class NetDis(nn.Module):

    def __init__(self):
        super(NetDis, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3380, targetLearnSize)
        self.active = nn.Hardtanh()

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = self.active(x)
        # b = torch.cuda.FloatTensor([0])
        # if x >= b :
        #     x = torch.cuda.Tensor([1])
        # else:
        #     x = torch.cuda.Tensor([0])

        b = torch.FloatTensor([0])
        b = b.cuda(GPU)
        x = torch.ge(x,b).float()
        return x  # F.log_softmax(x)


class NetCont(nn.Module):

    def __init__(self):
        super(NetCont, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3380, targetLearnSize)
        self.active = nn.Hardtanh()


    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = (self.active(x) + 1)/2
        return x  # F.log_softmax(x)


#
#
# class InceptionA(nn.Module):
#
#     def __init__(self, in_channels):
#         super(InceptionA, self).__init__()
#         self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#
#         self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
#
#         self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
#         self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
#
#         self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
#
#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#
#         branch5x5 = self.branch5x5_1(x)
#         branch5x5 = self.branch5x5_2(branch5x5)
#
#         branch3x3dbl = self.branch3x3dbl_1(x)
#         branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
#         branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
#
#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)
#
#         outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
#         return torch.cat(outputs, 1)
#
#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
#
#         self.incept1 = InceptionA(in_channels=10)
#         self.incept2 = InceptionA(in_channels=20)
#
#         self.mp = nn.MaxPool2d(2)
#         self.fc = nn.Linear(14872, targetLearnSize)
#
#     def forward(self, x):
#         in_size = x.size(0)
#         x = F.relu(self.mp(self.conv1(x)))
#         x = self.incept1(x)
#         x = F.relu(self.mp(self.conv2(x)))
#         x = self.incept2(x)
#         x = x.view(in_size, -1)  # flatten the tensor
#         x = self.fc(x)
#         return x#F.log_softmax(x)
#

# class AttributeNetwork(nn.Module):
#     """docstring for RelationNetwork"""
#     def __init__(self,input_size,hidden_size,output_size):
#         super(AttributeNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size,hidden_size)
#         self.fc2 = nn.Linear(hidden_size,output_size)
#
#     def forward(self,x):
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#
#         return x
#
# model = AttributeNetwork(64,1200,targetLearnSize)
modelsDis = [NetDis() for _ in range(7)]
for modelDis in modelsDis:
    modelDis.cuda(GPU)

modelsCont = [NetCont() for _ in range(19)]
for modelCont in modelsCont:
    modelCont.cuda(GPU)

optimizersDis = [torch.optim.Adam(modelsDis[i].parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay = 0.01) for i in range(7)]


optimizersCont = [torch.optim.Adam(modelsCont[i].parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay = 0.01) for i in range(19)]


def train(epoch):
    print("train started")
    global train_loader
    for i in range(7):
        modelsDis[i].train()
    for i in range(19):
        modelsCont[i].train()
        # try:
    torch.set_num_threads(4)
    for batch_idx, data in enumerate(train_loader):
        # print("in train loop   ",batch_idx)
        # data, target = Variable(data), Variable(target)
        data = Variable(data).cuda(GPU)
        criterionDis = [nn.MSELoss().cuda(GPU) for _ in range(7)]
        criterionCont = [nn.MSELoss().cuda(GPU) for _ in range(19)]


        for i in range(7):
            optimizersDis[i].zero_grad()
            output = modelsDis[i](data)
            # print(output)
            try:
                m = targetDis[batch_idx * 64:(batch_idx + 1) * 64]
                loss = criterionDis[i](output, Variable(m[:, i].unsqueeze(0).view(-1,1)).cuda(GPU).float())
            except:
                m = targetDis[batch_idx * 64:(batch_idx + 1) * 64]
                loss = criterionDis[i](output, Variable(m[:, i].unsqueeze(0).view(-1,1)).cuda(GPU).float())

            loss = Variable(loss, requires_grad=True)
            if loss <= 0.01:
                continue
            loss.backward()
            # print(model.conv1.bias.grad)

            optimizersDis[i].step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        for i in range(19):
            optimizersCont[i].zero_grad()
            output = modelsCont[i](data)
            # print(output)
            try:
                m = targetCont[batch_idx * 64:(batch_idx + 1) * 64]
                loss = criterionCont[i](output, Variable(m[:, i].unsqueeze(0).view(-1, 1)).cuda(GPU).float())
            except:
                m = targetCont[batch_idx * 64:(batch_idx + 1) * 64]
                loss = criterionCont[i](output, Variable(m[:, i].unsqueeze(0).view(-1, 1)).cuda(GPU).float())
            if loss <= 0.01:
                continue
            loss.backward()
            # print(model.conv1.bias.grad)

            optimizersCont[i].step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
    # except:
    #     for img in train_loader.dataset:
    #         if img.shape != torch.Size([3,64,64]):
    #             ews = type(train_loader.dataset)
    #             idvv = train_loader.dataset.index(img)
    #             print(trainNames[train_loader.dataset.index(img)])


correctCounter = 0
sagCounter = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


def test():
    global correctCounter
    correctCounter = 0

    for batch_idx, data in enumerate(train_loader_valid):
        data = Variable(data).cuda(GPU)
        s = data.cpu().detach().numpy().shape[0]
        testOPDist = np.empty([s, 7])
        testOPCont = np.empty([s, 19])
        testOP = np.empty([s,30])
        for i in range(7):
            output = modelsDis[i](data)
            output = output.cpu().detach().numpy()
            testOPDist[:, i] = output.flatten()

        for i in range(19):
            output = modelsCont[i](data)
            output = output.cpu().detach().numpy()
            testOPCont[:, i] = output.flatten()
        print(batch_idx)
        counter = 0
        testOP = sagTosag1(testOPCont,testOPDist)

        for op in testOP:
            # m = atr_to_label(op)
            # print(m ,'::::' ,validNames[counter],"::::",train_pics_dict[validNames[counter]])
            testTarget = train_pics_dict[validNames[batch_idx*64 + counter]]
            if atr_to_label(op) == testTarget:
                correctCounter = correctCounter + 1
            sagId = 0
            for opp in op:
                if math.fabs(opp - train_pics_to_attr_dict[trainName][sagId]) < 0.1:
                    sagCounter[sagId] = sagCounter[sagId] +1
                    
                sagId = sagId + 1
            counter = counter + 1


torch.set_num_threads(4)
#
# for epoch in range(1,20):
#     train(epoch)
#     idx=0;
#     for modelDis in modelsDis:
#         torch.save(modelDis,"Save/modelDis"+str(idx)+".pt")
#         idx = idx+1
#     idx = 0;
#     for modelCont in modelsCont:
#         torch.save(modelCont, "Save/modelCont" + str(idx) + ".pt")
#         idx = idx + 1

# testId = 0
idx = 0
for modelDis in modelsDis:
    modelDis = torch.load("Save/modelDis"+str(idx)+".pt")
    idx = idx+1
idx = 0
for modelCont in modelsCont:
    modelCont = torch.load("Save/modelCont" + str(idx) + ".pt")

    idx = idx + 1

test()
for x in range(30) :
    print("this is correct num:  ",x, sagCounter[x], "Pers:", correctCounter / (len(validNames)))

print("this is correct num:", correctCounter, "Pers:", correctCounter / (len(validNames)))

print("alaki")
