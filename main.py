from loadIm import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
from cython.parallel import parallel,prange

import numpy as np

#print(attr)

train_loader = load_train_images()
train_loader_valid = load_train_images_validation()
####### CNN

targetLearnSize = 1
#
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5 )
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3380, targetLearnSize)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x#F.log_softmax(x)

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
models = []
optimizers = []
for i in range(30):
    models[i] = Net()
    optimizers[i] = optim.SGD(models[i].parameters(), lr=0.01, momentum=0.5)

print("targetTest")
def train(epoch):

    print("train started")
    global train_loader
    for i in range(30):
        models[i].train()
        # try:
    torch.set_num_threads(4)
    with parallel(num_threads = 4):
        criterion=[]
        for batch_idx,data in enumerate(train_loader):
            #print("in train loop   ",batch_idx)
            #data, target = Variable(data), Variable(target)
            data = Variable(data)
            for i in range(30):
                optimizers[i].zero_grad()
                output = models[i](data)
                #print(output)
                criterion[i] = nn.MSELoss()
                try:
                    m = target[batch_idx*64:(batch_idx+1)*64]
                    loss = criterion[i](output, m[:,i])
                except:
                    m = target[batch_idx * 64:(batch_idx + 1) * 64]
                    loss = criterion[i](output, m[:,i])
                loss.backward()
                #print(model.conv1.bias.grad)

                optimizers[i].step()

                if(batch_idx% 10 == 0) :
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

testOP = []
def test():
    global correctCounter
    correctCounter = 0
    with parallel(num_threads=4):
        for batch_idx,data in enumerate(train_loader_valid):
            data = Variable(data)
            for i in range(30):
                output = models[i](data)
                output = output.detach().numpy()
                testOP[:,i] = output

            print(batch_idx)
            counter = 0
            for op in testOP:
                #m = atr_to_label(op)
                #print(m ,'::::' ,validNames[counter],"::::",train_pics_dict[validNames[counter]])
                if op == train_pics_dict[validNames[counter]]:
                    correctCounter = correctCounter+1
                counter = counter + 1

            
torch.set_num_threads(4)
with parallel(num_threads = 16):
    for epoch in range(1, 2 ):
        train(epoch)
test()
print("this is correct num:",correctCounter,"Pers:",correctCounter/(len(validNames)))

print("alaki")