from loadIm import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

#print(attr)
torch.set_num_threads(16)
train_loader = load_train_images()
train_loader_valid = load_train_images_validation()
####### CNN

targetLearnSize = 30

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


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    print("train started")
    global train_loader
    model.train()
    # try:
    for batch_idx,data in enumerate(train_loader):
        #print("in train loop   ",batch_idx)
        #data, target = Variable(data), Variable(target)
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        criterion = nn.MSELoss()
        try:
            loss = criterion(output, target[batch_idx*64:(batch_idx+1)*64])
        except:
            loss = criterion(output, target[batch_idx * 64: ])
        loss.backward()
        #print(model.conv1.bias.grad)

        optimizer.step()

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


def test():
    global correctCounter
    correctCounter = 0
    for batch_idx,data in enumerate(train_loader_valid):
        data = Variable(data)
        output = model(data)
        output = output.detach().numpy()

        print(batch_idx)
        counter = 0
        for op in output:
            m = atr_to_label(op)
            #print(m ,'::::' ,validNames[counter],"::::",train_pics_dict[validNames[counter]])
            if m == train_pics_dict[validNames[counter]]:
                correctCounter = correctCounter+1
            counter = counter + 1
            
            

for epoch in range(1, 100):
    train(epoch)
test()
print("this is correct num:",correctCounter,"Pers:",correctCounter/(len(validNames)))

print("alaki")