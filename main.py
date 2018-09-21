from loadIm import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

#print(attr)

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
    model.train()
    for batch_idx,data in enumerate(train_loader):
        #data, target = Variable(data), Variable(target)
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        #print(output)
        criterion = nn.MSELoss()
        loss = criterion(output, target)
        loss.backward()
        #print(model.conv1.bias.grad)

        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    for batch_idx,data in enumerate(train_loader_valid):
        data = Variable(data)
        output = model(data)
        output = output.detach().numpy()

        print(batch_idx)
        for op in output:
           # m = np.array(op).reshape(-1, 1).transpose()
            print(atr_to_label(op.reshape(-1,1)[:,0]))

for epoch in range(1, 10):
    train(epoch)

test()

print("alaki")