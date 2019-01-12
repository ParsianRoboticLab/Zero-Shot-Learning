import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import gan
gan.main()
from base_funcs import *
from loadIm import *

# print(attr)
torch.set_num_threads(16)
train_loader = load_train_images()
train_loader_valid = load_train_images_validation()
testLoader = test_loader()
####### CNN



final_res = {}





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=225):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.active = nn.Tanh()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.active(x)
        return x


#############
try:
    model = torch.load("Save/model.pt")
    model.cuda()
except:
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    # model = SqueezeNet(version=1.1)
    model.cuda()

# optimizer = torch.optim.Adam(model.parameters(),lr=0.01, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(),lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay = 0.01)


def train(epoch):
    batchSize = 16
    print("train started")
    global train_loader
    model.train()
    # try:
    for batch_idx ,data in enumerate(train_loader):
        # print("in train loop   ",batch_idx)
        # data, target = Variable(data), Variable(target)
        data = Variable(data).cuda()
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        criterion = nn.MSELoss()
        try:
            loss = criterion(output, Variable(target[batch_idx *batchSize:(batch_idx +1 ) *batchSize]).cuda())
        except:
            loss = criterion(output, Variable(target[batch_idx * batchSize: ]).cuda())
        loss.backward()
        # print(model.conv1.bias.grad)

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
correctAttrCounter = []
for i in range(50):
    correctAttrCounter.append(0)


def test():
    global correctCounter
    correctCounter = 0
    batchS = 8
    for batch_idx, data in enumerate(train_loader_valid):

        data = Variable(data).cuda()
        #        s = data.detach().numpy().shape[0]
        #       testOP = np.empty([s, 30])
        #         for i in range(len(models)):
        #             output = models[i](data)
        #             output = output.detach().numpy()
        #             testOP[:, i] = output.flatten()
        output = model(data).cpu().detach().numpy()
        label_list = list(readable_to_label_dict.values())
        for i in range(output.shape[0]):
            hot_vector = output[i ,:] / np.sum(output[i ,:])
            avg_sum = np.zeros(len(list(labels_to_atrs_dict.values())[0]))
            for j in range(len(hot_vector)):
                avg_sum += hot_vector[j] * labels_to_atrs_dict[label_list[j]]
            hot_lbl = simpleDap(avg_sum)
            # hot_lbl = label_list[np.argmax(output[i,:])]
            if train_pics_dict[validNames[ i +batch_idx *batchS]] == hot_lbl:
                correctCounter += 1

        print(batch_idx)
        counter = 0


print(train_loader)

print(train_loader)
for epoch in range(1, 20000000):
    train(epoch)
    torch.save(model, "Save/model "+ ".pt")



manifold = False

def createFinalRes():
    global final_res
    batchS = 16
    for batch_idx, data in enumerate(testLoader):
        data = Variable(data).cuda()
        s = data.cpu().detach().numpy().shape[0]
        testOPDist = np.empty([s, 7])
        testOPCont = np.empty([s, 17])
        testOP = np.empty([s, 30])

        data = Variable(data).cuda()
        output = model(data).cpu().detach().numpy()

        label_list = list(readable_to_label_dict.values())
        for i in range(output.shape[0]):
            hot_vector = output[i, :] / np.sum(output[i, :])
            avg_sum = np.zeros(len(list(labels_to_atrs_dict.values())[0]))
            for j in range(len(hot_vector)):
                avg_sum += hot_vector[j] * labels_to_atrs_dict[label_list[j]]
            hot_lbl = simpleDap(avg_sum)
            img_name = testNames[batch_idx * batchS + i]
            final_res[img_name] = hot_lbl
        print(batch_idx)
        counter = 0



#
createFinalRes()
import datetime
subname = "../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
f = open(subname, 'w+')
f.write('')
f = open(subname, 'a+')
for key, value in final_res.items():
    strrr = str(key) + '\t' + str(value) + '\n'
    f.write(strrr)
print("this is correct num:" ,correctCounter ,"Pers:" ,correctCounter /(len(validNames)))

print("alaki")
