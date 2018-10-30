import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from base_funcs import *
from loadIm import *
import os

# print(attr)
torch.set_num_threads(16)
train_loader = load_train_images()
train_loader_valid = load_train_images_validation()
####### CNN

targetLearnSize = 30

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

    def __init__(self, block, layers, num_classes=160):
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


# models = [ResNet(Bottleneck, [3, 8, 36, 3]) for _ in range(1)]
#





##############
"""
squeezenet
"""

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.1, num_classes=160):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.Tanh(),
            nn.MaxPool2d(3, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

# model = SqueezeNet(version=1.1)
# model.cuda()

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
    batchSize = 4
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
        label_list = list(seen_readable_to_label_dict.values())
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
        # print(output.shape)
        # for op in output:
        #     # m = atr_to_label(op)
        #     # print(m ,'::::' ,validNames[counter],"::::",train_pics_dict[validNames[counter]])
        #     #             print(newAttr_to_label(op))
        #     #             print(train_pics_dict[validNames[counter +  batch_idx*64]])
        #     #             print('test',op)
        #     #             print('real',train_pics_to_attr_dict[validNames[counter +  batch_idx*64]])
        #     mahi = simpleDap(op)
        #     ################### correct Attr
        #     # print(counter + batch_idx * batchS)
        #     picName = validNames[counter + batch_idx * batchS]
        #     sagId = 0
        #     # print(op.shape)
        #     for opp in op:
        #
        #         if math.fabs(opp - train_pics_to_attr_dict[picName][sagId]) < 0.1:
        #             correctAttrCounter[sagId] = correctAttrCounter[sagId] + 1
        #         sagId = sagId + 1
        #     # print(mahi)
        #     if mahi == train_pics_dict[picName]:
        #         correctCounter = correctCounter + 1
        #         print(mahi)
        #         # print(train_pics_dict[picName])
        #         summ = 0
        #         for s in op:
        #             summ = summ + s
        #     #                 print(op)
        #     #                 print(summ)

        # counter = counter + 1


print(train_loader)
test()
print("this is correct num:" ,correctCounter ,"Pers:" ,correctCounter /(len(validNames)))

print(train_loader)
for epoch in range(1, 10000):
    train(epoch)
    if(epoch %100 == 0):
        test()
        print("this is correct num:", correctCounter, "Pers:", correctCounter / (len(validNames)))
    torch.save(model, "Save/model "+ ".pt")
test()
print("this is correct num:" ,correctCounter ,"Pers:" ,correctCounter /(len(validNames)))

print("alaki")