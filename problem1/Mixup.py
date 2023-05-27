#!/usr/bin/env python
# coding: gbk


import matplotlib
matplotlib.use('SVG')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from trainer import trainer
from resnet import resnet18


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_class = 100
batch_size = 128
data_dir = '/home/leijingshi/cvpr/data/cifar100/'
param_dir = '/home/leijingshi/DL/mid/recognition/param/Mixup.pt'
figure_dir = '/home/leijingshi/DL/mid/recognition/figure/Mixup_'
num_epochs_train = 250
milestone = [100,200]
lr = 0.1

mean = [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
std = [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transform_test = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean, std)])

train_dataset = datasets.cifar.CIFAR100(root=data_dir, train=True, transform=transform_train, download=True)
test_dataset = datasets.cifar.CIFAR100(root=data_dir, train=False, transform=transform_test, download=True)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

net = resnet18(num_class)
optimizer = optim.SGD(net.parameters(),lr = lr,momentum=0.9,weight_decay = 1e-04)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestone,gamma=0.1)

trainer_ = trainer(train_iter,test_iter,net,optimizer,device,scheduler)
train_loss,test_loss,test_acc = trainer_.train_mix(num_epochs_train,'mixup',param_dir)

num = list(range(1,1+len(train_loss)))
plt.figure()
plt.plot(num,train_loss,label='train_set')
plt.plot(num,test_loss,label='test_set')
plt.xlabel('iterations')
plt.ylabel('CrossEntropyLoss')
plt.legend()
plt.savefig(figure_dir+'loss.jpg')

plt.figure()
plt.plot(num,test_acc)
plt.xlabel('iterations')
plt.ylabel('Accuracy')
plt.savefig(figure_dir+'accuracy.jpg')
