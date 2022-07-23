import torch
import os
import time
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#首先需要根据自己的cuda版本下载对应的torch版本
#再依次运行
#pip install torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
#pip install torchcsprng-0.2.1+cu101-cp38-cp38-linux_x86_64.whl
#pip install torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl


print("是否使用GPU训练：{}".format(torch.cuda.is_available()))
if torch.cuda.is_available:
    print("GPU名称为：{}".format(torch.cuda.get_device_name()))

transform = transforms.Compose(
    [transforms.RandomResizedCrop(size=224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5864, 0.5179, 0.5050),
                          (0.3562, 0.3491, 0.3462))])

dataset_train = ImageFolder('data/train',transform=transform)
dataset_valid =ImageFolder('data/val',transform=transform)
train_iter_size = len(dataset_train)
val_data_size = len(dataset_valid)
print("训练数据集的长度为：{}".format(train_iter_size))
print("验证数据集的长度为：{}".format(val_data_size))
# print("测试数据集的长度为：{}".format(test_data_size))

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, drop_last=True)

device = '0' if torch.cuda.is_available() else 'cpu'

print(f"Training on device {device}.")

net = models.resnet152(pretrained=True)
# print(net)
in_features = net.fc.in_features
net.fc=nn.Sequential(nn.Linear(in_features, 64),
                     nn.Linear(64, 10))

# for param in net.parameters():
#     param.requires_grad = False

# 查看总参数及训练参数
total_params = sum(p.numel() for p in net.parameters())
print('总参数个数:{}'.format(total_params))
total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('需训练参数个数:{}'.format(total_trainable_params))

net=net.to(device) # 将模型迁移到gpu

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.fc.parameters(), lr=1e-2, weight_decay=1e-5,momentum=0.8)

def train(net, train_iter, test_iter, num_epochs, optimizer, loss):
    for epoch in range(num_epochs):
        # 训练过程
        net.train()  # 启用 BatchNormalization 和 Dropout
        train_l_sum, train_acc_sum, train_num = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # 计算准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            train_num += y.shape[0]
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch + 1, train_l_sum / train_num, train_acc_sum / train_num))

        # 测试过程
        if (epoch + 1) % 5 == 0:
            test_acc_sum, test_num = 0.0, 0
            with torch.no_grad():  # 不会求梯度、反向传播
                net.eval()  # 不启用 BatchNormalization 和 Dropout
                for X, y in test_iter:
                    X = X.cuda()
                    y = y.cuda()
                    test_acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                    test_num += y.shape[0]
                print('test acc %.3f' % (test_acc_sum / test_num))

            torch.save(net.module.state_dict(), "model/best_model.pth")

ss_time = time.time()

train(
      net,
      train_iter=train_loader,
      test_iter=val_loader,
      num_epochs=50,
      optimizer=optimizer,
      loss=loss
     )