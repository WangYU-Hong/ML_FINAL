# -*- coding: utf-8 -*-
"""ML_final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vS2J2eXp-iIFBZ9GwvYM_Z1PWniWXnWw
"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch

#Define a Convolutional Neural Network

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
num_classes = 2
num_epochs = 20
batch_size = 64
learning_rate = 0.005
device = 'cpu'
model = AlexNet(num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
# Define transforms
#data augumation https://www.cnblogs.com/zhangxiann/p/13570884.html
#https://blog.csdn.net/qq_45802081/article/details/120248050
ran = 5
batch_size = 16
transform=transforms.Compose([
    transforms.RandomResizedCrop(size=240),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(45),
    #transforms.ColorJitter(contrast=(1, 5)),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#64%//mean=[.5,.5,.5],std=[.5,.5,.5]#60
])
# 训练数据的 transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=240, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0,translate=(0.05,0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def cnn(train_data_path,test_data_path):
    train_accuracy = 0
    test_accuracy = 0
    # 测试数据的 transforms
    transform_test = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #train_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\train_noface'
    #test_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\test_noface'

    # Load training dataset
    train_dataset = ImageFolder(root=train_data_path, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load testing dataset
    test_dataset = ImageFolder(root=test_data_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = ('adults','children')
    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            train_accuracy = 100 * correct / total
            print('Accuracy of the network on the {}/{} training images: {} %'.format(correct,total, 100 * correct / total))

        with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    del images, labels, outputs
                test_accuracy = 100 * correct / total
                print('Accuracy of the network on the {}/{} testing images: {} %'.format(correct,total, 100 * correct / total))  
    return train_accuracy , test_accuracy
train_accuracy_f , test_accuracy_f = cnn(r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\face_detect\train_face',r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\face_detect\test_face')
train_accuracy_nf , test_accuracy_nf = cnn(r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\face_detect\train_noface',r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\face_detect\test_noface')
print("final result : face detect({} % , {} % ) , no face detect({} % , {} % )".format(train_accuracy_f , test_accuracy_f,train_accuracy_nf , test_accuracy_nf))