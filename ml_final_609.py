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


# Define transforms
#data augumation https://www.cnblogs.com/zhangxiann/p/13570884.html
#https://blog.csdn.net/qq_45802081/article/details/120248050
ran = 8
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

# 测试数据的 transforms
transform_test = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define paths
train_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\train_face'
test_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\test_face'
#train_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\train_noface'
#test_data_path = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset_after_detect\test_noface'

# Load training dataset
train_dataset = ImageFolder(root=train_data_path, transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Load testing dataset
test_dataset = ImageFolder(root=test_data_path, transform=transform_test)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('adults','children')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
# show images
#imshow(torchvision.utils.make_grid(images))

# 2. Define a Convolutional Neural Network
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Adjust fc1 input size based on the output size after convolutions and pooling
        self.fc1_input_size = self._calculate_fc1_input_size((3, 240,240 ))
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def _calculate_fc1_input_size(self, input_size):
        # Calculate the size of the feature map after convolutions and pooling
        x = torch.randn(1, *input_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

net = Net()
# 3. Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
#lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


for epoch in range(ran):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print the average loss for the epoch
        print(f'Epoch [{epoch + 1}/{ran}], Loss: {running_loss / len(trainloader):.3f}')
        # Evaluate training accuracy
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                outputs = net(images)
                #outputs = outputs.reshape(-1)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        print(f'Training Accuracy: {train_accuracy:.2f}%  correct {correct_train}  total {total_train}')
print('Finished Training')

"""
with torch.no_grad():
        for data in trainloader:
                images, labels = data
                outputs = net(images)
                #outputs = outputs.reshape(-1)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        print(f'Training Accuracy: {train_accuracy:.2f}%  correct {correct_train}  total {total_train}')
print('Finished Training')
"""

# Evaluate testing accuracy after training

correct_test = 0
total_test = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
test_accuracy = 100 * correct_test / total_test
print(f'Testing Accuracy: {test_accuracy:.2f}% correct {correct_test}  total {total_test}')
#PATH = r'C:\Users\USER\Desktop\sophomore\machine learning\final project_111705010\dataset\dataset'
#torch.save(net.state_dict(), PATH)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

del dataiter