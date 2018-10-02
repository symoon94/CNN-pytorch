import torch
import torch.nn as nn 

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.optim as optim

batch_size=32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

trainset = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_box1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv_box2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2))
        self.conv_box3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2 , stride=2))
        self.fc_box1 = nn.Sequential(
            nn.Linear(288, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc_box2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.fc_box3 = nn.Sequential(
            nn.Linear(64, 10))

    def forward(self, x):
        x = self.conv_box1(x)  # 32 16 31 31
        x = self.conv_box2(x)  # 32 32 15 15
        x = self.conv_box3(x)  # 32 32 3 3
        x = x.view(x.size(0), -1)  # 32 288
        x = self.fc_box1(x)  # 32 128
        x = self.fc_box2(x)  # 32 64
        x = self.fc_box3(x)  # 32 10
        return x
        

cnn=CNN()

lossfunction = nn.CrossEntropyLoss()  #  Softmax & NLLLoss 2 in 1
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

epoch_num = 3

for epoch in range(epoch_num):
    for i, data in enumerate(trainloader, 0):
        x_data, y_data = data

        optimizer.zero_grad() # initialize parameters
        y_pred = cnn(x_data)
        loss = lossfunction(y_pred, y_data)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('epoch %s step [%s/%s] --> loss %.3f '%(epoch, i, len(trainloader),loss.item()))


torch.save(cnn.state_dict(),'./cnn.pth')

## Evaluation ##

total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        x_data, y_data = data

        y_pred = cnn(x_data)
        _, y_pred = torch.max(y_pred, 1)
        total += y_data.size(0)
        correct += (y_pred == y_data).sum().item()

print('Accuracy: %s'%(100*(correct/total)))


'''
## Visualization Session ##
trainset
'''