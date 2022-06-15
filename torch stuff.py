#this whole thing is just me following a tutorial on pytorch
#implementing an mnist model
#its not relevant to the rest of the project i just didnt want to create
#a new one

from numpy import argmax
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as f
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os

#there is some duplicate file in conda so i need to allow for it
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(torch.cuda.is_available())

# device = torch.device("gpu")

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fully_c_1 = nn.Linear(28*28, 64)
        self.fully_c_2 = nn.Linear(64, 64)
        self.fully_c_3 = nn.Linear(64, 64)
        self.fully_c_4 = nn.Linear(64, 10)

    def forward(self, data):
        data = f.relu(self.fully_c_1(data))
        data = f.relu(self.fully_c_2(data))
        data = f.relu(self.fully_c_3(data))

        return f.log_softmax(self.fully_c_4(data), dim=1)

net = Net().cuda()

optimizer = opt.Adam(net.parameters(), lr=1e-3)

epochs = 3

# print(torch.cuda.is_available())

for epoch in range(epochs):
    for data in trainset:
        features, labels = data
        features, labels = features.cuda(), labels.cuda()
        # print("f", features[0])
        # print("l", labels[0])
        net.zero_grad()
        out = net(features.view(-1, 28*28))
        loss = f.nll_loss(out, labels)
        print(loss)
        loss.backward()
        optimizer.step()
    
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        total += 10
        features, labels = data
        out = net(features.view(-1, 28*28))
        
        for i in range(out.size(dim=1)):
            if torch.argmax(out[i]) == labels[i]:
                correct += 1

print(f'ACCURACY: {(correct / total) * 100}%')

    










