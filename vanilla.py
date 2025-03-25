import torch
import torch.nn.functional as F
import torch.nn as nn
    
class MnistModel(nn.Module):
    def __init__(self, device = 'cpu',minst=False):
        super().__init__()
        self.minst = minst
        self.device = device
        if self.minst:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=(7, 7)) 
        else:
            self.conv1 = nn.Conv2d(3, 6, 7)
        self.pool = nn.MaxPool2d(kernel_size=3)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear = nn.Linear(294,512)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.linear(x)
        x = self.fc(x)
        return x