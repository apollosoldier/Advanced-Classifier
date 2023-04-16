import torch
import torch.nn as nn

class BaseClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the BaseClassifier object by defining the layers of the neural network.

        Input:
        @param - num_classes: the number of classes to be classified

        Output:
        @return - None
        """
        super(BaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Input:
        @param  - x: the input data

        Output:
        @return - the output of the neural network
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
