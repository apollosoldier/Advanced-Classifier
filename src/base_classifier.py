import torch
import torch.nn as nn
from torchvision.models import resnet34

class BaseClassifier(nn.Module):
    def __init__(self, num_classes=10):
        """
        Initialize the BaseClassifier object by using a pre-trained ResNet34 model.

        Input:
        @param - num_classes: the number of classes to be classified

        Output:
        @return - None
        """
        super(BaseClassifier, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Input:
        @param  - x: the input data

        Output:
        @return - the output of the neural network
        """
        return self.resnet(x)
