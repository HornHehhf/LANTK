import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy import special
import numpy as np
import math


class MLPNet(nn.Module):
    def __init__(self, input_size):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        # 40960
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class MLPNetSigmoid(nn.Module):
    def __init__(self, input_size):
        super(MLPNetSigmoid, self).__init__()
        self.fc1 = nn.Linear(input_size, 40960)
        self.fc2 = nn.Linear(40960, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.sigmoid(self.fc1(out))
        out = self.fc2(out)
        return out


class MLPNet3Layer(nn.Module):
    def __init__(self, input_size):
        super(MLPNet3Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 8192)
        self.fc2 = nn.Linear(8192, 8192)
        # 16384, 40960
        self.fc3 = nn.Linear(8192, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MLPProb(nn.Module):
    def __init__(self, input_size):
        super(MLPProb, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class MLPLinear(nn.Module):
    def __init__(self, input_size):
        super(MLPLinear, self).__init__()
        self.fc1 = nn.Linear(input_size, 40960)
        self.fc2 = nn.Linear(40960, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class MLPLinear1Layer(nn.Module):
    def __init__(self, input_size):
        super(MLPLinear1Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        return out


class MLPLinear3Layer(nn.Module):
    def __init__(self, input_size):
        super(MLPLinear3Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 8192)
        self.fc2 = nn.Linear(8192, 8192)
        self.fc3 = nn.Linear(8192, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class MLPLinearProb(nn.Module):
    def __init__(self, input_size):
        super(MLPLinearProb, self).__init__()
        self.fc1 = nn.Linear(input_size, 40960)
        self.fc2 = nn.Linear(40960, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = torch.sigmoid(self.fc2(out))
        return out


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        fc1_out = F.relu(self.fc1(out))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc1_out + fc2_out)
        return fc3_out


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_features(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        x = self.encoder(x)     # b, 8, 2, 2
        x = x.view(x.size(0), -1)   # b, 32
        return x


class MLPNetNoBias(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPNetNoBias, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = self.fc1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out / math.sqrt(self.hidden_size)
