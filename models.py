import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        return out


class MLPNetNoBias(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPNetNoBias, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = self.fc1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out / math.sqrt(self.hidden_size)


class MLPNetNoBiasLinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPNetNoBiasLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = self.fc1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out / math.sqrt(self.hidden_size)


class MLPNetNoBiasFalse(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPNetNoBiasFalse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = self.fc1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out / math.sqrt(self.hidden_size)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(16, 16 * 4, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(16 * 4, 16 * 4, 3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(16 * 4, 32 * 4, 3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(32 * 4, 32 * 4, 3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(32 * 4, 64 * 4, 3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(64 * 4, 64 * 4, 3, stride=1, padding=1)
        self.fc = nn. Linear(64 * 8 * 8 * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv31(x))
        x = self.conv32(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


class CNN_multi(nn.Module):
    def __init__(self):
        super(CNN_multi, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(16, 16 * 8, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(16 * 8, 16 * 8, 3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(16 * 8, 32 * 8, 3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(32 * 8, 32 * 8, 3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(32 * 8, 64 * 8, 3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(64 * 8, 64 * 8, 3, stride=1, padding=1)
        self.fc = nn. Linear(64 * 8 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv31(x))
        x = self.conv32(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


class CNN_multi_V2(nn.Module):
    def __init__(self):
        super(CNN_multi_V2, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv22 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv23 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv24 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv31 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv32 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv34 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.fc = nn. Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = F.relu(self.conv24(x))
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.conv34(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out
