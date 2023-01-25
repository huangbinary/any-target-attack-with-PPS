"""Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch.nn as nn
import torch.nn.functional as F

from tools import NUM_CLASS


class PreActResNet18(nn.Module):
    def __init__(self, data_name):
        super(PreActResNet18, self).__init__()
        self.in_channels = 64

        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, 64, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(PreActBlock, 128, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(PreActBlock, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(PreActBlock, 512, stride=2, num_blocks=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASS[data_name])

    def _make_layer(self, block, channels, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreActBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.index = None

        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.index is not None:
            out += shortcut[:, self.index, :, :]
        else:
            out += shortcut
        return out


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.1)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=0)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(0.1)
        self.maxpool5 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.fc6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.1)
        self.fc9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


if __name__ == '__main__':
    pass
