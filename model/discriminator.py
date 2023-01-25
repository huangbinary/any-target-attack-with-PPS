from torch import nn

from tools import NUM_CHANNEL
from .common import ConvBNRelu


class Discriminator(nn.Module):
    def __init__(self, data_name):
        super(Discriminator, self).__init__()
        self.channels = 64
        layers = [ConvBNRelu(NUM_CHANNEL[data_name], self.channels)]
        for _ in range(3 - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(self.channels, 1)

    def forward(self, image):
        x = self.before_linear(image)
        # b x c x 1 x 1 --> b x c
        x.squeeze_(-1).squeeze_(-1)
        x = self.linear(x)
        # x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    pass
