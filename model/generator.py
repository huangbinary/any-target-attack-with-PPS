import torch
from torch import nn

from tools import NUM_CHANNEL, IMG_SIZE
from .common import ConvBNRelu


class Generator(nn.Module):
    def __init__(self, data_name, num_class):
        super(Generator, self).__init__()
        in_channel = NUM_CHANNEL[data_name]
        self.img_size = IMG_SIZE[data_name]
        self.conv_channels = 64

        layers = [ConvBNRelu(in_channel, self.conv_channels)]

        for _ in range(4 - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + in_channel + num_class, self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, in_channel, kernel_size=1)

    def forward(self, image, message):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1, -1, self.img_size, self.img_size)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


if __name__ == '__main__':
    pass
