# from mindspore import nn
#
#
# def _make_layer(channels):
#     in_channels = channels[0]
#     layers = []
#     for out_channels in channels[1:]:
#         layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode="same",
#                                 has_bias=True))
#         layers.append(nn.ReLU())
#         in_channels = out_channels
#     return nn.SequentialCell(layers)
#
#
# class Vgg16(nn.Cell):
#     """VGG16 module."""
#
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.b1 = _make_layer([3, 64, 64])
#         self.b2 = _make_layer([64, 128, 128])
#         self.b3 = _make_layer([128, 256, 256, 256])
#         self.b4 = _make_layer([256, 512, 512, 512])
#         self.b5 = _make_layer([512, 512, 512, 512])
#
#         self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
#         self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
#         self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
#         self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
#         self.m5 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
#
#     def construct(self, x):
#         # block1
#         x = self.b1(x)
#         x = self.m1(x)
#
#         # block2
#         x = self.b2(x)
#         x = self.m2(x)
#
#         # block3
#         x = self.b3(x)
#         x = self.m3(x)
#
#         # block4
#         x = self.b4(x)
#         block4 = x
#         x = self.m4(x)
#
#         # block5
#         x = self.b5(x)
#         block5 = x
#         x = self.m5(x)
#
#         return block4, block5, x

import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer


def _make_layer(base, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight = 'ones'

            weight_shape = (v, in_channels, 3, 3)
            # weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32)

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=0,
                               pad_mode='same',
                               has_bias=True,
                               weight_init=weight)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return layers[:23], layers[23:]


class Vgg(nn.Cell):

    def __init__(self, base, batch_norm=False):
        super(Vgg, self).__init__()
        self.conv4_3, self.conv5_3 = _make_layer(base, batch_norm=batch_norm)
        self.conv4_3 = nn.SequentialCell(self.conv4_3)
        self.conv5_3 = nn.SequentialCell(self.conv5_3)

    def construct(self, x):
        conv4_3 = self.conv4_3(x)
        conv5_3 = self.conv5_3(conv4_3)
        return conv4_3, conv5_3

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    init.Normal,
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]


def vgg16():
    net = Vgg(cfg)
    return net
