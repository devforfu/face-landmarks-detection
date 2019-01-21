from torch import nn
from torchvision.models import resnet18


def get_activation(name):
    return {
        'relu': nn.ReLU(inplace=True),
        'prelu': nn.PReLU(),
        'prelu3': nn.PReLU(3),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'linear': None
    }[name]


def fc(ni, no, bn=True, activ='linear', dropout=None):
    layers = [nn.Linear(ni, no)]
    if bn:
        layers.append(nn.BatchNorm1d(no))
    if activ is not None:
        activation = get_activation(activ)
        if activation is not None:
            layers.append(activation)
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return layers


def conv(ni, no, kernel, stride, groups=1, lrn=False, bn=False, pad=0, pool=None, activ='prelu'):
    bias = not (lrn or bn)
    layers = [nn.Conv2d(ni, no, kernel, stride, pad, bias=bias, groups=groups)]
    activation = get_activation(activ)
    if activation is not None:
        layers += [nn.PReLU()]
    if lrn:
        layers.append(nn.LocalResponseNorm(2))
    elif bn:
        layers.append(nn.BatchNorm2d(no))
    if pool is not None:
        layers.append(nn.MaxPool2d(*pool))
    return layers


def res3x3(blocks, ni, no, bn=True, upsample=False):
    shortcut = nn.Conv2d(ni, no, 1, stride=2) if upsample else Identity()
    layers = conv(ni, no, 3, stride=1, pad=1, bn=bn)
    for _ in range(blocks - 1):
        layers += conv(no, no, 3, stride=1, pad=1, bn=bn)
    return ResidualBlock(layers, shortcut)


class ResidualBlock(nn.Module):

    def __init__(self, layers, shortcut=None):
        super().__init__()
        self.shortcut = shortcut if shortcut is not None else Identity()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class Identity(nn.Module):
    def forward(self, x):
        return x
