from torch import nn


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


def conv(ni, no, kernel, stride, groups=1, lrn=False, bn=False, pool=None, activ='prelu'):
    layers = [nn.Conv2d(ni, no, kernel, stride, groups=groups)]
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
