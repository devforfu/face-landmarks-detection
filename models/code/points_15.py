import torch
from torch import nn
from torchvision.transforms.functional import to_tensor, resize


class ResNet(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        layers = conv(ni, 32, kernel=3, stride=2, pad=1, activ='leaky_relu', bn=True)
        layers += [
            res3x3(3, 32, 32, activ='leaky_relu'),
            res3x3(3, 32, 64, activ='leaky_relu', upsample=True),
            res3x3(3, 64, 64, activ='leaky_relu'),
            res3x3(3, 64, 128, activ='leaky_relu', upsample=True),
            res3x3(3, 128, 128, activ='leaky_relu')]
        layers += bottleneck()
        layers += fc(256, 128, bn=True, activ='relu')
        layers += fc(128, no, bn=False)
        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))

    @staticmethod
    def prepare(images):
        tensors = [to_tensor(resize(img, 96)).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0)
        return batch

    def predict(self, images):
        batch = self.prepare(images)
        preds = self.forward(batch)
        return preds.clone().detach().numpy()


model_factory = ResNet


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


def res3x3(blocks, ni, no, bn=True, upsample=False, activ='relu'):
    shortcut = nn.Conv2d(ni, no, 1, stride=2) if upsample else Identity()
    layers = []
    for _ in range(blocks - 1):
        layers += conv(ni, ni, 3, stride=1, pad=1, bn=bn, activ=activ)
    layers += conv(ni, no, 3, stride=2 if upsample else 1, pad=1, bn=bn)
    return ResidualBlock(layers, shortcut)


class ResidualBlock(nn.Module):

    def __init__(self, layers, shortcut=None):
        super().__init__()
        self.shortcut = shortcut if shortcut is not None else Identity()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        return x + res


class Identity(nn.Module):
    def forward(self, x):
        return x


class AdaptiveConcatPool2d(nn.Module):
    """Applies average and maximal adaptive pooling to the tensor and
    concatenates results into a single tensor.

    The idea is taken from fastai library.
    """
    def __init__(self, size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.max(x), self.avg(x)], 1)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


def bottleneck():
    return [AdaptiveConcatPool2d(1), Flatten()]