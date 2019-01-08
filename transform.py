from copy import deepcopy

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize

from utils import split, to_centered, to_np, to_absolute


def binomial(p):
    return np.random.rand() <= p


def rotation_matrix(x, y=None, angle=5):
    h, w = x.shape[:2]
    minmax = (-angle, angle) if isinstance(angle, int) else angle
    a = np.random.uniform(*minmax)
    m = cv.getRotationMatrix2D((w/2, h/2), a, 1)
    m = np.r_[m, [[0, 0, 1]]]
    return m


def shift_matrix(x, y=None, shift=0.01):
    h, w = x.shape[:2]
    sx, sy = (shift, shift) if isinstance(shift, float) else shift
    shift_x = np.random.randint(-w*sx, w*sx)
    shift_y = np.random.randint(-h*sy, h*sy)
    m = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0, 1]])
    return m


def mirror_matrix(x, y=None, horizontal=True):
    h, w = x.shape[:2]
    c1, c2 = (-1, 1) if horizontal else (1, -1)
    s1, s2 = (w, 0) if horizontal else (0, h)
    return np.float32([[c1, 0, s1], [0, c2, s2], [0, 0, 1]])


def perspective_matrix(x, y=None, percentage=(0.05, 0.12)):
    h, w = x.shape[:2]

    def rx():
        return int(w * np.random.uniform(*percentage))

    def ry():
        return int(h * np.random.uniform(*percentage))

    tl = [0 + rx(), 0 + ry()]
    tr = [w - 1 - rx(), 0 + ry()]
    br = [w - 1 - rx(), h - 1 - ry()]
    bl = [0 + rx(), h - 1 - ry()]
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    m = cv.getPerspectiveTransform(src, dst)
    return m


class MatrixAugmentation:
    def __init__(self, conf, default_prob=0.5):
        augmentations = {
            'rotation': rotation_matrix,
            'shift': shift_matrix,
            'mirror': mirror_matrix,
            'perspective': perspective_matrix}

        pipe = []
        for params in deepcopy(conf):
            name = params.pop('name')
            func = augmentations.get(name)
            if func is None:
                raise ValueError(f'unknown augmentation function: {name}')
            p = params.pop('p', default_prob)
            pipe.append((p, func, params))

        self.pipe = pipe

    def __call__(self, image, target=None):
        h, w = image.shape[:2]

        m = np.eye(3)
        for p, func, params in self.pipe:
            if binomial(p):
                m = func(image, target, **params) @ m

        aug_image = cv.warpPerspective(image, m, (w, h))
        if target is not None:
            new = target.copy()
            n = len(target) // 2
            for i in range(n):
                x, y = target[i], target[i + n]
                denom = m[2][0] * x + m[2][1] * y + m[2][2]
                new[i] = (m[0][0] * x + m[0][1] * y + m[0][2]) / denom
                new[i + n] = (m[1][0] * x + m[1][1] * y + m[1][2]) / denom
            target = new

        return aug_image, target


class CropResizeFace:
    def __init__(self, size=224, pad=0.05):
        self.size = size
        self.pad = pad
        self.failed = []

    def __call__(self, image, target):
        size = self.size
        xs, ys = split(target)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()

        w, h = x_max - x_min, y_max - y_min
        x_pad = w * self.pad
        y_pad = h * self.pad

        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max),

        cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        if 0 in cropped.shape:
            self.failed.append((image, target))
            cropped = image
        else:
            xs -= x_min
            ys -= y_min

        new_w, new_h = (size, size) if isinstance(size, int) else size
        old_h, old_w = cropped.shape[:2]
        cropped = cv.resize(cropped, (new_w, new_h))
        rx, ry = new_w / old_w, new_h / old_h
        xs *= rx
        ys *= ry
        return cropped, np.r_[xs, ys]


class AdjustGamma:
    def __init__(self, min_gamma=0.5, max_gamma=1.5):
        self.minmax = min_gamma, max_gamma

    def __call__(self, image, target=None):
        gamma = np.random.uniform(*self.minmax)
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)])
        adjusted = cv.LUT(
            image.astype(np.uint8),
            table.astype(np.uint8))
        return adjusted.astype(np.float32), target


class ToXY:
    """Converts image and landmarks into tensors of appropriate format.

    Includes inverse transformation to convert training and prediction tensors
    back into visualization-suitable format.
    """

    def __init__(self, stats=None, y_rescale=True):
        self.stats = stats
        self.y_rescale = y_rescale

    def __call__(self, image, points):
        return self.transform(image, points)

    def transform(self, image, points):
        if self.y_rescale:
            xs, ys = split(points)
            h, w = image.shape[:2]
            points = np.r_[to_centered(xs, ys, w, h)]
        t_img = to_tensor(image.astype(np.uint8))
        t_pts = torch.tensor(points, dtype=t_img.dtype)
        if self.stats is not None:
            t_img = normalize(t_img, *self.stats)
        return t_img, t_pts

    def inverse(self, t_image, t_points):
        np_img, np_pts = [to_np(t) for t in (t_image, t_points)]
        np_img = np_img.transpose(1, 2, 0)
        np_pts = np_pts.flatten()
        if self.y_rescale:
            h, w = np_img.shape[:2]
            xs, ys = to_absolute(*split(np_pts), w, h)
            np_pts = np.r_[xs, ys]
        if self.stats is not None:
            mean, std = self.stats
            np_img *= np.array(std)
            np_img += np.array(mean)
        np_img *= 255
        np_img = np_img.astype(np.uint8)
        return np_img, np_pts