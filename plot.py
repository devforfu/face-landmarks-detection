import os

from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from utils import to_absolute


def show(root, i, ax=None):
    """Shows i-th face and landmarks from the `root` folder."""

    img = imread(root / f'{i}.jpeg')
    pts = np.loadtxt(root / f'{i}.txt', delimiter=',')
    h, w = img.shape[:2]
    y, x = pts[:, 0], pts[:, 1]
    x, y = to_absolute(x, y, w, h)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(f'index={i}\n({w}w x {h}h)')
    ax.set_axis_off()
    ax.scatter(x, y, edgecolor='white', color='lightgreen', alpha=0.8)


def show_random_grid(root, n=4, figsize=(10, 10)):
    """Takes n*n random images from the dataset and shows them using grid layout."""

    f, axes = plt.subplots(n, n, figsize=figsize)
    n_images = len(os.listdir(root))//2
    indexes = np.random.choice(n_images, size=n*n, replace=False)
    for idx, ax in zip(indexes, axes.flat):
        show(root, idx, ax=ax)
    f.tight_layout(h_pad=0.05, w_pad=0.05)


def plot_loss_curve(learning_rates, log_loss, zoom=None, ax=None, figsize=(10, 8)):
    """Plots curve reflecting the dependency between learning rate and training loss."""

    lrs, losses = learning_rates, log_loss
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    if zoom is not None:
        min_lr, max_lr = zoom
        lrs, losses = zip(*[(x, y) for x, y in zip(lrs, losses) if min_lr <= x <= max_lr])
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%1.0e'))
