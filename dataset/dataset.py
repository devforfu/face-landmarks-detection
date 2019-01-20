from pathlib import Path

from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FaceLandmarks(Dataset):
    """Generic class to handle dataset of faces with facial keypoints.

    The class expects data to be stored as a set of images and text files in a single folder. The
    names of files should match. Example:
        /root
            - 1.jpeg
            - 1.txt
            - 2.jpeg
            - 2.txt
            ...

    Parameters:
         folder: Path to the folder with data.
         transforms: List of transformations applied to images and landmarks.
         to_tensors: Callable that converts dataset samples into tensors.

    """
    def __init__(self, folder, n_landmarks, transforms=None, to_tensors=None):
        folder = Path(folder).expanduser()
        if not folder.exists():
            raise FileNotFoundError(f'folder doesn\'t exist: {folder}')
        self.folder = folder
        self.n_landmarks = n_landmarks
        self.transforms = transforms
        self.to_tensors = to_tensors
        self.items = read_keypoints(folder)

    @staticmethod
    def read_items(folder, test_size=0.1):
        return train_test_split(read_keypoints(folder), test_size=test_size)

    @staticmethod
    def create_datasets(folder, n_landmarks, test_size=0.1, transforms=None, to_tensors=None):
        train_trf, valid_trf = take_or_none(transforms, 2)
        train_converter, valid_converter = take_or_none(to_tensors, 2)
        train, valid = FaceLandmarks.read_items(folder, test_size)
        return [
            FaceLandmarks(train, n_landmarks, train_trf, train_converter),
            FaceLandmarks(valid, n_landmarks, valid_trf, valid_converter)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self._transform(*self.get(item))

    def get(self, item):
        img_path, pts_path = self.items[item]
        img = imread(img_path)
        keypoints = np.loadtxt(pts_path, delimiter='.').T.flatten()
        ys, xs = split(keypoints, self.n_landmarks)
        pts = np.r_[xs, ys].astype(np.float32)
        return img, pts

    def show(self, item, ax=None, **fig_kwargs):
        self._show(*self.get(item), ax=ax, **fig_kwargs)

    def show_transformed(self, item, ax=None, **fig_kwargs):
        self._show(*self._transform(*self.get(item), as_tensors=False), ax=ax, **fig_kwargs)

    def show_random_grid(self, n=4, figsize=(10, 10), transformed=False):
        size = len(self.items)
        indexes = np.random.choice(size, n * n, replace=False)
        f, axes = plt.subplots(n, n, figsize=figsize)
        show = self.show_transformed if transformed else self.show
        for idx, ax in zip(indexes, axes.flat):
            show(idx, ax=ax)

    def _show(self, img, pts, ax=None, **fig_kwargs):
        if ax is None:
            f, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.imshow(img.astype(np.uint8), cmap='gray' if len(img.shape) == 2 else None)
        ax.scatter(*split(pts, self.n_landmarks), color='lightgreen', edgecolor='white', alpha=0.8)
        ax.set_axis_off()

    def _transform(self, img, pts, as_tensors=True):
        if self.transforms:
            for transform in self.transforms:
                img, pts = transform(img, pts)
        if as_tensors and self.to_tensors is not None:
            img, pts = self.to_tensors(img, pts)
        return img, pts

def split(target, n):
    """Splits landmarks into two arrays of X and Y values."""
    return target[:n//2], target[n//2:]


def read_keypoints(folder, img_ext='jpeg', pts_ext='txt', path_to_str=True):
    """Reads folder with images and keypoints."""

    images = read_files(folder, img_ext)
    points = read_files(folder, pts_ext)
    return [(str(i), str(p))
            if path_to_str else (i, p)
            for i, p in zip(images, points)]


def read_files(folder, ext):
    return sort_by_integer_stem(folder.glob(f'*.{ext}'))


def sort_by_integer_stem(files):
    return list(sorted(files, key=lambda f: int(f.stem)))


def take_or_none(seq, n):
    if seq is None:
        return [None] * n
    if n <= len(seq):
        return seq[:n]
    seq = seq[:]
    seq += [None] * (n - len(seq))
    return seq
