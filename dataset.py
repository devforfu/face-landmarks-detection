from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

from utils import split, read_keypoints


class FaceLandmarks:

    def __init__(self, items, transforms=None, to_tensors=None):
        self.items = items
        self.transforms = transforms
        self.to_tensors = to_tensors

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self._transform(*self.get(item))

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

    def get(self, item):
        record = self.items[item]
        img = imread(record['image_path'])
        pts = np.array(record['x_pos'] + record['y_pos'], dtype=np.float32)
        return img, pts

    def _show(self, img, pts, ax=None, **fig_kwargs):
        if ax is None:
            f, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.imshow(img.astype(np.uint8), cmap='gray' if len(img.shape) == 2 else None)
        ax.scatter(*split(pts), color='lightgreen', edgecolor='white', alpha=0.8)
        ax.set_axis_off()

    def _transform(self, img, pts, as_tensors=True):
        if self.transforms:
            for transform in self.transforms:
                img, pts = transform(img, pts)
        if as_tensors and self.to_tensors is not None:
            img, pts = self.to_tensors(img, pts)
        return img, pts


class LandmarksFromFiles(FaceLandmarks):
    """Same as FaceLandmarks dataset, but reads images from the folder instead
    of using records with meta information.
    """
    def __init__(self, folder, transforms=None, to_tensors=None):
        super().__init__(read_keypoints(folder), transforms, to_tensors)

    def get(self, item):
        img_path, pts_path = self.items[item]
        img = imread(img_path)
        ys, xs = split(np.loadtxt(pts_path, delimiter=',').T.flatten())
        pts = np.r_[xs, ys].astype(np.float32)
        return img, pts
