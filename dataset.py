from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from plot import split


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
        pts = np.array(record['x_pos'] + record['y_pos'], dtype='float32')
        return img, pts

    def _show(self, img, pts, ax=None, **fig_kwargs):
        if ax is None:
            f, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.imshow(img.astype(np.uint8))
        ax.scatter(*split(pts), color='lightgreen', edgecolor='white', alpha=0.8)
        ax.set_axis_off()

    def _transform(self, img, pts, as_tensors=True):
        if self.transforms:
            for transform in self.transforms:
                img, pts = transform(img, pts)
        if as_tensors and self.to_tensors is not None:
            img, pts = self.to_tensors(img, pts)
        return img, pts
