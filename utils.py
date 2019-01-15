from basedir import get_num_landmarks


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def to_np(t):
    return t.cpu().detach().contiguous().numpy()


def split(target, n=None):
    """Splits landmarks into two arrays of X and Y values."""
    n = n or get_num_landmarks()
    return target[:n//2], target[n//2:]


def to_centered(xs, ys, w, h):
    """Converts absolute landmarks coordinates into relative ones in range [-1, 1]."""
    return 2*xs/w - 1, 2*ys/h - 1


def to_absolute(xs, ys, w, h):
    """Inverse function converting to_centered result back into absolute coordinates."""
    return w*(xs + 1)/2., h*(ys + 1)/2.


def read_keypoints(folder, img_ext='jpeg', pts_ext='txt', path_to_str=True):
    """Reads folder with images and keypoints."""
    images = read_files(folder, img_ext)
    points = read_files(folder, pts_ext)
    return [(str(i), str(p)) if path_to_str else (i, p) for i, p in zip(images, points)]


def read_files(folder, ext):
    return sort_by_integer_stem(folder.glob(f'*.{ext}'))


def sort_by_integer_stem(files):
    return list(sorted(files, key=lambda f: int(f.stem)))
