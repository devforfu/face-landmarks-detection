from basedir import NUM_LANDMARKS


IMAGENET_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def to_np(t):
    return t.cpu().detach().contiguous().numpy()


def split(target):
    """Splits landmarks into two arrays of X and Y values."""
    return target[:NUM_LANDMARKS//2], target[NUM_LANDMARKS//2:]


def to_centered(xs, ys, w, h):
    """Converts absolute landmarks coordinates into relative ones in range [-1, 1]."""
    return 2*xs/w - 1, 2*ys/h - 1


def to_absolute(xs, ys, w, h):
    """Inverse function converting to_centered result back into absolute coordinates."""
    return w*(xs + 1)/2., h*(ys + 1)/2.
