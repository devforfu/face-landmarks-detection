from pathlib import Path

_use_simple = False

# UMDFaces dataset
DATA = Path.home()/'data'/'umdfaces'/'batch3'
META = DATA/'umdfaces_batch3_ultraface.csv'
CROPPED = DATA.parent/'cropped'

# Kaggle Face Keypoints dataset
SIMPLE = Path.home()/'data'/'keypoints'
TRAIN_CSV = SIMPLE/'train.csv'
TEST_CSV = SIMPLE/'test.csv'
TRAIN = SIMPLE/'train'
VALID = SIMPLE/'valid'
TEST = SIMPLE/'test'
SMALL_IMG_SIZE = 96


def use_simple(use=True):
    """If True, then use 30 landmarks to split array with target, otherwise - 42."""
    global _use_simple
    _use_simple = use


def get_num_landmarks():
    """Returns number of landmarks per sample of the dataset."""
    return 30 if _use_simple else 42
