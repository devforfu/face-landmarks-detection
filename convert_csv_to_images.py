"""
Converts keypoints from CSV files into JPEG images.

Landmarks storage format (in absolute coordinates):
y1,x1
y2,x2
...
y15,x15
"""
import argparse

import numpy as np
import pandas as pd
import PIL.Image
from sklearn.model_selection import train_test_split

from simple_basedir import SIMPLE, IMG_SIZE, NUM_LANDMARKS
from utils import to_centered, split


def main():
    args = parse_args()
    trn_df = pd.read_csv(args.train_csv)
    tst_df = pd.read_csv(args.test_csv)
    trn_df.dropna(inplace=True)
    coords = list(trn_df.columns[:-1])
    x_cols = [col for col in coords if col.endswith('_x')]
    y_cols = [col for col in coords if col.endswith('_y')]
    coords = x_cols + y_cols
    index = np.arange(len(trn_df))
    train, valid = train_test_split(index, test_size=0.1, random_state=args.seed)
    save_images(trn_df, args.train_dir, train, coords)
    save_images(trn_df, args.valid_dir, valid, coords)
    save_images(tst_df, args.test_dir)
    (args.input_dir/'keypoints_legend.txt').open('w').write(','.join(coords))
    print('Done!')


def save_images(df, path, subset=None, coords=None):
    if subset is None:
        subset = np.arange(len(df))
    else:
        df = df.iloc[subset]
    print(f'Saving {len(df)} images into folder {path}...')
    sz = IMG_SIZE, IMG_SIZE
    records = df.to_dict(orient='records')
    path.mkdir(parents=True, exist_ok=True)
    for i, record in zip(subset, records):
        np_img = np.fromstring(record.pop('Image'), sep=' ').reshape(sz)
        img = PIL.Image.fromarray(np_img.astype(np.uint8))
        img_path = path/f'{i}.jpeg'
        img.save(img_path, format='jpeg')
        if coords is not None:
            keypoints = np.array([record[coord] for coord in coords])
            xs, ys = to_centered(*split(keypoints, NUM_LANDMARKS), *sz)
            keypoints = np.c_[ys, xs]
            np.savetxt(path/f'{i}.txt', keypoints, fmt='%.4f', delimiter=',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input-dir',
        default=SIMPLE,
        help='Path to folder with train/test CSV files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=SIMPLE,
        help='Path to folder where to store the images'
    )
    parser.add_argument(
        '-s', '--seed',
        default=1, type=float,
        help='Random generator seed'
    )
    parser.add_argument(
        '-r', '--rescale',
        action='store_true',
        help='Scale landmarks relatively to image size'
    )
    args = parser.parse_args()
    args.train_csv = args.input_dir / 'training.csv'
    args.test_csv = args.input_dir / 'test.csv'
    args.train_dir = args.output_dir / 'train'
    args.valid_dir = args.output_dir / 'valid'
    args.test_dir = args.output_dir / 'test'
    return args



if __name__ == '__main__':
    main()
