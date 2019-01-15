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

from simple_basedir import SIMPLE, IMG_SIZE


def main():
    args = parse_args()
    trn_df = pd.read_csv(args.train_csv)
    tst_df = pd.read_csv(args.test_csv)
    coords = list(trn_df.columns[:-1])
    index = np.arange(len(trn_df))
    train, valid = train_test_split(index, test_size=0.2, random_state=args.seed)
    save_images(trn_df[train], args.train_dir, coords)
    save_images(trn_df[valid], args.valid_dir, coords)
    save_images(tst_df, args.test_dir)


def save_images(df, path, coords=None):
    sz = IMG_SIZE, IMG_SIZE
    records = df.to_dict(orient='records')
    path.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        np_img = np.array([int(x) for x in record.pop('image').split()]).reshape(sz)
        img = PIL.Image.fromarray(np_img)
        img_path = path/f'{i}.jpeg'
        img.save(img_path, format='jpeg')
        if coords is not None:
            keypoints = np.array([record[coord] for coord in coords])
            keypoints = np.c_[keypoints[:, 1], keypoints[:, 0]]
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
    args = parser.parse_args()
    args.train_csv = args.input_dir / 'training.csv'
    args.test_csv = args.input_dir / 'test.csv'
    args.train_dir = args.output_dir / 'train'
    args.valid_dir = args.output_dir / 'valid'
    args.test_dir = args.output_dir / 'test'
    return args



if __name__ == '__main__':
    main()
