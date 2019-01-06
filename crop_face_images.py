"""
Takes the original UMD Faces dataset and converts into cropped images and
landmarks stored as .txt files.
"""
import argparse
import os
import random

import pandas as pd
import PIL.Image

from basedir import META


SEED = 1
random.seed(SEED)


def main():
    args = parse_args()
    root = args.meta.parent
    faces_df = create_faces_dataframe(args.meta)
    faces_df['file'] = faces_df.file.map(lambda p: root/p)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--meta',
        default=META,
        help='Path to the file with information about dataset files'
    )
    parser.add_argument(
        '-j', '--jobs',
        default=None,
        help='Number of parallel workers; single-threaded mode if None'
    )
    parser.add_argument(
        '-p', '--pad',
        default=0.15,
        help='Margin around face (as a percentage of face size)'
    )
    parser.add_argument(
        '-d', '--debug',
        default=False, action='store_true',
        help='Run script in debugging mode'
    )
    args = parser.parse_args()
    args.parallel = args.jobs is not None and args.jobs > 1
    os.environ['PYTHONBREAKPOINT'] = 'pudb.set_trace' if args.debug else '0'
    return args


def create_faces_dataframe(filename):
    meta = pd.read_csv(filename)
    meta.columns = meta.columns.str.lower()
    cols = meta.columns
    file_cols = ['subject_id', 'file']
    face_cols = cols[cols.str.startswith('face')].tolist()
    x_cols = cols[cols.str.startswith('p') & cols.str.endswith('x')].tolist()
    y_cols = cols[cols.str.startswith('p') & cols.str.endswith('y')].tolist()
    df = meta[file_cols + face_cols + x_cols + y_cols]
    return df


def create_dataset_item(info):
    return {
        'subject_id': info.subject_id,
        'image_path': info.file,
        'x_pos': [info[k] for k in info.keys() if k[0] == 'p' and k[-1] == 'x'],
        'y_pos': [info[k] for k in info.keys() if k[0] == 'p' and k[-1] == 'y'],
        'face': (info.face_x, info.face_y, info.face_width, info.face_height)}


def to_centered(xs, ys, w, h):
    return 2*xs/w - 1, 2*ys/h - 1


def crop_and_safe(info, pad=None, convert=None):
    x, y, w, h = info['face']
    img = PIL.Image.open(info['input_path'])


if __name__ == '__main__':
    main()
