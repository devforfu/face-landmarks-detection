"""
Takes the original UMD Faces dataset and converts into cropped images and
landmarks stored as .txt files.
"""
import argparse
from multiprocessing import Pool, cpu_count
import os
from functools import partial
import random

import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm

from basedir import META, CROPPED


SEED = 1
random.seed(SEED)


def main():
    args = parse_args()
    root = args.meta.parent
    faces_df = create_faces_dataframe(args.meta)
    n = len(faces_df)
    faces_df['file'] = faces_df.file.map(lambda p: root/p)
    faces_df['output_image'] = [args.output_dir/f'{i}.jpeg' for i in range(n)]
    faces_df['output_points'] = [args.output_dir/f'{i}.txt' for i in range(n)]
    run(faces_df, args)


def run(df, args):
    n = len(df)
    convert = to_centered if args.centered else None
    with Pool(args.jobs) as pool:
        records = df.to_dict(orient='records')
        results = list(tqdm(
            pool.imap(
                partial(worker, pad=args.pad, convert=convert),
                records),
            total=n))
    data = pd.DataFrame(results)
    data.to_csv(args.output_dir.parent/'meta.csv', index=None)


def worker(record, pad, convert):
    info = create_dataset_item(record)
    crop_and_safe(info, pad=pad, convert=convert)


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
        'subject_id': info['subject_id'],
        'image_path': str(info['file']),
        'output_image': str(info['output_image']),
        'output_points': str(info['output_points']),
        'x_pos': [info[k] for k in info.keys() if k[0] == 'p' and k[-1] == 'x'],
        'y_pos': [info[k] for k in info.keys() if k[0] == 'p' and k[-1] == 'y'],
        'face': (info['face_x'], info['face_y'], info['face_width'], info['face_height'])}


def to_centered(xs, ys, w, h):
    return 2*xs/w - 1, 2*ys/h - 1


def crop_and_safe(info, pad=None, convert=None):
    x, y, w, h = [int(round(x)) for x in info['face']]
    img = PIL.Image.open(info['image_path'])
    img_h, img_w = img.size
    if pad is not None and pad > 0:
        dx, dy = int(w*pad), int(h*pad)
    else:
        dx = dy = 0
    x -= dx
    y -= dy
    w += dx*2
    h += dy*2
    # box = (
    #     max(0,     x),
    #     max(0,     y),
    #     min(img_w, x + w),
    #     min(img_h, y + h))
    box = x, y, x + w, y + h
    cropped = img.crop(box=box)
    x_pos, y_pos = [np.array(info[k]) for k in ('x_pos', 'y_pos')]
    if convert is not None:
        x_pos, y_pos = convert(x_pos - x, y_pos - y, w, h)
    cropped.save(info['output_image'], format='jpeg')
    np.savetxt(info['output_points'], np.c_[y_pos, x_pos], fmt='%.4f', delimiter=',')
    return {'image': info['output_image'], 'points': info['output_points']}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--meta',
        default=META,
        help='Path to the file with information about dataset files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=CROPPED,
        help='Path to the output folder with cropped faces'
    )
    parser.add_argument(
        '-j', '--jobs',
        default=cpu_count(), type=int,
        help='Number of parallel workers; single-threaded mode if None'
    )
    parser.add_argument(
        '-p', '--pad',
        default=0.15, type=float,
        help='Margin around face (as a percentage of face size)'
    )
    parser.add_argument(
        '-c', '--centered',
        default=False, action='store_true',
        help='Convert landmarks from absolute to centered coordinates'
    )
    parser.add_argument(
        '-d', '--debug',
        default=False, action='store_true',
        help='Run script in debugging mode'
    )
    args = parser.parse_args()
    args.parallel = args.jobs is not None and args.jobs > 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONBREAKPOINT'] = 'pudb.set_trace' if args.debug else '0'
    return args


if __name__ == '__main__':
    main()
