from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm

from prepare import Preprocessor


class UMDFaces(Preprocessor):

    def parse_args(self, args=None, namespace=None):
        self.add_argument(
            '--meta',
            required=True,
            help='Path to the file with information about dataset files'
        )
        return super().parse_args(args, namespace)

    def run(self, args=None):
        if args is None:
            args = self.parse_args()
        df = create_faces_dataframe(args.meta)
        n = len(df)
        root = Path(args.meta).expanduser().parent
        df['file'] = df.file.map(lambda p: root/p)
        df['output_image'] = [args.output_dir/f'{i}.jpeg' for i in range(n)]
        df['output_points'] = [args.output_dir/f'{i}.txt' for i in range(n)]

        with Pool(args.jobs) as pool:
            records = df.to_dict(orient='records')
            results = list(tqdm(
                pool.imap(
                    partial(worker, pad=args.pad),
                    records),
                total=n))

        data = pd.DataFrame(results)
        data.to_csv(args.output_dir.parent/'meta.csv', index=None)


def worker(record, pad):
    crop_and_safe(create_dataset_item(record), pad=pad)


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


def crop_and_safe(info, pad=None):
    x, y, w, h = [int(round(x)) for x in info['face']]
    img = PIL.Image.open(info['image_path'])
    if pad is not None and pad > 0:
        dx, dy = int(w*pad), int(h*pad)
    else:
        dx = dy = 0
    x, y, w, h = x - dx, y - dy, w + dx*2, h + dy*2
    box = x, y, x + w, y + h
    cropped = img.crop(box=box)
    x_pos, y_pos = [np.array(info[k]) for k in ('x_pos', 'y_pos')]
    x_pos -= x
    y_pos -= y
    cropped.save(info['output_image'], format='jpeg')
    np.savetxt(info['output_points'], np.c_[y_pos, x_pos], fmt='%.4f', delimiter=',')
    return {'image': info['output_image'], 'points': info['output_points']}
