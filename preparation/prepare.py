from argparse import ArgumentParser
import os
from pathlib import Path


class Preprocessor(ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        self.add_argument(
            '-p', '--pad',
            default=0.15, type=float,
            help='Margin around face (as a percentage of face size)'
        )
        self.add_argument(
            '-o', '--output-dir',
            required=True,
            help='Path to the output folder with prepared face images'
        )
        self.add_argument(
            '-j', '--jobs',
            default=1, type=int,
            help='Number of parallel workers'
        )
        self.add_argument(
            '-d', '--debug',
            default=False, action='store_true',
            help='Run script in debugging mode'
        )
        args = super().parse_args()
        args.parallel = args.jobs is not None and args.jobs > 1
        args.output_dir = Path(args.output_dir).expanduser()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ['PYTHONBREAKPOINT'] = 'pudb.set_trace' if args.debug else '0'
        return args

    def run(self, args=None):
        raise NotImplementedError()
