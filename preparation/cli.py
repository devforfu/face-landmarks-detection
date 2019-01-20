import sys
from prepare_umdfaces import UMDFaces


PREPROCESSORS = {'umdfaces': UMDFaces()}


def main():
    PREPROCESSORS[sys.argv.pop(1)].run()


if __name__ == '__main__':
    main()
