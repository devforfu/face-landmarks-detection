from pathlib import Path

DATA = Path.home()/'data'/'umdfaces'/'batch3'
META = DATA/'umdfaces_batch3_ultraface.csv'
CROPPED = DATA.parent/'cropped'
NUM_LANDMARKS = 42
