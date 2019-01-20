import json
import os


def dataset_info(name, meta_file):
    with open(meta_file) as file:
        datasets = json.load(file)
    if name not in datasets:
        raise ValueError(f'unknown dataset: {name}')
    config = datasets[name]
    folder = os.path.expanduser(config['folder'])
    n_landmarks = config['num_of_landmarks']
    return folder, n_landmarks
