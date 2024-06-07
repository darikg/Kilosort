import csv
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pathlib import Path


def parse_args() -> Path:
    parser = ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    return args.folder


def write_to_db(root: Path):
    # outputs saved to results_dir
    results_dir = root / 'kilosort4'

    amplitudes = np.load(results_dir / 'amplitudes.npy')
    st = np.load(results_dir / 'spike_times.npy')
    clu = np.load(results_dir / 'spike_clusters.npy')

    subfolders, n_samples = [], []
    with open(root / 'concatenated.csv', newline='') as f:
        csvfile = csv.reader(f)
        _ = next(csvfile)  # skip header
        for (subfolder, n_samples) in csvfile:
            print(subfolder, n_samples)



if __name__ == '__main__':
    write_to_db(parse_args())
