import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from spikeinterface.extractors import read_intan
import kilosort

from argparse import ArgumentParser
from logging import basicConfig

from probeinterface import generate_linear_probe, ProbeGroup, write_prb
from spikeinterface.extractors import read_intan
from tqdm import tqdm

basicConfig(
    level=logging.DEBUG,
    format= '[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def parse_args() -> Path:
    parser = ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    return args.folder


def main():
    folder = parse_args()
    settings = {'data_dir': str(folder), 'n_chan_bin': 32}
    probe_dict = kilosort.io.load_probe(folder / 'probe.prb')
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = kilosort.run_kilosort(
        settings=settings, probe=probe_dict)


if __name__ == '__main__':
    main()