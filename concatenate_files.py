from __future__ import annotations

import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Iterator

import numpy as np
from spikeinterface.extractors import read_intan, IntanRecordingExtractor
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


def write_probe_file(probe_file: Path):
    probe = generate_linear_probe(
        num_elec=32,
        ypitch=65,  # um
        contact_shape_params=dict(radius=10),
    )
    probe.set_device_channel_indices([
        25, 6, 21, 10, 26, 5, 20, 11,
        22, 9, 27, 4, 19, 12, 28, 3,
        24, 7, 18, 13, 29, 2, 17, 14,
        15, 16, 1, 30, 8, 23, 0, 31,
    ])
    probegroup = ProbeGroup()
    probegroup.add_probe(probe)
    write_prb(str(probe_file), probegroup)


def parse_args() -> Path:
    parser = ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    return args.folder


def get_subfolders(root: Path) -> list[Path]:
    subfolders = []
    for f in root.iterdir():
        if f.is_dir():
            subfolders.append(f)

    subfolders.sort(key=lambda folder: folder.stem)  # sort by timestamp

    return subfolders


def get_chunk_indices(recording, chunksize: int) -> list[tuple[int, int, int]]:  # (segment, start, end)
    # Determine start/end indices for each segment
    indices = []
    for k in range(recording.get_num_segments()):
        n = recording.get_num_samples(segment_index=k)
        i = 0 + k * chunksize
        while i < n:
            j = i + chunksize if (i + chunksize) < n else n
            indices.append((i, j, k))
            i += chunksize

    return indices


class IntanRecording:
    def __init__(self, folder: Path, recording: IntanRecordingExtractor):
        self.folder = folder
        self.recording = recording

    @cached_property
    def n_samples(self) -> int:
        return self.recording.get_total_samples()

    @cached_property
    def n_channels(self) -> int:
        return self.recording.get_traces(start_frame=0, end_frame=1, segment_index=0).shape[1]

    @staticmethod
    def from_root(root: Path) -> Iterator[IntanRecording]:
        for info_file in tqdm(list(root.glob('**/info.rhd')), desc='Loading recordings', leave=False):
            recording = read_intan(info_file, stream_id='0')
            ir = IntanRecording(folder=info_file.parent, recording=recording)
            _ = ir.n_channels
            _ = ir.n_samples
            yield ir


def concatenate(root: Path):
    binary_file = root / 'concatenated.bin'
    chunksize = 300000
    max_workers = min(32, (os.cpu_count() or 1) + 4)

    recordings = list(IntanRecording.from_root(root))
    n_samples = [r.n_samples for r in recordings]
    n_channels = {r.n_channels for r in recordings}
    if len(n_channels) != 1:
        ch = ', '.join(str(i) for i in n_channels)
        msg = f"Expected all recordings to have the same number of channels, got {ch}"
        raise ValueError(msg)

    with (root / 'concatenated.csv').open('w+', newline='') as summary:
        csvfile = csv.writer(summary)
        csvfile.writerow(['subfolder', 'n_samples'])
        for r in recordings:
            csvfile.writerow([r.folder.parts[-1], r.n_samples])

    n_total = sum(n_samples)
    memmapped_output = np.memmap(str(binary_file), dtype=np.int16, mode='w+', shape=(n_total, n_channels.pop()))
    subfolder_start_idx = 0

    for r in (pbar0 := tqdm(recordings, desc='Writing recordings')):
        pbar0.set_postfix(dict(subfolder=r.folder.parts[-1]))
        indices = get_chunk_indices(r.recording, chunksize=chunksize)

        def copy_chunk(memmap, i, j, k):
            t = r.recording.get_traces(start_frame=i, end_frame=j, segment_index=k)
            memmap[subfolder_start_idx + np.arange(i, j), :] = t
            memmap.flush()
            del t

        n_chunks = len(indices)
        pbar1 = tqdm(total=n_chunks, desc=f'Writing chunks for {r.folder}', leave=False)

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            _futures = [exe.submit(copy_chunk, memmapped_output, i, j, k) for i, j, k in indices]
            while exe._work_queue.qsize() > 0:  # noqa
                time.sleep(1)
                pbar1.n = n_chunks - exe._work_queue.qsize()  # noqa
                pbar1.refresh()

        subfolder_start_idx += r.n_samples

    del memmapped_output
    write_probe_file(root / 'probe.prb')


def main():
    root = parse_args()
    concatenate(root)


if __name__ == '__main__':
    main()
