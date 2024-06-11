import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import kilosort
from kilosort.run_kilosort import setup_logger
from kilosort.xper.concatenate_files import concatenate
from kilosort.xper.write_to_db import write_to_db

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('command', type=str, help='concat | sort | upload | all')
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    return args


def sort_spikes(data_dir: Path, results_dir: Path):
    # ilosort\parameters.py for other settable params
    settings = dict(
        data_dir=str(data_dir),
        n_chan_bin=32,
        fs=20_000,
    )
    probe_dict = kilosort.io.load_probe(data_dir / 'probe.prb')
    _ = kilosort.run_kilosort(settings=settings, probe=probe_dict, init_logging=False, results_dir=results_dir)


def main():
    args = parse_args()
    results_dir = args.folder / 'kilosort4'
    results_dir.mkdir(exist_ok=True)
    setup_logger(results_dir=results_dir)

    if args.command == 'concat':
        concatenate(args.folder)
    elif args.command == 'sort':
        sort_spikes(args.folder, results_dir=results_dir)
    elif args.command == 'upload':
        write_to_db(args.folder)
    elif args.command == 'all':
        concatenate(args.folder)
        sort_spikes(args.folder, results_dir=results_dir)
        write_to_db(args.folder)


if __name__ == '__main__':
    main()