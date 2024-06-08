from __future__ import annotations

import csv
import logging
from argparse import ArgumentParser
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd

logging.basicConfig(
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



def cslstr(x):
    return ', '.join(str(i) for i in x)


def load_recordings(root: Path, curs):
    # Returns df with columns recording_id, cum_samples
    logger.debug("Loading recording data from db")
    notefiles, n_samples = [], []
    with open(root / 'concatenated.csv', newline='') as f:
        csvfile = csv.reader(f)
        _ = next(csvfile)  # skip header
        for (subfolder, n) in csvfile:
            notefiles.append(subfolder.replace('experiment', 'notes') + '.txt')
            n_samples.append(int(n))

    recordings = pd.DataFrame(dict(
        notefile=notefiles,
        cum_samples=np.append(0, np.cumsum(n_samples[:-1])),
    ))

    nf_str = ', '.join(f'"{f}"' for f in recordings.notefile)
    curs.execute(f"SELECT recording_id, notefile FROM Recordings WHERE notefile IN ({nf_str})")
    rdf = pd.DataFrame.from_records(curs.fetchall(), columns=['recording_id', 'notefile'], index='notefile')
    curs.execute(f"DELETE FROM ClusteredRuns WHERE recording_id IN ({cslstr(rdf.recording_id)})")

    recordings = rdf.join(
        recordings.set_index('notefile'), on='notefile', how='inner'
    ).reset_index(drop=True)
    return recordings


def load_trials(curs, recordings: pd.DataFrame):
    # Returns df with columns trial_id, start_sample, stop_sample
    logger.debug("Loading trial data from db")
    curs.execute(f"""
            SELECT    recording_id, trial_id, start_sample, stop_sample 
            FROM      Trials INNER JOIN Recordings USING (recording_id)
            WHERE     recording_id IN ({cslstr(recordings.recording_id)})
            ORDER BY  recording_id, start_sample
        """)
    trials = pd.DataFrame.from_records(
        curs.fetchall(),
        columns=['recording_id', 'trial_id', 'start_sample', 'stop_sample'],
    )
    trials = trials.join(recordings.set_index('recording_id'), on='recording_id')
    trials.start_sample += trials.cum_samples
    trials.stop_sample += trials.cum_samples
    trials.drop(["recording_id", "cum_samples"], axis=1, inplace=True)
    trials.set_index(pd.Series(np.arange(len(trials)), name='trial_idx'), inplace=True)
    logger.debug("Found %s trials", len(trials))
    return trials


def write_to_db(root: Path, db):
    logger.info("Loading kilosort data")
    results_dir = root / 'kilosort4'

    # One value per cluster
    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
    clu_labels = pd.read_csv(results_dir / 'cluster_KSLabel.tsv', sep='\t')['KSLabel'].values
    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
    chan_map = np.load(results_dir / 'channel_map.npy')
    templates = np.load(results_dir / 'templates.npy')
    chan_best = (templates ** 2).sum(axis=1).argmax(axis=-1)
    chan_best = chan_map[chan_best]

    # One value per spike
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    spike_sample = np.load(results_dir / 'spike_times.npy')
    cluster_idx = np.load(results_dir / 'spike_clusters.npy')
    sr = 20_000
    curs = db.cursor()

    recordings = load_recordings(root, curs)
    trials = load_trials(curs, recordings)
    _session_id = write_clusters(curs, recordings, camps, contam_pct, chan_best, clu_labels)
    write_spikes(
        curs=curs,
        trials=trials,
        cluster_idx=cluster_idx,
        amplitudes=amplitudes,
        spike_sample=spike_sample,
        sr=sr,
    )
    db.commit()
    logger.debug("Finished")


def write_spikes(
        curs,
        trials: pd.DataFrame,
        cluster_idx: np.ndarray,
        amplitudes: np.ndarray,
        spike_sample: np.ndarray,
        sr: float,
        temp_csv_file=r"data.csv"
):

    trial_id_str = ','.join(str(i) for i in trials.trial_id.unique())
    q = f'DELETE FROM SortedSpikeTimes WHERE trial_id IN ({trial_id_str})'
    curs.execute(q)

    spikes = pd.DataFrame(dict(
        spike_sample=spike_sample,
        channel=cluster_idx,
        amplitude=amplitudes,
        trial_idx=np.searchsorted(trials.start_sample, spike_sample) - 1,
    ))
    trial_spks = spikes.join(trials, how='inner', on='trial_idx')
    n0 = len(trial_spks)
    trial_spks.query('start_sample < spike_sample < stop_sample', inplace=True)
    n1 = len(trial_spks)
    logger.debug("Assigned %s of %s spikes to trials", n0, n1)
    trial_spks['time'] = (trial_spks.spike_sample - trial_spks.start_sample) / sr
    trial_spks = trial_spks.loc[:, ['trial_id', 'channel', 'time', 'amplitude']]

    logger.info("Writing spike data to temporary file")
    trial_spks.to_csv(temp_csv_file, index=False)

    logger.info("Writing spike data to database")
    q = rf"""
        LOAD DATA LOCAL INFILE '{temp_csv_file}' 
        INTO TABLE SortedSpikeTimes
        FIELDS TERMINATED BY ','
        LINES TERMINATED by '\r\n'
        IGNORE 1 LINES
    """
    curs.execute(q)
    logger.info("Wrote %s spikes", len(trial_spks))


def write_clusters(curs, recordings, camps, contam_pct, chan_best, clu_labels) -> int:
    # Write session and clusters
    curs.execute('INSERT INTO ClusteredSessions () VALUES ()')
    curs.execute('SELECT MAX(session_id) FROM ClusteredSessions')
    session_id, = curs.fetchone()
    logger.debug("Wrote new session id %s", session_id)
    q = "INSERT INTO ClusteredRuns (session_id, recording_id) VALUES (%s, %s)"
    curs.executemany(q, [(session_id, rid) for rid in recordings.recording_id])
    logger.info("Wrote %s recording ids for session %s", len(recordings), session_id)
    q = "INSERT INTO Clusters (session_id, channel, amplitude, contam_pct, best_channel, label) VALUES (%s, %s, %s, %s, %s, %s)"
    data = [
        (session_id, i, amp, cptc, int(best_ch), lbl)
        for i, (amp, cptc, best_ch, lbl)
        in enumerate(zip(camps, contam_pct, chan_best, clu_labels))
    ]
    curs.executemany(q, data)
    logger.info("Wrote %s clusters", len(data))
    return session_id


def main():
    db = mysql.connector.connect(
        user='xper_rw',
        password='up2nite',
        host='172.30.6.54',
        database='SpikeData',
        client_flags=[mysql.connector.constants.ClientFlag.LOCAL_FILES],
        allow_local_infile=True,
    )
    root = parse_args()
    write_to_db(root, db)


if __name__ == '__main__':
    main()

