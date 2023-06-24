from pathlib import Path

import numpy as np
import pytest
import torch

from kilosort import run_kilosort, default_settings


# Use `pytest --runslow` option to include this in tests.
@pytest.mark.slow
def test_pipeline(data_directory, results_directory, saved_ops, torch_device):

    ops, st, clu, _, _, _, _, _ = run_kilosort(
        data_dir=data_directory, probe_name='neuropixPhase3B1_kilosortChanMap.mat',
        device=torch_device
        )
    st = st[:,0]  # only first column is spike times, that's all that gets saved
        
    # Check that spike times and spike cluster assignments match
    st_load = np.load(results_directory / 'spike_times.npy')
    clu_load = np.load(results_directory / 'spike_clusters.npy')
    saved_yblk = saved_ops['yblk']
    saved_dshift = saved_ops['dshift']
    saved_iKxx = saved_ops['iKxx'].to(torch_device)

    # Datashift output
    assert np.allclose(saved_yblk, ops['yblk'])
    assert np.allclose(saved_dshift, ops['dshift'])
    assert torch.allclose(saved_iKxx, ops['iKxx'])
    # Final spike/neuron readout
    # Less than 2.5% difference in spike count, 5% difference in number of units
    # TODO: Make sure these are reasonable error bounds
    spikes_error = np.abs(st.size - st_load.size)/np.max([st.size, st_load.size])
    print(f'Proportion difference in total spike count: {spikes_error}')
    print(f'Count from run_kilosort: {st.size}')
    print(f'Count from saved test results: {st_load.size}')
    assert spikes_error <= 0.025

    n = np.unique(clu).size
    n_load = np.unique(clu_load).size
    unit_count_error = np.abs(n - n_load)/np.max([n, n_load])
    print(f'Proportion difference in number of units: {unit_count_error}')
    print(f'Number of units from run_kilosort: {n}')
    print(f'Number of units from saved test results: {n_load}')
    assert unit_count_error <= 0.05
