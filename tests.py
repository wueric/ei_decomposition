import visionloader as vl
import numpy as np

import torch

import argparse

from typing import List, Dict, Tuple, Callable

import pickle

import tqdm

from lib.ei_decomposition import fast_time_shifts_and_amplitudes_shared_shifts, bspline_upsample_waveforms, \
    coarse_to_fine_time_shifts_and_amplitudes

if __name__ == '__main__':
    device = torch.device('cuda')

    # for now, don't bother with argparse since we still don't have an automatic way
    # to pick canonical waveforms
    print("Loading data")
    dataset = vl.load_vision_data('/Volumes/Lab/Users/ericwu/yass-ei/2018-03-01-0/data001',
                                  'data001',
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()

    # hardcoded test data
    example_cell = 616
    dendritic_electrode = 78
    somatic_electrode = 61
    axonic_electrode = 300

    cell_to_fit = 616

    ei_example = dataset.get_ei_for_cell(example_cell).ei
    ei_to_fit = dataset.get_ei_for_cell(cell_to_fit).ei

    normalized_dendritic = ei_example[dendritic_electrode, :] / np.linalg.norm(ei_example[dendritic_electrode, :])
    normalized_somatic = ei_example[somatic_electrode, :] / np.linalg.norm(ei_example[somatic_electrode, :])
    normalized_axonic = ei_example[axonic_electrode, :] / np.linalg.norm(ei_example[axonic_electrode, :])

    upsampled_ei = bspline_upsample_waveforms(ei_to_fit, 5)
    padded_ei = np.pad(upsampled_ei,
                       [(0, 0), (100, 100)],
                       mode='constant')

    basis_stacked = np.stack([normalized_axonic, normalized_somatic, normalized_dendritic], axis=0)
    basis_upsampled = bspline_upsample_waveforms(basis_stacked, 5)

    padded_upsampled = np.pad(basis_upsampled,
                              [(0, 0), (100, 100)],
                              mode='constant')

    n_canonical_waveforms, n_timepoints = padded_upsampled.shape

    low_shift, high_shift = (-100, 100)
    shift_steps = np.r_[low_shift:high_shift:5]
    mg = np.stack(np.meshgrid(*[shift_steps for _ in range(n_canonical_waveforms)]), axis=0)
    valid_phase_shifts_matrix = mg.reshape((n_canonical_waveforms, -1))

    random_amplitudes = np.random.uniform(0, 10,
                                          size=(512, valid_phase_shifts_matrix.shape[1], n_canonical_waveforms))

    print("Running fit")
    amplitudes, phases = coarse_to_fine_time_shifts_and_amplitudes(
        np.fft.rfft(padded_ei, axis=1),
        np.fft.rfft(padded_upsampled, axis=1),
        n_timepoints,
        (-100, 100),
        5,
        5,
        2,
        device,
        l1_regularization_lambda=7.5e-2,
        max_batch_size=2048
    )

    save_dict = {
        'basis': basis_upsampled,
        'shifts': phases,
        'weights': amplitudes
    }

    with open('test{0}.p'.format(cell_to_fit), 'wb') as pfile:
        pickle.dump(save_dict, pfile)
