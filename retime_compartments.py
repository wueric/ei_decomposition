import numpy as np

import argparse
import pickle
from typing import Dict, Tuple

from lib.ei_decomposition import generate_fourier_phase_shift_matrices

ABS_MAX_FRAC = 0.05

ALIGN_TO_SAMPLE_NUM = 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='shift waveforms and retime phase shifts so that we can compare timing within each cell')

    parser.add_argument('input_pickle_path', type=str, help='path to input pickle')
    parser.add_argument('output_pickle_path', type=str, help='path to output_pickle')

    args = parser.parse_args()

    with open(args.input_pickle_path, 'rb') as input_pfile:

        metadata_dict = pickle.load(input_pfile)
        orig_data_dict = pickle.load(input_pfile)

    # now, load the waveforms and figure out what sample shift we want
    unshifted_waveforms = orig_data_dict['waveforms']

    n_compartments, waveform_len = unshifted_waveforms.shape

    abs_waveform = np.abs(unshifted_waveforms)

    alignment_threshold = np.max(abs_waveform, axis=1) * ABS_MAX_FRAC
    exceeds_threshold_sample_num = np.zeros((n_compartments, ), dtype=np.int32)
    compartment_gt, sample_gt = np.where(abs_waveform > alignment_threshold)
    for compartment_idx in range(exceeds_threshold_sample_num.shape[0]):
        min_sample = np.min(sample_gt[compartment_gt == compartment_idx])
        exceeds_threshold_sample_num[compartment_idx] = min_sample

    # shape (n_compartments, )
    # negative is advance, positive is delay
    num_samples_delay = ALIGN_TO_SAMPLE_NUM - exceeds_threshold_sample_num

    # shape (n_compartments, n_rfft_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(num_samples_delay,
                                                                 waveform_len)

    # shape (n_compartments, n_rfft_frequencies)
    waveform_rfft = np.fft.rfft(unshifted_waveforms, axis=1)

    # shape (n_compartments, n_samples)
    waveform_shifted_td = np.fft.irfft(waveform_rfft * phase_shift_matrices, axis=1, n=waveform_len)

    orig_decomposition_dict = orig_data_dict['decomposition']
    new_decomposition_dict = {} # type: Dict[int, Tuple[np.ndarray, np.ndarray]]
    for cell_id, (amplitudes, shifts) in orig_decomposition_dict.items():

        # shape (n_channels, n_compartments)
        new_phase_shifts = num_samples_delay[None,:] + shifts[:,:]
        new_decomposition_dict[cell_id] = (amplitudes, new_phase_shifts)

    new_pickle_dict = {
        'decomposition' : new_decomposition_dict,
        'waveforms' : waveform_shifted_td,
        'mse' : orig_data_dict['mse']
    }

    with open(args.output_pickle_path, 'wb') as output_pfile:
        pickle.dump(metadata_dict, output_pfile)
        pickle.dump(new_pickle_dict, output_pfile)




