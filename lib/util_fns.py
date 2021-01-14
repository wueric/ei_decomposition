from collections import namedtuple

import numpy as np
from scipy import interpolate as interpolate

from typing import List, Dict, Union, Tuple, Sequence

EIDecomposition = namedtuple('EIDecomposition', ['amplitude', 'delay'])

def bspline_upsample_waveforms(waveforms: np.ndarray,
                               upsample_factor: int) -> np.ndarray:
    '''

    :param waveform: shape (n_waveforms, n_samples) original waveforms
    :return: upsampled waveforms, shape (n_waveforms, n_shifts, n_samples)
    '''

    n_waveforms, n_orig_samples = waveforms.shape
    upsampled = np.zeros((n_waveforms, n_orig_samples * upsample_factor))

    orig_time_samples = np.r_[0:n_orig_samples]  # shape (n_samples, )
    upsample_timepoints = np.linspace(0, n_orig_samples, n_orig_samples * upsample_factor)

    for idx in range(n_waveforms):
        orig_waveform_1d = waveforms[idx, :]
        bspline = interpolate.splrep(orig_time_samples, orig_waveform_1d)

        waveform_shifted = interpolate.splev(upsample_timepoints, bspline)
        # shape (n_shifts, n_orig_samples)
        upsampled[idx, :] = waveform_shifted

    return upsampled


def generate_fourier_phase_shift_matrices(sample_delays: np.ndarray,
                                          n_frequencies: int) -> np.ndarray:
    '''
    Generate complex-valued sample delay matrices corresponding
        to integer sample delays

    When we take the Fourier transform of the canonical waveforms, which
        have shape (n_canonical_waveforms, n_samples), we get a transform
        vector with shape (n_canonical_waveforms, n_frequencies = n_samples)

    We want to multiply element-wise each entry in the Fourier transform by
        the corresponding phase shift e^{-j 2 * pi * tau * f / F} and broadcast
        accordingly to account of the different number of shifts

    :param sample_delays: integer, shape (n_observations, n_canonical_waveforms)
    :param n_frequencies: number of frequencies, equal to number of samples
    :return: phase delay matrix, complex valued,
        shape (n_observations, n_canonical_waveforms, n_frequencies)
    '''
    sample_delays_slice = [slice(None) for _ in range(sample_delays.ndim)]
    sample_delays_slice.append(None)
    sample_delays_slice = tuple(sample_delays_slice)

    phase_radians_slice = [None for _ in range(sample_delays.ndim)]
    phase_radians_slice.append(slice(None))
    phase_radians_slice = tuple(phase_radians_slice)

    # this has value f/F
    frequencies = np.fft.rfftfreq(n_frequencies)  # shape (n_frequencies, )

    # this is 2 * pi * f / F
    phase_radians = 2 * np.pi * frequencies  # shape (n_frequencies, )

    # this is -j * 2 * pi * tau * f / F
    complex_exponential_argument = -1j * sample_delays[sample_delays_slice] * phase_radians[phase_radians_slice]

    return np.exp(complex_exponential_argument)


def debug_evaluate_error(observed_ft: np.ndarray,
                         fit_real_amplitudes: np.ndarray,
                         canonical_waveform_ft: np.ndarray,
                         time_shifts: np.ndarray,
                         n_true_frequencies: int) -> float:
    '''

    :param observed_ft: Fourier transform of observed data, complex-valued,
        shape (n_observations, n_frequencies)
    :param fit_real_amplitudes: real-valued scale amplitude of each canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param canonical_waveform_ft: Fourier transform of canonical waveforms, complex-valued,
        shape (n_canonical_waveforms, n_frequencies)
    :param time_shifts: Time shifts required for each canonical waveform to fit each observation,
        shape (n_observations, n_canonical_waveforms)
    :param n_true_frequencies: int, the number of normal FFT frequencies (not the number of
        rFFT frequencies)
    :return: MSE error, real valued time domain power, imaginary valued time domain power
    '''
    n_observations, _ = observed_ft.shape

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    time_shift_matrices = generate_fourier_phase_shift_matrices(time_shifts,
                                                                n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    shifted_no_scale_ft = canonical_waveform_ft[None, :, :] * time_shift_matrices

    model_ft = np.squeeze(fit_real_amplitudes[:, None, :] @ shifted_no_scale_ft, axis=1)

    diff = observed_ft - model_ft
    errors = np.linalg.norm(diff, axis=1)
    mean_error = np.mean(errors)  # type: float

    return mean_error


def pack_significant_electrodes_into_matrix(eis_by_cell_id: Dict[int, np.ndarray],
                                            cell_order: List[int],
                                            snr_abs_threshold: Union[float, int]) \
        -> Tuple[np.ndarray, Dict[int, Tuple[slice, Sequence[int]]]]:
    matrix_indices_by_cell_id = {}  # type: Dict[int, Tuple[slice, Sequence[int]]]
    to_concat = []  # type: List[np.ndarray]

    cat_low = 0  # type: int
    for cell_id in cell_order:
        ei_mat = eis_by_cell_id[cell_id]

        chans_sufficient_magnitude = np.max(np.abs(ei_mat), axis=1) > snr_abs_threshold
        n_chans_sufficient = np.sum(chans_sufficient_magnitude)  # type: int

        readback_slice = slice(cat_low, cat_low + n_chans_sufficient)
        matrix_indices_by_cell_id[cell_id] = (readback_slice, chans_sufficient_magnitude)
        to_concat.append(ei_mat[chans_sufficient_magnitude, :])

        cat_low += n_chans_sufficient

    ei_data_mat = np.concatenate(to_concat, axis=0)

    return ei_data_mat, matrix_indices_by_cell_id


def unpack_amplitudes_and_phases_into_ei_shape(packed_amplitude_matrix: np.ndarray,
                                               packed_phase_matrix: np.ndarray,
                                               orig_ei_by_cell_id : Dict[int, np.ndarray],
                                               cell_order: List[int],
                                               unpack_slice_dict: Dict[int, Tuple[slice, Sequence[int]]]) \
    -> Dict[int, EIDecomposition]:

    n_observations, n_basis_vectors = packed_amplitude_matrix.shape

    result_dict = {} # type: Dict[int, EIDecomposition]
    for cell_id in cell_order:
        orig_ei_mat = orig_ei_by_cell_id[cell_id]
        n_channels = orig_ei_mat.shape[0]

        slice_section, sufficient_snr = unpack_slice_dict[cell_id]

        amplitude_matrix = np.zeros((n_channels, n_basis_vectors), dtype=np.float32)
        amplitude_matrix[sufficient_snr, :] = packed_amplitude_matrix[slice_section, :]

        delay_vector = np.zeros((n_channels, n_basis_vectors), dtype=np.int32)
        delay_vector[sufficient_snr, :] = packed_phase_matrix[slice_section, :]

        result_dict[cell_id] = (amplitude_matrix, delay_vector)

    return result_dict

