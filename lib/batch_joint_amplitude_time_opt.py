from typing import Optional, Tuple, Callable, List

import numpy as np
import torch
import tqdm


def batched_build_at_a_matrix(ft_canonical: np.ndarray,
                              valid_phase_shifts: np.ndarray,
                              n_true_frequencies: int) -> np.ndarray:
    '''
    Computes the matrix A^T A for all desired time shifts of the basis waveforms in A, assuming that
        the time shifts are shared among the observed data waveforms, and so there is no need to calculate
        separate values for A^T A for each observed data waveform

    Assumes that the basis waveforms are not shared along the batch dimension, and so there are batch
        distinct sets of basis waveforms

    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    :param valid_phase_shifts: integer array, shape (batch, n_canonical_waveforms, n_valid_phase_shifts)
        Note that dim0 could be 1 in this case, corresponding to shared shifts
    :param n_true_frequencies: int
    :return: batched A^T A matrix, shape (batch, n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    '''

    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies) ->
    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_timepoints)
    circular_corr_td = np.fft.irfft(ft_canonical[:, :, None, :] * np.conjugate(ft_canonical[:, None, :, :]),
                                    n=n_true_frequencies,
                                    axis=3)

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_timepoints), axis 2 corresponds to the shifted waveforms
    # relative to axis 1 fixed waveforms
    # (..., i,j,t)^{th} entry corresponds to cross correlation of i^{th} canonical waveform with j^{th} canonical waveform
    #   that has been delayed by t samples
    # This means that circular_conv_td is not symmetric for dims (1, 2)

    # now we have to build the batched at_a matrix by grabbing the relevant pieces
    # not so straightforward, since we care about relative timing instead of absolute timing

    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_valid_phase_shifts)
    # or shape (1, n_canonical_waveforms, n_canonical_waveforms, n_valid_phase_shifts)
    relative_shifts = valid_phase_shifts[:, None, :, :] - valid_phase_shifts[:, :, None, :]

    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_valid_phase_shifts)
    taken_piece = np.take_along_axis(circular_corr_td, relative_shifts, axis=3)

    # shape (batch, n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix_np = taken_piece.transpose((0, 3, 1, 2))

    return at_a_matrix_np


def batched_build_at_b_vector(observed_ft: np.ndarray,
                              ft_canonical: np.ndarray,
                              valid_phase_shifts: np.ndarray,
                              n_true_frequencies: int) -> np.ndarray:
    '''
    Computes the vector A^T b for all desired time shifts of the basis waveforms in A, assuming that
        the time shifts are shared among the observed data waveforms, and so there is no need to calculate
        separate values for A^T A for each observed data waveform

    Assumes that the basis waveforms are not shared along the batch dimension, and so there are
        batch sets of basis waveforms

    :param observed_ft: shape (batch, n_observations, n_rfft_frequencies), complex-valued, RFFT coefficients
        of the observations
    :param ft_canonical: shape (batch, n_basis_waveforms, n_rfft_frequencies), complex-valued, RFFT coefficients
        of the basis waveforms
    :param valid_phase_shifts: integer array, shape (batch, n_basis_waveforms, n_valid_phase_shifts) corresponding
        to the time shifts of the basis waveforms in A
    :param n_true_frequencies: Number of real frequencies = number of timepoints, so that we can
        compute the iRFFT without losing or gaining coefficients
    :return:
    '''
    # shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies) ->
    # shape (batch, n_observations, n_canonical_waveforms, n_timepoints)
    data_circ_conv_td = np.fft.irfft(observed_ft[:, :, None, :] * np.conjugate(ft_canonical[:, None, :, :]),
                                     n=n_true_frequencies,
                                     axis=3)
    # The (..., i,j,t)^{th} entry corresponds to cross correlation of the i^{th} data waveform with the j^{th} canonical
    #   waveform that has been delayed by t samples

    # we have to build A^T b from this matrix
    # shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
    at_b_perm = np.take_along_axis(data_circ_conv_td, valid_phase_shifts[:, None, :, :], axis=3)

    # shape (batch, n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_np = at_b_perm.transpose((0, 1, 3, 2))

    return at_b_np


def batched_build_unshared_at_a_matrix(ft_canonical: np.ndarray,
                                       unshared_phase_shifts: np.ndarray,
                                       n_true_frequencies: int) -> np.ndarray:
    '''
    Computes the matrix A^T A for all desired time shifts of the basis waveforms in A, assuming that
        the time shifts are not shared among observed waveforms

    Assumes that the basis waveforms are not shared along the batch dimension, and so there are batch
        distinct sets of basis waveforms

    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer, shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
    :param n_true_frequencies: integer, number of FFT frequencies = number of timepoints, so that we can compute
        the iRFFT without adding or losing coefficients
    :return: np.ndarray, shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
        last two dimensions are A^T A matrices
    '''

    batch, n_observations, n_canonical_waveforms, n_phase_shifts = unshared_phase_shifts.shape

    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_rfft_freqs) ->
    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_timepoints)
    circular_corr_td = np.fft.irfft(ft_canonical[:, :, None, :] * np.conjugate(ft_canonical[:, None, :, :]),
                                    n=n_true_frequencies,
                                    axis=3)

    # shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix_np = np.zeros((batch, n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms),
                              dtype=np.float32)

    for j in range(n_canonical_waveforms):
        # shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
        unshared_relative_shift = unshared_phase_shifts - unshared_phase_shifts[:, :, j, :][:, :, None, :]

        # shape (batch, n_canonical_waveforms, n_phase_shifts, n_observations)
        # -> shape (batch, n_canonical_waveforms, n_observations * n_phase_shifts)
        unshared_relative_shift_flat = unshared_relative_shift.transpose(0, 2, 3, 1).reshape(
            (batch, n_canonical_waveforms, -1))

        # shape (batch, n_canonical_waveforms, n_observations * n_phase_shifts)
        taken_piece_flat = np.take_along_axis(circular_corr_td[:, j, :, :], unshared_relative_shift_flat, axis=2)

        # shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
        taken_piece = taken_piece_flat.reshape(batch, n_canonical_waveforms, n_phase_shifts, n_observations).transpose(
            0, 3, 1, 2)

        # shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms)
        at_a_matrix_np[:, :, :, :, j] = taken_piece.transpose((0, 1, 3, 2))

    return at_a_matrix_np


def batched_build_unshared_at_b_vector(observed_ft: np.ndarray,
                                       ft_canonical: np.ndarray,
                                       unshared_phase_shifts: np.ndarray,
                                       n_true_frequencies: int) -> np.ndarray:
    '''
    Computes the matrix A^T b for all desired time shifts of the basis waveforms in A, assuming that
        the time shifts are not shared among observed waveforms

    Assumes that the basis waveforms are not shared along the batch dimension, and so there are batch
        distinct sets of basis waveforms

    :param observed_ft: observed waveforms in Fourier domain, complex valued,
            shape (batch, n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer,
            shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
    :param n_true_frequencies:
    :return: np.ndarray, shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms)
    '''

    # shape (batch, n_observations, n_canonical_waveforms, n_timepoints)
    data_circ_conv_td = np.fft.irfft(observed_ft[:, :, None, :] * np.conjugate(ft_canonical[:, None, :, :]),
                                     n=n_true_frequencies,
                                     axis=3)

    # The (i,j,t)^{th} entry corresponds to cross correlation of the i^{th} data waveform with the j^{th} canonical
    #   waveform that has been delayed by t samples

    # we have to build A^T b from this matrix
    # shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
    at_b_perm = np.take_along_axis(data_circ_conv_td, unshared_phase_shifts, axis=3)

    # shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms)
    at_b_np = at_b_perm.transpose((0, 1, 3, 2))

    return at_b_np
