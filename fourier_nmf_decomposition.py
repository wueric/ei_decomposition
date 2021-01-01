import visionloader as vl
import numpy as np
import scipy.interpolate as interpolate

import torch

import argparse

from typing import List, Dict, Tuple, Sequence, Optional, Union

import pickle

import tqdm

from collections import namedtuple


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


def nonnegative_least_squares_optimize_amplitudes(observation_matrix_np: np.ndarray,
                                                  amplitude_matrix_real_np: np.ndarray,
                                                  shifted_canonical_basis_np: np.ndarray,
                                                  device: torch.device,
                                                  l1_regularization_lambda: Optional[float] = None,
                                                  max_iter=100,
                                                  converge_epsilon=1e-3) \
        -> np.ndarray:
    '''
    Performs the nonnegative least squares optimization to fit the real-valued amplitude A matrix
    Algorithm used is proximal gradient descent over nonnegative orthant
    Eigenvalues for convergence and step size determination

    We have to solve N different optimization problems, where N is the
        number of data observations

    :param observation_matrix_np: observed waveforms in time domain,
        shape (n_observations, n_timepoints)
    :param amplitude_matrix_real_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param shifted_canonical_basis_np: shifted canonical waveforms for each observation, real-valued. Delay and
        inverse Fourier transform taken care of
        shape (n_observations, n_canonical_waveforms, n_timepoints)
    :param device:
    :return: optimized real amplitude matrix, shape (n_observations, n_canonical_waveforms)
    '''

    # put everything on GPU
    # shape (n_observations, n_timepoints)
    observe_mat = torch.tensor(observation_matrix_np, dtype=torch.float32, device=device)

    # shape (n_timepoints, n_observations)
    observe_mat_t = observe_mat.permute((1, 0))

    # shape (n_observations, n_canonical_waveforms)
    amplitudes = torch.tensor(amplitude_matrix_real_np, dtype=torch.float32, device=device)

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    shifted_basis = torch.tensor(shifted_canonical_basis_np, dtype=torch.float32, device=device)

    # shape (n_observations, n_timepoints, n_canonical_waveforms)
    shifted_basis_t = shifted_basis.permute((0, 2, 1))

    at_a_matrix = shifted_basis @ shifted_basis_t
    # shape (n_observations, n_canonical_waveforms, n_canonical_waveforms)

    # Pytorch eigenvalues suck terribly, so perform the eigenvalue decomposition
    # on CPU with numpy
    at_a_numpy = at_a_matrix.cpu().numpy()
    eigenvalues_np, _ = np.linalg.eigh(at_a_numpy)
    eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)

    # eigenvalues has shape (batch, n_waveforms)
    max_eigenvalue, _ = torch.max(eigenvalues, dim=1)  # shape (n_observations, )
    min_eigenvalue, _ = torch.min(eigenvalues, dim=1)  # shape (n_observations, )

    convergence_factor = 0.5 * (max_eigenvalue - min_eigenvalue)  # shape (n_observations, )

    # boundaries for the step size
    # we have to have 0 < step_size <= 1/L where L is the largest eigenvalue
    # we make step_size smaller to be safe
    step_size = 1.0 / (2 * max_eigenvalue)  # has shape (n_observations, )

    ax_minus_b = shifted_basis_t @ amplitudes[:, :, None] - observe_mat[:, :, None]
    # shape (n_observations, n_timepoints, 1)

    # (n_observations, n_canonical_waveforms, n_timepoints) x (n_observations, n_timepoints, 1)
    # = (n_observations, n_canonical_waveforms, 1) -> (n_observations, n_canonical_waveforms)
    gradient = torch.squeeze(shifted_basis @ ax_minus_b, -1)
    if l1_regularization_lambda is not None:
        gradient += l1_regularization_lambda

    for iter_count in range(max_iter):

        # shape (n_observations, n_canonical_waveforms)
        next_amplitudes = torch.clamp(amplitudes - step_size[:, None] * gradient,
                                      min=0.0)

        ax_minus_b = shifted_basis_t @ next_amplitudes[:, :, None] - observe_mat[:, :, None]
        # shape (n_observations, n_timepoints, 1)

        # (n_observations, n_canonical_waveforms, n_timepoints) x (n_observations, n_timepoints, 1)
        # = (n_observations, n_canonical_waveforms, 1) -> (n_observations, n_canonical_waveforms)
        gradient = torch.squeeze(shifted_basis @ ax_minus_b, -1)

        if l1_regularization_lambda is not None:
            gradient += l1_regularization_lambda

        step_distance = next_amplitudes - amplitudes  # shape (n_observations, n_canonical_waveforms)
        step_distance_square_mag = torch.sum(step_distance * step_distance,
                                             dim=1)  # shape (n_observations, )
        convergence_bound = convergence_factor * step_distance_square_mag  # shape (n_observations, )
        worst_bound = torch.max(convergence_bound).item()

        amplitudes = next_amplitudes

        if worst_bound < converge_epsilon:
            break

    return amplitudes.cpu().numpy()


def fourier_complex_least_squares_optimize_waveforms3(amplitude_matrix_real_np: np.ndarray,
                                                      phase_delays_np: np.ndarray,
                                                      ft_complex_observations_np: np.ndarray,
                                                      n_true_frequencies: int,
                                                      device: torch.device,
                                                      sobolev_lambda: Optional[float] = None) -> np.ndarray:
    '''

    :param amplitude_matrix_real_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param phase_delays_np: integer sample delays for each canonical waveform, for each observation
        shape (n_observations, n_canonical_waveforms)
    :param ft_complex_observations_np: complex-valued Fourier transform of the observed data,
        shape (n_observations, n_rfft_frequencies)
    :param n_true_frequencies : int, number of frequencies = n_samples for the normal FFT
        (not the number of rFFT frequencies)
    :param device:
    :return: tuple of real component, imaginary component of canonical waveform Fourier transform
        each has shape (n_canonical_waveforms, n_rfft_frequencies)
    '''

    n_observations, n_canonical_waveforms = amplitude_matrix_real_np.shape
    _, n_rfft_frequencies = ft_complex_observations_np.shape

    # real valued, shape (n_observations, n_canonical_waveforms)
    amplitude_mat_torch = torch.tensor(amplitude_matrix_real_np, dtype=torch.float32, device=device)

    # shape (n_observations, n_rfft_frequencies)
    real_observe_ft_torch = torch.tensor(ft_complex_observations_np.real, dtype=torch.float32, device=device)
    imag_observe_ft_torch = torch.tensor(ft_complex_observations_np.imag, dtype=torch.float32, device=device)

    # complex-valued, shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    complex_phase_matrix = generate_fourier_phase_shift_matrices(phase_delays_np,
                                                                 n_true_frequencies)

    real_phase_mat_torch = torch.tensor(complex_phase_matrix.real, dtype=torch.float32, device=device)
    imag_phase_mat_torch = torch.tensor(complex_phase_matrix.imag, dtype=torch.float32, device=device)

    # shape (n_observations, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    # the (l^th, k^{th}, j^{th}, f^{th}) entry is real{P}^{(l)}_{f,k} * real{P}^{(l)}_{f,j}
    ####################### (l, k, None, f) ################# (l, None, j, f) ###########
    real_real_phase = real_phase_mat_torch[:, :, None, :] * real_phase_mat_torch[:, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is imag{P}^{(l)}_{f,k} * imag{P}^{(l)}_{f,j}
    imag_imag_phase = imag_phase_mat_torch[:, :, None, :] * imag_phase_mat_torch[:, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is imag{P}^{(l)}_{f,k} * real{P}^{(l)}_{f,j}
    imag_real_phase = imag_phase_mat_torch[:, :, None, :] * real_phase_mat_torch[:, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is real{P}^{(l)}_{f,k} * imag{P}^{(l)}_{f,j}
    real_imag_phase = real_phase_mat_torch[:, :, None, :] * imag_phase_mat_torch[:, None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_canonical_waveforms)
    # the (l^{th}, k^{th}, j^{th}) entry is A_{k,l} A_{j,l}
    amplitude_outer_product = amplitude_mat_torch[:, :, None] * amplitude_mat_torch[:, None, :]

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    # each different row in dim0 corresponds to a different equation
    eq1_group_real_coeff = torch.sum((real_real_phase + imag_imag_phase) * amplitude_outer_product[:, :, :, None],
                                     dim=0)
    eq1_group_imag_coeff = torch.sum((imag_real_phase - real_imag_phase) * amplitude_outer_product[:, :, :, None],
                                     dim=0)
    # shape (n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_coeff = torch.cat([eq1_group_real_coeff, eq1_group_imag_coeff], dim=1)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    ################# (l, k, f) ############### (l, k, None) ########################### (l, None, f)
    eq1_rhs_re = real_phase_mat_torch[:, :, :] * amplitude_mat_torch[:, :, None] * real_observe_ft_torch[:, None, :]
    eq1_rhs_im = imag_phase_mat_torch[:, :, :] * amplitude_mat_torch[:, :, None] * imag_observe_ft_torch[:, None, :]

    # shape (n_canonical_waveforms, n_frequencies)
    eq1_rhs = torch.sum(eq1_rhs_re + eq1_rhs_im, dim=0)

    # shape (n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    eq2_group_coeff = torch.cat([-1.0 * eq1_group_imag_coeff, eq1_group_real_coeff], dim=1)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs_p = real_phase_mat_torch[:, :, :] * amplitude_mat_torch[:, :, None] * imag_observe_ft_torch[:, None, :]
    eq2_rhs_m = imag_phase_mat_torch[:, :, :] * amplitude_mat_torch[:, :, None] * real_observe_ft_torch[:, None, :]

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs = torch.sum(eq2_rhs_p - eq2_rhs_m, dim=0)

    # interleave equations from groups 1 and 2 so that the sobolev regularization thing
    # can be represented as addition along the diagonal of the matrix
    joint_coeff_permute = torch.empty((2 * n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies),
                                      dtype=torch.float32,
                                      device=device)
    joint_coeff_permute[0::2, :, :] = eq1_group_coeff
    joint_coeff_permute[1::2, :, :] = eq2_group_coeff

    # shape (n_rfft_frequencies, 2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
    joint_coeff = joint_coeff_permute.permute(2, 0, 1)

    # now deal with the Sobolev regularization if specified
    if sobolev_lambda is not None:

        frequencies = np.fft.rfftfreq(n_true_frequencies)  # shape (n_rfft_frequencies, )

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
        canonical_waveforms_identity = np.eye(2 * n_canonical_waveforms) * 2 * np.pi

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms,, n_rfft_frequencies)
        canonical_waveform_freq_diag = canonical_waveforms_identity[:, :, None] * frequencies[None, None, :]

        diagonal_regularize = 2 * sobolev_lambda * (1 - np.cos(canonical_waveform_freq_diag))

        diagonal_regularize_torch = torch.tensor(diagonal_regularize, dtype=torch.float32, device=device)

        joint_coeff = joint_coeff + diagonal_regularize_torch

    joint_rhs_permute = torch.empty((2 * n_canonical_waveforms, n_rfft_frequencies),
                                    dtype=torch.float32,
                                    device=device)
    joint_rhs_permute[0::2, :] = eq1_rhs
    joint_rhs_permute[1::2, :] = eq2_rhs

    # shape (n_rfft_frequencies, 2 * n_canonical_waveforms)
    joint_rhs = joint_rhs_permute.permute(1, 0)

    # soln has shape (n_rfft_frequencies, 2 * n_canonical_waveforms)
    soln, _ = torch.solve(joint_rhs[:, :, None], joint_coeff)

    # shape (2 * n_canonical_waveforms, n_rfft_frequencies)
    soln_perm = soln.squeeze(2).permute(1, 0)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    soln_real_seg = soln_perm[0::2, :].cpu().numpy()
    soln_imag_seg = soln_perm[1::2, :].cpu().numpy()

    return soln_real_seg + 1j * soln_imag_seg


def parallel_combinatorial_template_match(observation_matrix_np: np.ndarray,
                                          real_amplitude_matrix_np: np.ndarray,
                                          possible_shifted_canonical_waveforms_np: np.ndarray):
    '''

    :param observation_matrix_np:
    :param real_amplitude_matrix_np:
    :param possible_shifted_canonical_waveforms_np:
    :return:
    '''

    raise NotImplementedError


def torch_fit_integer_shifts_all_but_one_template_match(observed_ft: np.ndarray,
                                                        real_amplitude_matrix_np: np.ndarray,
                                                        ft_canonical: np.ndarray,
                                                        previous_shifts_integer: np.ndarray,
                                                        valid_phase_shifts: np.ndarray,
                                                        n_true_frequencies: int,
                                                        device: torch.device) -> np.ndarray:
    '''
        Crude approximate all-but-one template match.

        Description of algorithm:

            For the i^{th} canonical waveform:
                Deconvolve all j != i canonical waveforms from the observed data
                    using fixed amplitude and shift from previous iterations
                Find the optimal shift for the i^{th} waveform with a zero-padded correlation

        :param observation_matrix_np: observed waveforms in Fourier domain, complex valued
            shape (n_observations, n_rfft_frequencies)
        :param real_amplitude_matrix_np: real-valued amplitudes for each observation, each shifted canonical waveform,
            shape (n_observations, n_canonical_waveforms)
        :param ft_canonical: Fourier transform of unshifted canonical waveforms, complex valued
            shape (n_canonical_waveforms, n_rfft_frequencies)
        :param previous_shifts_integer: previous canonical waveform shifts, shape (n_observations, n_canonical_waveforms)
        :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
        :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
        :param device: torch device
        :return: optimal timeshifts, shape (n_observations, n_canonical_waveforms)
        '''

    n_observations, _ = observed_ft.shape
    _, n_canonical_waveforms = real_amplitude_matrix_np.shape

    # component-wise complex representation, shape (2, n_observations, n_rfft_frequencies)
    observed_ft_torch = torch.tensor(np.stack([observed_ft.real, observed_ft.imag], axis=0),
                                     dtype=torch.float32,
                                     device=device)

    # real-valued, shape (n_observations, n_canonical_waveforms)
    real_amplitude_matrix_torch = torch.tensor(real_amplitude_matrix_np, dtype=torch.float32, device=device)

    # component-wise complex representation, shape (2, n_canonical_waveforms, n_rfft_frequencies)
    ft_canonical_torch = torch.tensor(np.stack([ft_canonical.real, ft_canonical.imag],
                                               axis=0),
                                      dtype=torch.float32,
                                      device=device)

    # complex-valued,
    # shape (n_valid_phase_shifts, n_rfft_frequencies)
    phaseshift_all_allowed_matrices = generate_fourier_phase_shift_matrices(valid_phase_shifts,
                                                                            n_true_frequencies)

    # component-wise complex representation, shape (2, n_valid_phase_shifts, n_rfft_frequencies)
    phaseshift_all_allowed_torch = torch.tensor(np.stack([phaseshift_all_allowed_matrices.real,
                                                          phaseshift_all_allowed_matrices.imag],
                                                         axis=0),
                                                dtype=torch.float32,
                                                device=device)

    # complex-valued,
    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(previous_shifts_integer,
                                                                 n_true_frequencies)

    # component-wise complex representation, shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shift_torch = torch.tensor(np.stack([phase_shift_matrices.real, phase_shift_matrices.imag],
                                              axis=0),
                                     dtype=torch.float32,
                                     device=device)

    # complex-valued, multiplication (x + ai) (y + bi) = (xy - ab) + (xb + ay)i
    # shape (2, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shifted_canonical_ft_torch = torch.stack([
        phase_shift_torch[0, :, :, :] * ft_canonical_torch[0, None, :, :] - \
        phase_shift_torch[1, :, :, :] * ft_canonical_torch[1, None, :, :],

        phase_shift_torch[1, :, :, :] * ft_canonical_torch[0, None, :, :] + \
        phase_shift_torch[0, :, :, :] * ft_canonical_torch[1, None, :, :]
    ], dim=0)

    # real_amplitude_matrix_torch is (n_observations, n_canonical_waveforms)
    # real * complex multiplication
    # shape (2, n_observations, n_canonical_waveforms, n_frequencies)
    scaled_shifted_canonical_ft_torch = real_amplitude_matrix_torch[None, :, :, None] * phase_shifted_canonical_ft_torch

    # shape (2, n_observations, n_frequencies)
    canonical_reconstruction_ft = torch.sum(scaled_shifted_canonical_ft_torch, dim=2)

    # shape (2, n_observations, n_frequencies)
    deconvolved_ft_torch = observed_ft_torch - canonical_reconstruction_ft

    # shape (n_observations, n_canonical_waveforms)
    output_timeshifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)

    for i in range(n_canonical_waveforms):
        # shape (2, n_observations, n_frequencies)
        all_but_one_resid_ft = deconvolved_ft_torch + scaled_shifted_canonical_ft_torch[:, :, i, :]

        # real_amplitude_matrix_torch is (n_observations, n_canonical_waveforms)
        # ft_canonical_torch is (2, n_canonical_waveforms, n_rfft_frequencies)
        # shape (2, n_observations, n_frequencies)
        possible_ft_scaled = real_amplitude_matrix_torch[None, :, i, None] * ft_canonical_torch[:, None, i, :]

        # shape (2, n_observations, n_allowed_shifts, n_frequencies)
        possible_shifts_ft_scaled = torch.stack([
            possible_ft_scaled[0, :, None, :] * phaseshift_all_allowed_torch[0, None, :, :] - \
            possible_ft_scaled[1, :, None, :] * phaseshift_all_allowed_torch[1, None, :, :],

            possible_ft_scaled[0, :, None, :] * phaseshift_all_allowed_torch[1, None, :, :] + \
            possible_ft_scaled[1, :, None, :] * phaseshift_all_allowed_torch[0, None, :, :]
        ], dim=0)

        # shape (2, n_observations, n_allowed_shifts, n_frequencies)
        all_possible_subtracted = all_but_one_resid_ft[:, :, None, :] - possible_shifts_ft_scaled

        # shape (n_observations, n_allowed_shifts)
        magnitudes = torch.sum(all_possible_subtracted * all_possible_subtracted, dim=(0, 3))

        _, min_shift_indices = torch.min(magnitudes, dim=1)
        best_shifts_idx = min_shift_indices.cpu().numpy()

        best_shifts = valid_phase_shifts[best_shifts_idx]

        output_timeshifts[:, i] = best_shifts

    return output_timeshifts


def fit_shifts_all_but_one_template_match(observed_ft: np.ndarray,
                                          real_amplitude_matrix_np: np.ndarray,
                                          ft_canonical: np.ndarray,
                                          previous_shifts_integer: np.ndarray,
                                          valid_phase_shifts: np.ndarray,
                                          n_true_frequencies: int) -> np.ndarray:
    '''
    Crude approximate all-but-one template match.

    Description of algorithm:

        For the i^{th} canonical waveform:
            Deconvolve all j != i canonical waveforms from the observed data
                using fixed amplitude and shift from previous iterations
            Find the optimal shift for the i^{th} waveform with a zero-padded correlation

    :param observation_matrix_np: observed waveforms in time domain,
        shape (n_observations, n_frequencies)
    :param real_amplitude_matrix_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param ft_canonical: unshifted canonical waveforms,
        shape (n_canonical_waveforms, n_frequencies)
    :param previous_shifts_integer: previous canonical waveform shifts, shape (n_observations, n_canonical_waveforms)
    :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
    :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
    :return: timeshifts, shape (n_observations, n_canonical_waveforms)
    '''
    n_observations, _ = observed_ft.shape
    _, n_canonical_waveforms = real_amplitude_matrix_np.shape

    # shape (n_valid_phase_shifts, n_frequencies)
    phaseshift_all_allowed_matrices = generate_fourier_phase_shift_matrices(valid_phase_shifts,
                                                                            n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(previous_shifts_integer,
                                                                 n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shifted_canonical_ft = phase_shift_matrices * ft_canonical[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    scaled_shifted_canonical_ft = real_amplitude_matrix_np[:, :, None] * phase_shifted_canonical_ft

    # shape (n_canonical_waveforms, n_frequencies)
    deconvolved_ft = observed_ft - np.sum(scaled_shifted_canonical_ft, axis=1)

    # shape (n_observations, n_canonical_waveforms)
    output_timeshifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    for i in range(n_canonical_waveforms):
        # shape (n_canonical_waveforms, n_frequencies)
        all_but_one_resid_ft = deconvolved_ft + scaled_shifted_canonical_ft[:, i, :]

        # shape (n_observations, n_frequencies)
        possible_ft_scaled = real_amplitude_matrix_np[:, i, None] * ft_canonical[None, i, :]

        # shape (n_observations, n_allowed_shifts, n_frequencies)
        possible_shifts_ft_scaled = possible_ft_scaled[:, None, :] * phaseshift_all_allowed_matrices[None, :, :]

        # shape (n_observations, n_allowed_shifts, n_frequencies)
        all_possible_subtracted = all_but_one_resid_ft[:, None, :] - possible_shifts_ft_scaled

        # shape (n_observations, n_allowed_shifts)
        magnitudes = np.linalg.norm(all_possible_subtracted, axis=2)

        # shape (n_observations, )
        best_shifts_idx = np.argmin(magnitudes, axis=1)
        best_shifts = valid_phase_shifts[best_shifts_idx]

        output_timeshifts[:, i] = best_shifts

    return output_timeshifts


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


def shifted_fourier_nmf(waveform_data_matrix: np.ndarray,
                        n_canonical_waveforms: int,
                        valid_sample_shifts: np.ndarray,
                        n_iter: int,
                        device: torch.device,
                        l1_regularization_lambda: Optional[float] = None,
                        sobolev_regularization_lambda : Optional[float] = None,
                        amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0)) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''

    We perform all complex number manipulations (Fourier transforms, phase shifts, etc)
        in numpy, and only manipulate separate real and complex matrices with torch.
        Pytorch complex api looks pretty terrible

    :param waveform_data_matrix:
    :param n_canonical_waveforms:
    :param valid_sample_shifts:
    :param n_iter:
    :param device:
    :param amplitude_initialize_range:
    :return:
    '''

    n_observations, n_samples = waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples

    prev_iter_real_amplitude_A = np.zeros((n_observations, n_canonical_waveforms),
                                          dtype=np.float32)
    prev_iter_waveform_td = np.zeros((n_canonical_waveforms, n_samples),
                                     dtype=np.float32)
    prev_iter_delays = np.zeros((n_observations, n_canonical_waveforms),
                                dtype=np.float32)

    # we do random initialization of all of the variables

    prev_iter_real_amplitude_A[:, :] = np.random.uniform(amplitude_initialize_range[0],
                                                         amplitude_initialize_range[1],
                                                         size=prev_iter_real_amplitude_A.shape)

    rand_choice_data_waveform = np.random.randint(0, n_observations, size=n_canonical_waveforms)
    prev_iter_waveform_td[:, :] = waveform_data_matrix[rand_choice_data_waveform, :]

    prev_iter_delays[:, :] = np.random.randint(np.min(valid_sample_shifts),
                                               np.max(valid_sample_shifts),
                                               size=prev_iter_delays.shape)

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(waveform_data_matrix, axis=1)

    print("Beginning optimization loop")
    pbar = tqdm.tqdm(total=n_iter)
    for iter_count in range(n_iter):
        # within each iteration, we have a three step optimization
        # (1) Given fixed canonical waveforms and integer shifts,
        #       solve for real-valued amplitudes with nonnegative linear
        #       least squares
        # (2) Given fixed amplitudes and shifts, solve for waveforms in
        #       frequency domain with unconstrained complex-valued
        #       linear least squares
        # (3) Given fixed amplitudes and waveforms, all-but-one deconvolve
        #       to solve for the shifts

        # shape (n_canonical_waveforms, n_frequencies)
        canonical_waveform_ft = np.fft.rfft(prev_iter_waveform_td, axis=1)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        delay_phase_shift_mat = generate_fourier_phase_shift_matrices(prev_iter_delays,
                                                                      n_frequencies_not_rfft)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        canonical_waveform_shift_ft = delay_phase_shift_mat * canonical_waveform_ft[None, :, :]

        # shape (n_observations, n_canonical_waveforms, n_timepoints)
        canonical_waveforms_shifted = np.fft.irfft(canonical_waveform_shift_ft, n=n_samples, axis=2)

        # shape (n_observations, n_canonical_waveforms)
        # print("Iter {0}, Nonnegative least squares".format(iter_count))
        iter_real_amplitudes = nonnegative_least_squares_optimize_amplitudes(waveform_data_matrix,
                                                                             prev_iter_real_amplitude_A,
                                                                             canonical_waveforms_shifted,
                                                                             device,
                                                                             l1_regularization_lambda=l1_regularization_lambda)

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        # print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms3(
            iter_real_amplitudes,
            prev_iter_delays,
            observations_fourier_transform,
            n_frequencies_not_rfft,
            device,
            sobolev_lambda=sobolev_regularization_lambda
        )

        # real valued np.ndarray, shape (n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_samples, axis=1))

        # shape (n_observations, n_canonical_waveforms)
        # print("Iter {0}, Delay estimation".format(iter_count))
        iter_sample_delays = torch_fit_integer_shifts_all_but_one_template_match(observations_fourier_transform,
                                                                                 iter_real_amplitudes,
                                                                                 iter_canonical_waveform_ft,
                                                                                 prev_iter_delays,
                                                                                 valid_sample_shifts,
                                                                                 n_frequencies_not_rfft,
                                                                                 device)

        # calculate progress metrics
        # (probably best done in Fourier domain, doesn't require clever shifting and can
        #  be trivially parallelized...)
        mse = debug_evaluate_error(observations_fourier_transform,
                                   iter_real_amplitudes,
                                   iter_canonical_waveform_ft,
                                   iter_sample_delays,
                                   n_frequencies_not_rfft)

        # now rescale the waveforms and amplitudes
        # such that the waveforms each have L2 norm 1

        # shape (n_canonical_waveforms, )
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=1)

        # now update the loop variables
        prev_iter_delays = iter_sample_delays
        prev_iter_real_amplitude_A = iter_real_amplitudes * raw_optimized_waveform_magnitude[None, :]
        prev_iter_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]

        pbar.set_postfix({'MSE': mse})
        pbar.update(1)

    return prev_iter_real_amplitude_A, prev_iter_waveform_td, prev_iter_delays


EIDecomposition = namedtuple('EIDecomposition', ['amplitude', 'delay'])


def decompose_cells_by_fitted_compartment(eis_by_cell_id: Dict[int, np.ndarray],
                                          device: torch.device,
                                          n_basis_vectors: int = 3,
                                          snr_abs_threshold: float = 5.0,
                                          supersample_factor: int = 4,
                                          shifts: Tuple[int, int] = (-100, 100),
                                          maxiter_decomp: int = 25,
                                          renormalize_data_waveforms: bool = False,
                                          l1_regularize_lambda: Optional[float] = None,
                                          sobolev_regularize_lambda : Optional[float] = None,
                                          output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, np.ndarray]]]:
    '''

    :param eis_by_cell_id:
    :param device:
    :param n_basis_vectors:
    :param l1_regularize_lambda:
    :param snr_abs_threshold:
    :param supersample_factor:
    :param shifts:
    :param maxiter_decomp:
    :return:
    '''
    matrix_indices_by_cell_id = {}  # type: Dict[int, Tuple[slice, Sequence[int]]]
    to_concat = []  # type: List[np.ndarray]

    temp_cell_order = list(eis_by_cell_id.keys())

    cat_low = 0
    for cell_id in temp_cell_order:
        ei_mat = eis_by_cell_id[cell_id]

        chans_sufficient_magnitude = np.max(np.abs(ei_mat), axis=1) > snr_abs_threshold
        n_chans_sufficient = np.sum(chans_sufficient_magnitude)  # type: int

        readback_slice = slice(cat_low, cat_low + n_chans_sufficient)
        matrix_indices_by_cell_id[cell_id] = (readback_slice, chans_sufficient_magnitude)
        to_concat.append(ei_mat[chans_sufficient_magnitude, :])

        cat_low += n_chans_sufficient

    ei_data_mat = np.concatenate(to_concat, axis=0)
    bspline_supersampled = bspline_upsample_waveforms(ei_data_mat, supersample_factor)

    # now zero pad before and after
    padded_channels_sufficient_magnitude = np.pad(bspline_supersampled,
                                                  [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                                  mode='constant')

    if renormalize_data_waveforms:
        mag_padded = np.linalg.norm(padded_channels_sufficient_magnitude, axis=1)
        padded_channels_sufficient_magnitude = padded_channels_sufficient_magnitude / mag_padded[:, None]

    amplitudes, waveforms, delays = shifted_fourier_nmf(padded_channels_sufficient_magnitude,
                                                        n_basis_vectors,
                                                        np.r_[shifts[0]:shifts[1]],
                                                        maxiter_decomp,
                                                        device,
                                                        l1_regularization_lambda=l1_regularize_lambda,
                                                        sobolev_regularization_lambda=sobolev_regularize_lambda)

    # now unpack the results
    result_dict = {}  # type: Dict[int, EIDecomposition]
    for cell_id in temp_cell_order:
        orig_ei_mat = eis_by_cell_id[cell_id]
        n_channels = orig_ei_mat.shape[0]

        slice_section, sufficient_snr = matrix_indices_by_cell_id[cell_id]

        amplitude_matrix = np.zeros((n_channels, n_basis_vectors), dtype=np.float32)
        amplitude_matrix[sufficient_snr, :] = amplitudes[slice_section, :]

        delay_vector = np.zeros((n_channels, n_basis_vectors), dtype=np.int32)
        delay_vector[sufficient_snr, :] = delays[slice_section, :]

        result_dict[cell_id] = (amplitude_matrix, delay_vector)

    if output_debug_dict:
        debug_dict = {
            'amplitudes': amplitudes,
            'waveforms': waveforms,
            'delays': delays,
            'raw_data': padded_channels_sufficient_magnitude
        }

        return result_dict, waveforms, debug_dict

    return result_dict, waveforms


if __name__ == '__main__':
    compute_device = torch.device('cuda')

    # for now, don't bother with argparse since we still don't have an automatic way
    # to pick canonical waveforms
    print("Loading data")
    dataset = vl.load_vision_data('/Volumes/Lab/Users/ericwu/yass-ei/2018-03-01-0/data001',
                                  'data001',
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()

    example_on_parasols = dataset.get_all_cells_of_type('ON parasol')

    # eis_by_cell_id = { example_on_parasols[0] : dataset.get_ei_for_cell(example_on_parasols[0]).ei }
    eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in example_on_parasols}

    # 5e-3 was good
    decomposition_dict, basis_waveforms, debug_dict = decompose_cells_by_fitted_compartment(eis_by_cell_id,
                                                                                            compute_device,
                                                                                            maxiter_decomp=50,
                                                                                            l1_regularize_lambda=5e-3,
                                                                                            sobolev_regularize_lambda=1e-3,
                                                                                            renormalize_data_waveforms=True,
                                                                                            output_debug_dict=True)

    with open('joint_fitting.p', 'wb') as joint_fit_file:
        pickle_dict = {
            'decomposition': decomposition_dict,
            'waveforms': basis_waveforms,
        }

        pickle.dump(pickle_dict, joint_fit_file)

        pickle.dump(debug_dict, joint_fit_file)
