import visionloader as vl
import numpy as np
import scipy.interpolate as interpolate

import torch

import argparse

from typing import List, Dict, Tuple, Callable

import pickle

import tqdm


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

    for iter_count in range(max_iter):

        # shape (n_observations, n_canonical_waveforms)
        next_amplitudes = torch.clamp(amplitudes - step_size[:, None] * gradient,
                                      min=0.0)

        ax_minus_b = shifted_basis_t @ next_amplitudes[:, :, None] - observe_mat[:, :, None]
        # shape (n_observations, n_timepoints, 1)

        # (n_observations, n_canonical_waveforms, n_timepoints) x (n_observations, n_timepoints, 1)
        # = (n_observations, n_canonical_waveforms, 1) -> (n_observations, n_canonical_waveforms)
        gradient = torch.squeeze(shifted_basis @ ax_minus_b, -1)

        step_distance = next_amplitudes - amplitudes  # shape (n_observations, n_canonical_waveforms)
        step_distance_square_mag = torch.sum(step_distance * step_distance,
                                             dim=1)  # shape (n_observations, )
        convergence_bound = convergence_factor * step_distance_square_mag  # shape (n_observations, )
        worst_bound = torch.max(convergence_bound).item()

        amplitudes = next_amplitudes

        if worst_bound < converge_epsilon:
            break

    return amplitudes.cpu().numpy()


def fourier_complex_least_squares_optimize_waveforms2(amplitude_matrix_real_np: np.ndarray,
                                                      phase_delays_np: np.ndarray,
                                                      ft_complex_observations_np: np.ndarray,
                                                      device) -> np.ndarray:
    '''

    :param amplitude_matrix_real_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param phase_delays_np: integer sample delays for each canonical waveform, for each observation
        shape (n_observations, n_canonical_waveforms)
    :param ft_complex_observations_np: complex-valued Fourier transform of the observed data,
        shape (n_observations, n_frequencies = n_samples)
    :param device:
    :return: tuple of real component, imaginary component of canonical waveform Fourier transform
        each has shape (n_canonical_waveforms, n_frequencies)
    '''

    _, n_frequencies = ft_complex_observations_np.shape

    # complex-valued, shape (n_observations, n_canonical_waveforms, n_frequencies)
    complex_phase_matrix = generate_fourier_phase_shift_matrices(phase_delays_np,
                                                                 n_frequencies)

    # complex-valued, shape (n_observations, n_canonical_waveforms, n_frequencies)
    rhs_submatrix_prediv = amplitude_matrix_real_np[:, :, None] * ft_complex_observations_np[:, None, :]
    rhs_matrix_hadamard_div = rhs_submatrix_prediv / complex_phase_matrix

    # real-valued, shape (n_observations, n_canonical_waveforms, n_frequencies)
    rhs_real_np, rhs_complex_np = np.real(rhs_matrix_hadamard_div), np.imag(rhs_matrix_hadamard_div)
    rhs_real_nosum = torch.tensor(rhs_real_np, dtype=torch.float32, device=device)
    rhs_complex_nosum = torch.tensor(rhs_complex_np, dtype=torch.float32, device=device)

    # real-valued, shape (n_canonical_waveforms, n_frequencies)
    rhs_real = torch.sum(rhs_real_nosum, dim=0)
    rhs_complex = torch.sum(rhs_complex_nosum, dim=0)

    # real-valued, shape (2, n_canonical_waveforms, n_frequencies)
    rhs_stack = torch.stack([rhs_real, rhs_complex], dim=0)

    # real-valued, shape (n_observations, n_canonical_waveforms)
    amplitude_matrix = torch.tensor(amplitude_matrix_real_np, dtype=torch.float32, device=device)

    # real-valued, shape (n_canonical_waveforms, n_canonical_waveforms)
    at_a = amplitude_matrix.permute(1, 0) @ amplitude_matrix

    lu_factor_at_a, lu_pivots_at_a = torch.lu(at_a)

    # shape (2, n_canonical_waveforms, n_frequencies)
    sols_componentwise = torch.lu_solve(rhs_stack, lu_factor_at_a, lu_pivots_at_a)

    real_sols = sols_componentwise[0, :, :].cpu().numpy()
    imag_sols = sols_componentwise[1, :, :].cpu().numpy()

    return real_sols + 1j * imag_sols


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


def fit_shifts_all_but_one_template_match(observed_ft: np.ndarray,
                                          real_amplitude_matrix_np: np.ndarray,
                                          ft_canonical: np.ndarray,
                                          previous_shifts_integer: np.ndarray,
                                          valid_phase_shifts: np.ndarray) -> np.ndarray:
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
    :return: timeshifts, shape (n_observations, n_canonical_waveforms)
    '''
    n_observations, n_frequencies = observed_ft.shape
    _, n_canonical_waveforms = real_amplitude_matrix_np.shape

    output_timeshifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    # shape (n_observations, n_timepoints)

    # shape (n_valid_phase_shifts, n_frequencies)
    phaseshift_all_allowed_matrices = generate_fourier_phase_shift_matrices(valid_phase_shifts,
                                                                            n_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(previous_shifts_integer,
                                                                 n_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shifted_canonical_ft = phase_shift_matrices * ft_canonical[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    scaled_shifted_canonical_ft = real_amplitude_matrix_np[:, :, None] * phase_shifted_canonical_ft

    # shape (n_canonical_waveforms, n_frequencies)
    deconvolved_ft = observed_ft - np.sum(scaled_shifted_canonical_ft, axis=1)

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
    frequencies = np.fft.fftfreq(n_frequencies)  # shape (n_frequencies, )

    # this is 2 * pi * f / F
    phase_radians = 2 * np.pi * frequencies  # shape (n_frequencies, )

    # this is -j * 2 * pi * tau * f / F
    complex_exponential_argument = -1j * sample_delays[sample_delays_slice] * phase_radians[phase_radians_slice]
    # shape (n_canonical_waveforms, n_frequencies, sample_delays)

    return np.exp(complex_exponential_argument)


def debug_evaluate_error(observed_ft: np.ndarray,
                         fit_real_amplitudes: np.ndarray,
                         canonical_waveform_ft: np.ndarray,
                         time_shifts: np.ndarray) -> Tuple[float, float, float]:
    '''

    :param observed_ft: Fourier transform of observed data, complex-valued,
        shape (n_observations, n_frequencies)
    :param fit_real_amplitudes: real-valued scale amplitude of each canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param canonical_waveform_ft: Fourier transform of canonical waveforms, complex-valued,
        shape (n_canonical_waveforms, n_frequencies)
    :param time_shifts: Time shifts required for each canonical waveform to fit each observation,
        shape (n_observations, n_canonical_waveforms)
    :return: MSE error, real valued time domain power, imaginary valued time domain power
    '''
    n_observations, n_frequencies = observed_ft.shape

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    time_shift_matrices = generate_fourier_phase_shift_matrices(time_shifts,
                                                                n_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    shifted_no_scale_ft = canonical_waveform_ft[None, :, :] * time_shift_matrices

    model_ft = np.squeeze(fit_real_amplitudes[:, None, :] @ shifted_no_scale_ft, axis=1)

    real_power = np.linalg.norm(np.real(model_ft)) ** 2
    imag_power = np.linalg.norm(np.imag(model_ft)) ** 2

    diff = observed_ft - model_ft
    errors = np.linalg.norm(diff, axis=1)
    mean_error = np.mean(errors)

    return mean_error, real_power, imag_power


def shifted_fourier_nmf(waveform_data_matrix: np.ndarray,
                        n_canonical_waveforms: int,
                        valid_sample_shifts: np.ndarray,
                        n_iter: int,
                        device: torch.device,
                        amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                        waveform_initialization_range: Tuple[float, float] = (-1.0, 1.0)) \
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
    :param waveform_initialization_range:
    :return:
    '''

    n_observations, n_samples = waveform_data_matrix.shape
    n_frequencies = n_samples

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
    prev_iter_waveform_td[:, :] = np.random.uniform(waveform_initialization_range[0],
                                                    waveform_initialization_range[1],
                                                    size=prev_iter_waveform_td.shape)
    prev_iter_delays[:, :] = np.random.randint(np.min(valid_sample_shifts),
                                               np.max(valid_sample_shifts),
                                               size=prev_iter_delays.shape)

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (n_observations, n_frequencies)
    print("Calculating observed Fourier transforms")
    observations_fourier_transform = np.fft.fft(waveform_data_matrix, axis=1)

    print("Beginning optimization loop")
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
        print("Iter {0}, Canonical waveform fft, {1}".format(iter_count, prev_iter_waveform_td.shape))
        canonical_waveform_ft = np.fft.fft(prev_iter_waveform_td, axis=1)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        print("Iter {0}, phase shift mat generation, {1}".format(iter_count, prev_iter_waveform_td.shape))
        delay_phase_shift_mat = generate_fourier_phase_shift_matrices(prev_iter_delays,
                                                                      n_frequencies)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        canonical_waveform_shift_ft = delay_phase_shift_mat * canonical_waveform_ft[None, :, :]

        # shape (n_observations, n_canonical_waveforms, n_timepoints)
        canonical_waveforms_shifted = np.real(np.fft.ifft(canonical_waveform_shift_ft, axis=2))
        print(canonical_waveform_ft.shape, canonical_waveforms_shifted.shape)

        # shape (n_observations, n_canonical_waveforms)
        print("Iter {0}, Nonnegative least squares".format(iter_count))
        iter_real_amplitudes = nonnegative_least_squares_optimize_amplitudes(waveform_data_matrix,
                                                                             prev_iter_real_amplitude_A,
                                                                             canonical_waveforms_shifted,
                                                                             device)

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms2(
            iter_real_amplitudes,
            prev_iter_delays,
            observations_fourier_transform,
            device
        )

        # real valued np.ndarray, shape (n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = np.real(np.fft.ifft(iter_canonical_waveform_ft, axis=1))

        # shape (n_observations, n_canonical_waveforms)
        print("Iter {0}, Delay estimation".format(iter_count))
        iter_sample_delays = fit_shifts_all_but_one_template_match(observations_fourier_transform,
                                                                   iter_real_amplitudes,
                                                                   iter_canonical_waveform_ft,
                                                                   prev_iter_delays,
                                                                   valid_sample_shifts)

        # calculate progress metrics
        # (probably best done in Fourier domain, doesn't require clever shifting and can
        #  be trivially parallelized...)
        mse, real_power, imag_power = debug_evaluate_error(observations_fourier_transform,
                                                           iter_real_amplitudes,
                                                           iter_canonical_waveform_ft,
                                                           iter_sample_delays)
        print("Iteration {0}: MSE {1}, real_power {2}, imag_power {3}".format(iter_count, mse, real_power, imag_power))

        # now update the loop variables
        prev_iter_delays = iter_sample_delays
        prev_iter_real_amplitude_A = iter_real_amplitudes
        prev_iter_waveform_td = iter_canonical_waveform_td

    return prev_iter_real_amplitude_A, prev_iter_waveform_td, prev_iter_delays


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

    example_on_parasols = dataset.get_all_cells_of_type('ON parasol')

    eis_for_cell = [dataset.get_ei_for_cell(cell_id).ei for cell_id in example_on_parasols]
    eis_stacked = np.concatenate(eis_for_cell, axis=0)
    eis_sufficient_magnitude = np.max(np.abs(eis_stacked), axis=1) > 5.0

    channels_sufficient_magnitude = eis_stacked[eis_sufficient_magnitude]

    # now 5x bspline supersample
    print("Bspline supersample")
    bspline_supersampled = bspline_upsample_waveforms(channels_sufficient_magnitude, 5)

    # now zero pad before and after
    print("Zero padding")
    padded_channels_sufficient_magnitude = np.pad(bspline_supersampled,
                                                  [(40, 40), (0, 0)],
                                                  mode='constant')

    print("Decomposition optimization")
    a, b, c = shifted_fourier_nmf(padded_channels_sufficient_magnitude,
                                  3,
                                  np.r_[-40:40],
                                  500,
                                  device)

