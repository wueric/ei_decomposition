import numpy as np
import torch
from scipy import interpolate as interpolate

from typing import List, Dict, Tuple, Sequence, Optional, Union
from collections import namedtuple

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

    # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    joint_coeff_permute = torch.cat([eq1_group_coeff, eq2_group_coeff], dim=0)

    # shape (n_rfft_frequencies, 2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
    joint_coeff = joint_coeff_permute.permute(2, 0, 1)

    if sobolev_lambda is not None:
        frequencies = np.fft.rfftfreq(n_true_frequencies)  # shape (n_rfft_frequencies, )

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
        canonical_waveforms_identity = np.eye(2 * n_canonical_waveforms) * 2 * np.pi

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms,, n_rfft_frequencies)
        canonical_waveform_freq_diag = canonical_waveforms_identity[:, :, None] * frequencies[None, None, :]

        diagonal_regularize = 2 * sobolev_lambda * np.power(1 - np.cos(canonical_waveform_freq_diag), 2)

        diagonal_regularize_torch_perm = torch.tensor(diagonal_regularize, dtype=torch.float32, device=device)
        diagonal_regularize_torch = diagonal_regularize_torch_perm.permute(2, 0, 1)

        joint_coeff = joint_coeff + diagonal_regularize_torch

    # shape (2 * n_canonical_waveforms, n_rfft_frequencies)
    joint_rhs_permute = torch.cat([eq1_rhs, eq2_rhs], dim=0)

    # shape (n_rfft_frequencies, 2 * n_canonical_waveforms)
    joint_rhs = joint_rhs_permute.permute(1, 0)

    # soln has shape (n_rfft_frequencies, 2 * n_canonical_waveforms)
    soln, _ = torch.solve(joint_rhs[:, :, None], joint_coeff)

    # shape (2 * n_canonical_waveforms, n_rfft_frequencies)
    soln_perm = soln.squeeze(2).permute(1, 0)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    soln_real_seg = soln_perm[:n_canonical_waveforms, :].cpu().numpy()
    soln_imag_seg = soln_perm[n_canonical_waveforms:, :].cpu().numpy()

    return soln_real_seg + 1j * soln_imag_seg


def fast_time_shifts_and_amplitudes_unique_shifts(observed_ft: np.ndarray,
                                                  ft_canonical: np.ndarray,
                                                  unique_phase_shifts: np.ndarray,
                                                  amplitude_matrix_real_np: np.ndarray,
                                                  n_true_frequencies,
                                                  max_iter: int,
                                                  device: torch.device,
                                                  l1_regularization_lambda: Optional[float] = None,
                                                  convergence_epsilon: float = 1e-3) \
        -> Tuple[np.ndarray, np.ndarray]:
    pass


def fast_time_shifts_and_amplitudes_shared_shifts(observed_ft: np.ndarray,
                                                  ft_canonical: np.ndarray,
                                                  valid_phase_shifts: np.ndarray,
                                                  amplitude_matrix_real_np: np.ndarray,
                                                  n_true_frequencies: int,
                                                  max_iter: int,
                                                  device: torch.device,
                                                  l1_regularization_lambda: Optional[float] = None,
                                                  converge_epsilon: float = 1e-3) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Iterative coarse-to-fine search helper function

    For fixed phase shifts, as defined by valid_phase_shifts, solves the nonnegative orthant least squares
        minimization problem with optional L1 regularization, and returns all solutions and objective fn values

    Notation for the below function

        objective fn is 1/2 |Ax-b|^2 = 1/2 (Ax-b)^T (Ax-b)
        gradient is A^T A x - A^T b

    Implementation notes:

    (A^T A)_{i,j}^{(z)} corresponds to the cross-correlation of the i^{th} canonical waveform, delayed by
        valid_phase_shifts[i,z] number of samples, with the j^{th} canonical waveform, delayed by
        valid_phase_shifts[j,z] number of samples

    :param observed_ft: observed waveforms in Fourier domain, complex valued, shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param valid_phase_shifts: integer array, shape (n_canonical_waveforms, n_valid_phase_shifts)
    :param n_true_frequencies: int
    :return:
    '''

    n_observations, n_rfft_frequencies = observed_ft.shape
    n_canonical_waveforms, n_valid_phase_shifts = valid_phase_shifts.shape

    #### Step 1: build A^T A from circular cross correlation #####################################
    # this one depends on relative timing for each of the canonical waveforms, so a bit tricky
    ft_canonical_conj = np.conjugate(ft_canonical)  # shape (n_canonical_waveforms, n_rfft_frequencies)
    circular_conv_ft = ft_canonical[:, None, :] * ft_canonical_conj[None, :, :]
    # shape (n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)

    circular_conv_td = np.fft.irfft(circular_conv_ft, n=n_true_frequencies, axis=2)
    # shape (n_canonical_waveforms, n_canonical_waveforms, n_timepoints), axis 1 corresponds to the shifted waveforms
    # relative to axis 0 fixed waveforms
    # (i,j,t)^{th} entry corresponds to cross correlation of i^{th} canonical waveform with j^{th} canonical waveform
    #   that has been delayed by t samples
    # This means that circular_conv_td is not symmetric for dims (0, 1)

    # now we have to build at_a matrix by grabbing the relevant pieces
    # not so straightforward, since we care about relative timing instead of absolute timing
    at_a_matrix_np = np.zeros((n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms),
                              dtype=np.float32)
    for j in range(n_canonical_waveforms):
        relative_shift = valid_phase_shifts - valid_phase_shifts[j, :][None, :]
        # shape (n_canonical_waveforms, n_valid_phase_shifts)

        taken_piece = np.take_along_axis(circular_conv_td[j, :, :], relative_shift[:, :], axis=1)
        # shape (n_canonical_waveforms, n_valid_phase_shifts)
        at_a_matrix_np[:, j, :] = taken_piece.transpose((1, 0))

    # shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix = torch.tensor(at_a_matrix_np, dtype=torch.float32, device=device)

    ##### Step 2: build A^T b from circular cross correlation with data matrix ##################
    # this one depends on absolute timing so it is much easier to pack
    data_circ_conv_ft = ft_canonical_conj[None, :, :] * observed_ft[:, None, :]
    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    data_circ_conv_td = np.fft.irfft(data_circ_conv_ft, n=n_true_frequencies, axis=2)
    # The (i,j,t)^{th} entry corresponds to cross correlation of the i^{th} data waveform with the j^{th} canonical
    #   waveform that has been delayed by t samples

    # we have to build A^T b from this matrix
    # shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    at_b_perm = np.take_along_axis(data_circ_conv_td, valid_phase_shifts[None, :, :], axis=2)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_np = at_b_perm.transpose((0, 2, 1))

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_torch = torch.tensor(at_b_np, dtype=torch.float32, device=device)

    #### Step 3: set up projected gradient descent problem #######################################
    eigenvalues_np, _ = np.linalg.eigh(at_a_matrix_np)
    eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)

    # eigenvalues has shape (n_valid_phase_shifts, n_waveforms)
    max_eigenvalue, _ = torch.max(eigenvalues, dim=1)  # shape (n_valid_phase_shifts, )
    min_eigenvalue, _ = torch.min(eigenvalues, dim=1)  # shape (n_valid_phase_shifts, )

    convergence_factor = 0.5 * (max_eigenvalue - min_eigenvalue)  # shape (n_valid_phase_shifts, )

    # boundaries for the step size
    # we have to have 0 < step_size <= 1/L where L is the largest eigenvalue
    # we make step_size smaller to be safe
    step_size = 1.0 / (2 * max_eigenvalue)  # has shape (n_valid_phase_shifts, )

    ##### Step 4: do the projected gradient descent (no batching) ################################
    ##### This function assumes that the batching is taken care of by the caller #################

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    amplitudes = torch.tensor(amplitude_matrix_real_np, dtype=torch.float32, device=device)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms, 1)
    at_a_x = at_a_matrix[None, :, :, :] @ amplitudes[:, :, :, None]

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    gradient = at_a_x.squeeze(3) - at_b_torch
    if l1_regularization_lambda is not None:
        gradient += l1_regularization_lambda

    for step_num in range(max_iter):

        # shape (n_observations, n_canonical_waveforms)
        next_amplitudes = torch.clamp(amplitudes - step_size[None, :, None] * gradient,
                                      min=0.0)

        # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
        at_a_x = (at_a_matrix[None, :, :, :] @ next_amplitudes[:, :, :, None]).squeeze(3)

        # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
        gradient = at_a_x - at_b_torch
        if l1_regularization_lambda is not None:
            gradient += l1_regularization_lambda

        step_distance = next_amplitudes - amplitudes
        # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
        step_distance = torch.sum(step_distance * step_distance, dim=2)  # shape (n_observations, n_valid_phase_shifts)
        convergence_bound = convergence_factor * step_distance  # shape (n_observations, n_valid_phase_shifts)
        worst_bound = torch.max(convergence_bound).item()

        amplitudes = next_amplitudes

        if worst_bound < converge_epsilon:
            break

    # now we have to calculate the objective values
    xt_at_a_x = (amplitudes[:, :, None, :] @ at_a_x[:, :, :, None]).squeeze()
    xt_at_b = (amplitudes[:, :, None, :] @ at_b_torch[:, :, :, None]).squeeze()

    partial_objective = 0.5 * xt_at_a_x - xt_at_b

    return amplitudes.cpu().numpy(), partial_objective.cpu().numpy()


def coarse_to_fine_time_shifts_and_amplitudes(observed_ft: np.ndarray,
                                              ft_canonical: np.ndarray,
                                              n_true_frequencies,
                                              valid_phase_shift_range: Tuple[int, int],
                                              first_pass_step_size: int,
                                              second_pass_width: int,
                                              device: torch.device,
                                              l1_regularization_lambda: Optional[float] = None,
                                              converge_epsilon: float = 1e-3,
                                              amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                                              max_batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    n_observations, _ = observed_ft.shape
    n_canonical_waveforms, n_rfft_frequencies = ft_canonical.shape

    ######### Step 1: first pass, perform nonnegative least squares minimization on a coarse ###############
    ######## grid of phase shifts, then pick the N best ####################################################
    low_shift, high_shift = valid_phase_shift_range
    shift_steps = np.r_[low_shift:high_shift:first_pass_step_size]
    mg = np.stack(np.meshgrid(*[shift_steps for _ in range(n_canonical_waveforms)]), axis=0)
    valid_phase_shifts_matrix = mg.reshape((n_canonical_waveforms, -1))

    _, n_valid_phase_shifts = valid_phase_shifts_matrix.shape

    amplitude_results = np.zeros((n_observations, n_valid_phase_shifts, n_canonical_waveforms),
                                 dtype=np.float32)
    objective_results = np.zeros((n_observations, n_valid_phase_shifts), dtype=np.float32)

    for low in range(0, n_valid_phase_shifts, max_batch_size):
        high = min(n_valid_phase_shifts, low + max_batch_size)

        amplitudes_random_init = np.random.uniform(amplitude_initialize_range[0],
                                                   amplitude_initialize_range[1],
                                                   size=(n_observations, high - low, n_canonical_waveforms))

        amplitude_batch, objective_batch = fast_time_shifts_and_amplitudes_shared_shifts(
            observed_ft,
            ft_canonical,
            valid_phase_shifts_matrix[low:high, :],
            amplitudes_random_init,
            n_true_frequencies,
            25,
            device,
            l1_regularization_lambda=l1_regularization_lambda,
            converge_epsilon=converge_epsilon
        )
        amplitude_results[:, low:high, :] = amplitude_batch
        objective_results[:, low:high] = objective_batch

    # pick the N best nodes to expand in detail

    return amplitude_results, objective_results


def greedy_template_match_time_shift(observed_ft: np.ndarray,
                                     ft_canonical: np.ndarray,
                                     valid_phase_shifts: np.ndarray,
                                     n_true_frequencies: int) -> np.ndarray:
    '''

    :param observed_ft: observed waveforms in Fourier domain, complex valued
            shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: Fourier transform of unshifted canonical waveforms, complex valued
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
    :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
    :return:
    '''
    n_observations, _ = observed_ft.shape
    n_canonical_waveforms, n_rfft_frequencies = ft_canonical.shape

    # shape (1, n_rfft_frequencies)
    single_phase_shift_matrix = generate_fourier_phase_shift_matrices(np.array([-1]), n_true_frequencies)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    ft_canonical_time_reversed = np.conjugate(ft_canonical)  # * single_phase_shift_matrix

    # shape (n_observations, n_rfft_Frequencies
    observed_ft_deconv = np.copy(observed_ft)

    deconv_already = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    deconv_time_shifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    for deconv_iter in range(n_canonical_waveforms):
        # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
        cross_corr_ft = observed_ft_deconv[:, None, :] * ft_canonical_time_reversed[None, :, :]
        cross_corr_td = np.fft.irfft(cross_corr_ft, axis=2, n=n_true_frequencies)

        cross_corr_td += (deconv_already[:, :, None] * (-1e9))

        # shape (n_observations, n_canonical_waveforms, n_valid_phase_shifts)
        valid_td_samples = np.take(cross_corr_td, valid_phase_shifts, axis=2)

        # shape (n_observations, n_canonical_waveforms)
        best_phase_shift_idx = np.argmax(valid_td_samples, axis=2)
        best_phase_shift_value = np.take_along_axis(valid_td_samples,
                                                    best_phase_shift_idx[:, :, None], axis=2).squeeze(2)

        # shape (n_observations, )
        best_canonical_waveform = np.argmax(best_phase_shift_value, axis=1)
        best_canonical_waveform_shift = np.take_along_axis(best_phase_shift_idx,
                                                           best_canonical_waveform[:, None], axis=1).squeeze(1)
        best_canonical_waveform_amplitude = np.take_along_axis(best_phase_shift_value,
                                                               best_canonical_waveform[:, None], axis=1).squeeze(1)

        # save outputs and update values for next iteration
        deconv_time_shifts[np.r_[0:n_observations], best_canonical_waveform] = valid_phase_shifts[
            best_canonical_waveform_shift]
        deconv_already[np.r_[0:n_observations], best_canonical_waveform] = 1

        observed_ft_deconv -= ft_canonical_time_reversed[best_canonical_waveform,
                              :] * best_canonical_waveform_amplitude[:, None]

    return deconv_time_shifts


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


def shifted_fourier_nmf_iterative_optimization2(waveform_data_matrix: np.ndarray,
                                                initialized_canonical_waveforms: np.ndarray,
                                                initialized_amplitudes: np.ndarray,
                                                intialized_delays: np.ndarray,
                                                valid_sample_shifts: np.ndarray,
                                                n_iter: int,
                                                device: torch.device,
                                                l1_regularization_lambda: Optional[float] = None,
                                                sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Subroutine for main optimization iterations, with different steps
        steps. Assumes all of the variables have been
        properly initialized by any method

    Order of the algorithm
        (1) With fixed waveforms and timeshifts, solve for the amplitudes with
            nonnegative least squares (with optional L1 regularization)
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with
            complex-valued least squares (with optional Sobolev gradient regularization)
        (3) With fixed amplitudes and waveforms, solve for the timeshifts
            with greedy Fourier-domain deconvolution

    Final nonnegative least squares with (with optional L1 regularization) to refit
        the amplitudes


    :param waveform_data_matrix: np.ndarray, time domain data matrix, shape (n_observations, n_timepoints)
    :param initialized_canonical_waveforms: np.ndarray, time domain canonical waveforms,
        shape (n_canonical_waveforms, n_timepoints)
    :param initialized_amplitudes: np.ndarray, initialized amplitudes, shape (n_observations, n_canonical_waveforms)
    :param intialized_delays: np.ndarray, initialized delays, shape (n_observations, n_canonical_waveforms)
    :param valid_sample_shifts: np.ndarray, valid sample shifts, shape (n_valid_sample_shifts)
    :param n_iter: int, number of iterations of the optimization to run
    :param device:
    :param l1_regularization_lambda:
    :param sobolev_regularization_lambda:
    :return:
    '''
    n_observations, n_samples = waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples

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
        canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        delay_phase_shift_mat = generate_fourier_phase_shift_matrices(intialized_delays,
                                                                      n_frequencies_not_rfft)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        canonical_waveform_shift_ft = delay_phase_shift_mat * canonical_waveform_ft[None, :, :]

        # shape (n_observations, n_canonical_waveforms, n_timepoints)
        canonical_waveforms_shifted = np.fft.irfft(canonical_waveform_shift_ft, n=n_samples, axis=2)

        # shape (n_observations, n_canonical_waveforms)
        # print("Iter {0}, Nonnegative least squares".format(iter_count))
        iter_real_amplitudes = nonnegative_least_squares_optimize_amplitudes(waveform_data_matrix,
                                                                             initialized_amplitudes,
                                                                             canonical_waveforms_shifted,
                                                                             device,
                                                                             l1_regularization_lambda=l1_regularization_lambda)

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        # print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms3(
            iter_real_amplitudes,
            intialized_delays,
            observations_fourier_transform,
            n_frequencies_not_rfft,
            device,
            sobolev_lambda=sobolev_regularization_lambda
        )

        # real valued np.ndarray, shape (n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_samples, axis=1))

        # now rescale the waveforms and amplitudes
        # such that the waveforms each have L2 norm 1
        # this is necessary for the greedy part of the
        # greedy deconvolution to work

        # real valued np.ndarray, shape (n_canonical_waveforms, )
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=1)

        initialized_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude[None, :]
        initialized_canonical_waveforms = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]
        iter_canonical_waveform_ft_scaled = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude[:, None]

        # shape (n_observations, n_canonical_waveforms)
        # print("Iter {0}, Delay estimation".format(iter_count))
        intialized_delays = greedy_template_match_time_shift(observations_fourier_transform,
                                                             iter_canonical_waveform_ft_scaled,
                                                             valid_sample_shifts,
                                                             n_frequencies_not_rfft)

        # calculate progress metrics
        mse = debug_evaluate_error(observations_fourier_transform,
                                   iter_real_amplitudes,
                                   iter_canonical_waveform_ft,
                                   intialized_delays,
                                   n_frequencies_not_rfft)

        pbar.set_postfix({'MSE': mse})
        pbar.update(1)

    # final recalculation of the amplitudes

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    delay_phase_shift_mat = generate_fourier_phase_shift_matrices(intialized_delays,
                                                                  n_frequencies_not_rfft)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    canonical_waveform_shift_ft = delay_phase_shift_mat * canonical_waveform_ft[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    canonical_waveforms_shifted = np.fft.irfft(canonical_waveform_shift_ft, n=n_samples, axis=2)

    # shape (n_observations, n_canonical_waveforms)
    initialized_amplitudes = nonnegative_least_squares_optimize_amplitudes(waveform_data_matrix,
                                                                           initialized_amplitudes,
                                                                           canonical_waveforms_shifted,
                                                                           device,
                                                                           l1_regularization_lambda=l1_regularization_lambda)

    mse = debug_evaluate_error(observations_fourier_transform,
                               initialized_amplitudes,
                               canonical_waveform_ft,
                               intialized_delays,
                               n_frequencies_not_rfft)

    pbar.set_postfix({'MSE': mse})
    pbar.update(1)

    return initialized_amplitudes, initialized_canonical_waveforms, intialized_delays, mse


def shifted_fourier_nmf_iterative_optimization3(waveform_data_matrix: np.ndarray,
                                                initialized_canonical_waveforms: np.ndarray,
                                                initialized_amplitudes: np.ndarray,
                                                intialized_delays: np.ndarray,
                                                valid_sample_shifts: np.ndarray,
                                                n_iter: int,
                                                device: torch.device,
                                                l1_regularization_lambda: Optional[float] = None,
                                                sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Subroutine for main optimization iterations, with different steps
        steps. Assumes all of the variables have been
        properly initialized by any method

    Order of the algorithm
        (1) With fixed waveforms, solve for the amplitudes and timeshifts with
            many parallel attempts at nonnegative least squares (with optional L1 regularization),
            and picking the best set of timeshifts and amplitudes
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with
            complex-valued least squares (with optional Sobolev gradient regularization)

    :param waveform_data_matrix: np.ndarray, time domain data matrix, shape (n_observations, n_timepoints)
    :param initialized_canonical_waveforms: np.ndarray, time domain canonical waveforms,
        shape (n_canonical_waveforms, n_timepoints)
    :param initialized_amplitudes: np.ndarray, initialized amplitudes, shape (n_observations, n_canonical_waveforms)
    :param intialized_delays: np.ndarray, initialized delays, shape (n_observations, n_canonical_waveforms)
    :param valid_sample_shifts: np.ndarray, valid sample shifts, shape (n_valid_sample_shifts)
    :param n_iter: int, number of iterations of the optimization to run
    :param device:
    :param l1_regularization_lambda:
    :param sobolev_regularization_lambda:
    :return:
    '''

    n_observations, n_samples = waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples

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
        canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

        coarse_to_fine_time_shifts_and_amplitudes(observations_fourier_transform,
                                                  canonical_waveform_ft,
                                                  n_frequencies_not_rfft,
                                                  (-100, 100),
                                                  5,
                                                  5,
                                                  device,
                                                  l1_regularization_lambda=l1_regularization_lambda)

    pass


def shifted_fourier_nmf_iterative_optimization(waveform_data_matrix: np.ndarray,
                                               initialized_canonical_waveforms: np.ndarray,
                                               initialized_amplitudes: np.ndarray,
                                               intialized_delays: np.ndarray,
                                               valid_sample_shifts: np.ndarray,
                                               n_iter: int,
                                               device: torch.device,
                                               l1_regularization_lambda: Optional[float] = None,
                                               sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Subroutine for main optimization iterations, assuming all of the variables have been
        properly initalized by any method

    Order of the algorithm
        (1) With fixed waveforms and timeshifts, solve for the amplitudes with
            nonnegative least squares (with optional L1 regularization)
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with
            complex-valued least squares (with optional Sobolev gradient regularization)
        (3) With fixed amplitudes and waveforms, solve for the timeshifts
            with simple Fourier domain all-but-one deconvolution

    :param waveform_data_matrix: np.ndarray, time domain data matrix, shape (n_observations, n_timepoints)
    :param initialized_canonical_waveforms: np.ndarray, time domain canonical waveforms,
        shape (n_canonical_waveforms, n_timepoints)
    :param initialized_amplitudes: np.ndarray, initialized amplitudes, shape (n_observations, n_canonical_waveforms)
    :param intialized_delays: np.ndarray, initialized delays, shape (n_observations, n_canonical_waveforms)
    :param valid_sample_shifts: np.ndarray, valid sample shifts, shape (n_valid_sample_shifts)
    :param n_iter: int, number of iterations of the optimization to run
    :param device:
    :param l1_regularization_lambda:
    :param sobolev_regularization_lambda:
    :return:
    '''
    n_observations, n_samples = waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples

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
        canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        delay_phase_shift_mat = generate_fourier_phase_shift_matrices(intialized_delays,
                                                                      n_frequencies_not_rfft)

        # shape (n_observations, n_canonical_waveforms, n_frequencies)
        canonical_waveform_shift_ft = delay_phase_shift_mat * canonical_waveform_ft[None, :, :]

        # shape (n_observations, n_canonical_waveforms, n_timepoints)
        canonical_waveforms_shifted = np.fft.irfft(canonical_waveform_shift_ft, n=n_samples, axis=2)

        # shape (n_observations, n_canonical_waveforms)
        # print("Iter {0}, Nonnegative least squares".format(iter_count))
        iter_real_amplitudes = nonnegative_least_squares_optimize_amplitudes(waveform_data_matrix,
                                                                             initialized_amplitudes,
                                                                             canonical_waveforms_shifted,
                                                                             device,
                                                                             l1_regularization_lambda=l1_regularization_lambda)

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        # print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms3(
            iter_real_amplitudes,
            intialized_delays,
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
                                                                                 intialized_delays,
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
        intialized_delays = iter_sample_delays
        initialized_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude[None, :]
        initialized_canonical_waveforms = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]

        pbar.set_postfix({'MSE': mse})
        pbar.update(1)

    return initialized_amplitudes, initialized_canonical_waveforms, intialized_delays, mse


def shifted_fourier_nmf(waveform_data_matrix: np.ndarray,
                        n_canonical_waveforms: int,
                        valid_sample_shifts: np.ndarray,
                        n_iter: int,
                        device: torch.device,
                        l1_regularization_lambda: Optional[float] = None,
                        sobolev_regularization_lambda: Optional[float] = None,
                        amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0)) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Shifted Fourier-domain NMF algorithm, with randomized initialization

    We perform all complex number manipulations (Fourier transforms, phase shifts, etc)
        in numpy, and only manipulate separate real and complex matrices with torch.
        Pytorch complex api looks pretty terrible

    Order of the optimization algorithm:
        (0) Randomly initialize waveforms, amplitudes, and timeshifts

        Then, iterate n_iter times over:
        (1) With fixed waveforms and timeshifts, solve for the amplitudes with
            nonnegative least squares (with optional L1 regularization)
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with
            complex-valued least squares (with optional Sobolev gradient regularization)
        (3) With fixed amplitudes and waveforms, solve for the timeshifts
            with simple Fourier domain all-but-one deconvolution

    :param waveform_data_matrix: np.ndarray, shape (n_waveforms, n_timepoints)
    :param n_canonical_waveforms: int, number of canonical waveforms
    :param valid_sample_shifts: np.ndarray, integer, contains all valid shifts to test for
    :param n_iter: int, number of iterations of the optimization to run
    :param device:
    :param amplitude_initialize_range:
    :return:
    '''

    n_observations, n_samples = waveform_data_matrix.shape

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

    return shifted_fourier_nmf_iterative_optimization2(waveform_data_matrix,
                                                       prev_iter_waveform_td,
                                                       prev_iter_real_amplitude_A,
                                                       prev_iter_delays,
                                                       valid_sample_shifts,
                                                       n_iter,
                                                       device,
                                                       l1_regularization_lambda=l1_regularization_lambda,
                                                       sobolev_regularization_lambda=sobolev_regularization_lambda)


def simple_deconv_time_shifts(waveform_data_matrix: np.ndarray,
                              normalized_canonical_waveforms: np.ndarray,
                              valid_sample_shifts: np.ndarray) -> np.ndarray:
    '''
    Estimate time shifts by calculating cross-correlations in Fourier domain

    :param waveform_data_matrix: np.ndarray, shape (n_observations, n_timepoints)
    :param normalized_canonical_waveforms: np.ndarray, shape (n_canonical_waveforms, n_timepoints)
        already normalized such that the L2 norm for each row is 1
    :param valid_sample_shifts: np.ndarray, all valid shifts
    :return: delays, shape (n_observations, n_canonical_waveforms)
    '''

    # shape (n_observations, n_rfft_frequencies)
    observed_ft = np.fft.rfft(waveform_data_matrix, axis=1)

    n_canonical_waveforms, n_timepoints = normalized_canonical_waveforms.shape

    # because we are calculating a cross-correlation, we first reverse the
    # the time domain canonical waveforms
    canonical_td_reversed = normalized_canonical_waveforms[:, ::-1]
    canonical_td_reversed = np.roll(canonical_td_reversed, 1, axis=1)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    canonical_reverse_ft = np.fft.rfft(canonical_td_reversed, axis=1)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    cross_correlation_ft = observed_ft[:, None, :] * canonical_reverse_ft[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    cross_correlation_td = np.fft.irfft(cross_correlation_ft, axis=2)

    # shape (n_observations, n_canonical_waveforms, n_valid_shifts)
    valid_timeshift_cross_correlations = np.take(cross_correlation_td, valid_sample_shifts, axis=2)

    # shape (n_observations, n_canonical_waveforms)
    best_timeshift_indices = np.argmax(valid_timeshift_cross_correlations, axis=2)
    return np.take(valid_sample_shifts, best_timeshift_indices, axis=0)


def optimize_initialized_waveforms_fourier_nmf(waveform_data_matrix: np.ndarray,
                                               initialized_canonical_waveforms: np.ndarray,
                                               valid_sample_shifts: np.ndarray,
                                               n_iter: int,
                                               device: torch.device,
                                               l1_regularization_lambda: Optional[float] = None,
                                               sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''
    Modified version of the above shifted_fourier_nmf routine, where instead of randomly initializing
        the canonical waveforms, the user instead specifies the starting waveforms

    In order for this algorithm to work, we change the order of the optimization to the following:

        (0) Solve for the timeshifts with cross-correlation (do it in Fourier domain),
            no all-but-one deconvolution since we have a strong prior for what the waveforms should be.

        Then, iterate n_iter times over
        (1) With fixed waveforms and timeshifts, solve for the amplitudes with
            nonnegative least squares (with optional L1 regularization)
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with
            complex-valued least squares (with optional Sobolev gradient regularization)
        (3) With fixed amplitudes and waveforms, solve for the timeshifts
            with simple Fourier domain all-but-one deconvolution

    :param waveform_data_matrix: np.ndarray, shape (n_observations, n_timepoints)
    :param initialized_canonical_waveforms: np.ndarray, shape (n_canonical_waveforms, n_timepoints)
        already normalized such that the L2 norm for each row is 1
    :param valid_sample_shifts:
    :param n_iter:
    :param device:
    :param l1_regularization_lambda:
    :param sobolev_regularization_lambda:
    :param amplitude_initialize_range:
    :return:
    '''
    n_observations, n_timepoints = waveform_data_matrix.shape
    n_canonical_waveforms, _ = initialized_canonical_waveforms.shape
    n_true_frequencies = n_timepoints

    # first calculate the timeshifts
    observed_ft = np.fft.rfft(waveform_data_matrix, axis=1)
    ft_canonical = np.fft.rfft(initialized_canonical_waveforms, axis=1)
    initialized_time_shifts = greedy_template_match_time_shift(observed_ft,
                                                               ft_canonical,
                                                               valid_sample_shifts,
                                                               n_true_frequencies)

    # these initial values literally do not matter, since we're going to outright solve for
    # the amplitudes anyway in the first step of the first iteration of the optimization step
    initialized_amplitudes = np.zeros((n_observations, n_canonical_waveforms), dtype=np.float32)
    initialized_amplitudes[:, :] = np.random.uniform(0.0, 10.0, size=initialized_amplitudes.shape)

    return shifted_fourier_nmf_iterative_optimization2(waveform_data_matrix,
                                                       initialized_canonical_waveforms,
                                                       initialized_amplitudes,
                                                       initialized_time_shifts,
                                                       valid_sample_shifts,
                                                       n_iter,
                                                       device,
                                                       l1_regularization_lambda=l1_regularization_lambda,
                                                       sobolev_regularization_lambda=sobolev_regularization_lambda)


EIDecomposition = namedtuple('EIDecomposition', ['amplitude', 'delay'])


def decompose_cells_by_fitted_compartment(eis_by_cell_id: Dict[int, np.ndarray],
                                          device: torch.device,
                                          n_basis_vectors: Optional[int] = None,
                                          initialized_basis_vectors: Optional[np.ndarray] = None,
                                          snr_abs_threshold: float = 5.0,
                                          supersample_factor: int = 5,
                                          shifts: Tuple[int, int] = (-100, 100),
                                          maxiter_decomp: int = 25,
                                          renormalize_data_waveforms: bool = True,
                                          l1_regularize_lambda: Optional[float] = None,
                                          sobolev_regularize_lambda: Optional[float] = None,
                                          output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray, float],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, float, Dict[str, np.ndarray]]]:
    '''

    :param eis_by_cell_id:
    :param device:
    :param n_basis_vectors:
    :param initialized_basis_vectors:
    :param l1_regularize_lambda:
    :param snr_abs_threshold:
    :param supersample_factor:
    :param shifts:
    :param maxiter_decomp:
    :return:
    '''

    # check the inputs for correctness
    # must either specify the number of basis waveforms, or specify initial basis waveforms outright
    if n_basis_vectors is None and initialized_basis_vectors is None:
        raise ValueError('Must specify either n_basis_vectors or initialized_basis_vectors')
    elif n_basis_vectors is not None and initialized_basis_vectors is not None:
        raise ValueError('Can specify only one of n_basis_vectors and initialized_basis_vectors')

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

    if n_basis_vectors is not None:
        amplitudes, waveforms, delays, mse = shifted_fourier_nmf(padded_channels_sufficient_magnitude,
                                                                 n_basis_vectors,
                                                                 np.r_[shifts[0]:shifts[1]],
                                                                 maxiter_decomp,
                                                                 device,
                                                                 l1_regularization_lambda=l1_regularize_lambda,
                                                                 sobolev_regularization_lambda=sobolev_regularize_lambda)
    else:

        # also need to supersample and pad the initial basis waveforms
        bspline_supersampled_basis = bspline_upsample_waveforms(initialized_basis_vectors, supersample_factor)
        padded_basis_waveforms_init = np.pad(bspline_supersampled_basis,
                                             [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                             mode='constant')
        amplitudes, waveforms, delays, mse = optimize_initialized_waveforms_fourier_nmf(
            padded_channels_sufficient_magnitude,
            padded_basis_waveforms_init,
            np.r_[shifts[0]:shifts[1]],
            maxiter_decomp,
            device,
            l1_regularization_lambda=l1_regularize_lambda,
            sobolev_regularization_lambda=sobolev_regularize_lambda
        )

        n_basis_vectors = initialized_basis_vectors.shape[0]

    if renormalize_data_waveforms:
        amplitudes = mag_padded[:, None] * amplitudes

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

        return result_dict, waveforms, mse, debug_dict

    return result_dict, waveforms, mse
