from typing import Optional, List, Tuple

import numpy as np
import torch

from lib.util_fns import generate_fourier_phase_shift_matrices


def _batch_rank_deficient_identifier(batched_amplitudes_real: np.ndarray,
                                     batched_valid_mat: np.ndarray,
                                     norm_cutoff: float = 1e-2) -> np.ndarray:
    '''
    Identifies cases where a cell is missing entire components (i.e. a row of batched_amplitudes_real is so
        close to 0 that, in effect, this cell is missing the corresponding basis waveform component entirely).

    This is necessary to help us avoid singular or ill-conditioned linear systems when solving for the waveforms
        in frequency domain

    To keep things really simple, we apply a simple heuristic that should help us identify how present a particular
        basis waveform is in the data. The heuristic is

        "In order to be included, each basis waveform must make up at least norm_cutoff fraction of the power
        on at least one channel"

    This function implicitly assumes that batched_amplitudes_real corresponds to amplitudes of already-normalized
        basis waveforms (i.e. when we computed batched_amplitudes_real, the basis waveforms that we used had to have
        L2 norm of 1), so power can be computed directly from batched_amplitudes_real without extra
        normalization factors

    :param batched_amplitudes_real: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (batch, n_observations, n_canonical_waveforms)
    :param batched_valid_mat: boolean matrix marking which entries of the above matrices correspond to real data,
        and which entries correspond to padding.
    :param norm_cutoff: floating point number, in interval [0, 1]. Should typically be very very small, something like
        1e-2 or something like that, since it is meant to be a fraction
    :return:

        * waveform_satisfies_crit, shape (batch, n_basis_waveforms), boolean-valued. Each entry is True
            if the corresponding basis waveform should be included in the linear system, and is False
            if the basis waveform should not be included in the linear system

    '''

    # shape (batch, n_observations, n_canonical_waveforms)
    power = batched_amplitudes_real * batched_amplitudes_real

    # shape (batch, n_observations)
    tot_power = np.sum(power, axis=2)
    tot_power[~batched_valid_mat] = 1.0
    tot_power[tot_power==0.0] = 1.0

    # shape (batch, n_observations, n_canonical_waveforms)
    power_fraction = power / tot_power[:, :, None]

    # shape (batch, n_observations, n_canonical_waveforms)
    # value is True if exceeds threshold AND the electrode is legit
    exceeds_power_frac = np.logical_and(power_fraction > norm_cutoff, batched_valid_mat[:, :, None])

    # shape (batch, n_canonical_waveforms)
    waveform_satisfies_crit = np.any(exceeds_power_frac, axis=1)

    return waveform_satisfies_crit


def _batch_assemble_coefficients_and_solve(batched_amplitudes_real: np.ndarray,
                                           batched_phase_delays: np.ndarray,
                                           batched_ft_observations: np.ndarray,
                                           batch_valid_mat: np.ndarray,
                                           n_true_frequencies: int,
                                           device: torch.device,
                                           observation_loss_weight: Optional[np.ndarray] = None,
                                           sobolev_lambda: Optional[float] = None):
    '''
    Can be used to solve reduced rank basis waveform matrices as well; i.e. n_basis_waveforms for the inputs
        of this function can be less than n_basis_waveforms overall in the case that a particular cell is
        missing certain compartments

    :param batched_amplitudes_real: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_phase_delays: integer sample delays for each canonical waveform, for each observation
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_ft_observations: complex-valued Fourier transform of the observed data,
        shape (batch, n_observations, n_rfft_frequencies)
    :param batch_valid_mat: boolean matrix marking which entries of the above matrices correspond to real data,
        and which entries correspond to padding.
        shape (batch, n_observations), boolean valued
    :return: shape (batch, n_basis_waveforms, n_rfft_frequencies)
        Complex-valued np.ndarray
    '''

    if observation_loss_weight is not None:
        batched_amplitudes_real = batched_amplitudes_real * observation_loss_weight[:, :, None]
        batched_ft_observations = batched_ft_observations * observation_loss_weight[:, :, None]

    batch, n_observations, n_canonical_waveforms = batched_amplitudes_real.shape
    _, _, n_rfft_frequencies = batched_ft_observations.shape

    # shape (batch, n_observations)
    valid_one_matrix = torch.tensor(batch_valid_mat.astype(np.float32), dtype=torch.float32, device=device)

    # real valued, shape (batch, n_observations, n_canonical_waveforms)
    amplitude_mat_torch = torch.tensor(batched_amplitudes_real, dtype=torch.float32, device=device)

    # shape (batch, n_observations, n_rfft_frequencies)
    real_observe_ft_torch = torch.tensor(batched_ft_observations.real, dtype=torch.float32, device=device)
    imag_observe_ft_torch = torch.tensor(batched_ft_observations.imag, dtype=torch.float32, device=device)

    # complex-valued, shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    complex_phase_matrix = generate_fourier_phase_shift_matrices(batched_phase_delays,
                                                                 n_true_frequencies)

    # both have shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    real_phase_mat_torch = torch.tensor(complex_phase_matrix.real, dtype=torch.float32, device=device)
    imag_phase_mat_torch = torch.tensor(complex_phase_matrix.imag, dtype=torch.float32, device=device)

    # shape (batch, n_observations, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    # the (l^th, k^{th}, j^{th}, f^{th}) entry is real{P}^{(l)}_{f,k} * real{P}^{(l)}_{f,j}
    ####################### (l, k, None, f) ################# (l, None, j, f) ###########
    real_real_phase = real_phase_mat_torch[:, :, :, None, :] * real_phase_mat_torch[:, :, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is imag{P}^{(l)}_{f,k} * imag{P}^{(l)}_{f,j}
    imag_imag_phase = imag_phase_mat_torch[:, :, :, None, :] * imag_phase_mat_torch[:, :, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is imag{P}^{(l)}_{f,k} * real{P}^{(l)}_{f,j}
    imag_real_phase = imag_phase_mat_torch[:, :, :, None, :] * real_phase_mat_torch[:, :, None, :, :]

    # the (l^th, k^{th}, j^{th}, f^{th}) entry is real{P}^{(l)}_{f,k} * imag{P}^{(l)}_{f,j}
    real_imag_phase = real_phase_mat_torch[:, :, :, None, :] * imag_phase_mat_torch[:, :, None, :, :]

    # shape (batch, n_observations, n_canonical_waveforms, n_canonical_waveforms)
    # the (l^{th}, k^{th}, j^{th}) entry is A_{k,l} A_{j,l}
    amplitude_outer_product = amplitude_mat_torch[:, :, :, None] * amplitude_mat_torch[:, :, None, :]

    # shape (batch, n_observations, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_presum_real_coeff = (real_real_phase + imag_imag_phase) * amplitude_outer_product[:, :, :, :, None]
    # we need to mask the contributions of some of the observations, since those correspond to null data
    eq1_group_presum_real_coeff = eq1_group_presum_real_coeff * valid_one_matrix[:, :, None, None, None]
    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_real_coeff = torch.sum(eq1_group_presum_real_coeff, dim=1)

    # shape (batch, n_observations, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_presum_imag_coeff = (imag_real_phase - real_imag_phase) * amplitude_outer_product[:, :, :, :, None]
    eq1_group_presum_imag_coeff = eq1_group_presum_imag_coeff * valid_one_matrix[:, :, None, None, None]
    # shape (batch, n_canonical_waveforms, n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_imag_coeff = torch.sum(eq1_group_presum_imag_coeff, dim=1)

    # shape (batch, n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    eq1_group_coeff = torch.cat([eq1_group_real_coeff, eq1_group_imag_coeff], dim=2)

    # shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    ################# (l, k, f) ############### (l, k, None) ########################### (l, None, f)
    eq1_rhs_re = real_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * real_observe_ft_torch[:, :,
                                                                                         None, :]
    eq1_rhs_im = imag_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * imag_observe_ft_torch[:, :,
                                                                                         None, :]

    # shape (batch, n_canonical_waveforms, n_frequencies)
    eq1_rhs = torch.sum((eq1_rhs_re + eq1_rhs_im) * valid_one_matrix[:, :, None, None], dim=1)

    # shape (batch, n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    eq2_group_coeff = torch.cat([-1.0 * eq1_group_imag_coeff, eq1_group_real_coeff], dim=2)

    # shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs_p = real_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * imag_observe_ft_torch[:, :,
                                                                                        None, :]
    eq2_rhs_m = imag_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * real_observe_ft_torch[:, :,
                                                                                        None, :]

    # shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs = torch.sum((eq2_rhs_p - eq2_rhs_m) * valid_one_matrix[:, :, None, None], dim=1)

    # shape (batch, 2 * n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
    joint_coeff_permute = torch.cat([eq1_group_coeff, eq2_group_coeff], dim=1)

    # shape (batch, n_rfft_frequencies, 2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
    joint_coeff = joint_coeff_permute.permute(0, 3, 1, 2)

    if sobolev_lambda is not None:
        frequencies = np.fft.rfftfreq(n_true_frequencies)  # shape (n_rfft_frequencies, )

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
        canonical_waveforms_identity = np.eye(2 * n_canonical_waveforms) * 2 * np.pi

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
        canonical_waveform_freq_diag = canonical_waveforms_identity[:, :, None] * frequencies[None, None, :]

        # shape (2 * n_canonical_waveforms, 2 * n_canonical_waveforms, n_rfft_frequencies)
        diagonal_regularize = 2 * sobolev_lambda * np.power(1 - np.cos(canonical_waveform_freq_diag), 2)

        diagonal_regularize_torch_perm = torch.tensor(diagonal_regularize, dtype=torch.float32, device=device)
        # shape (n_rfft_frequencies, 2 * n_canonical_waveforms, 2 * n_canonical_waveforms)
        diagonal_regularize_torch = diagonal_regularize_torch_perm.permute(2, 0, 1)

        joint_coeff = joint_coeff + diagonal_regularize_torch[None, :, :, :]

    # shape (batch, 2 * n_canonical_waveforms, n_rfft_frequencies)
    joint_rhs_permute = torch.cat([eq1_rhs, eq2_rhs], dim=1)

    # shape (batch, n_rfft_frequencies, 2 * n_canonical_waveforms)
    joint_rhs = joint_rhs_permute.permute(0, 2, 1)

    # soln has shape (batch, n_rfft_frequencies, 2 * n_canonical_waveforms)
    soln = torch.linalg.solve(joint_coeff, joint_rhs[:, :, :, None])

    # shape (batch, 2 * n_canonical_waveforms, n_rfft_frequencies)
    soln_perm = soln.squeeze(3).permute(0, 2, 1)

    # shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    soln_real_seg = soln_perm[:, :n_canonical_waveforms, :].cpu().numpy()
    soln_imag_seg = soln_perm[:, n_canonical_waveforms:, :].cpu().numpy()

    return soln_real_seg + 1j * soln_imag_seg


def batch_fourier_complex_least_square_optimize3(batched_amplitudes_real: np.ndarray,
                                                 batched_phase_delays: np.ndarray,
                                                 batched_ft_observations: np.ndarray,
                                                 batch_valid_mat: np.ndarray,
                                                 batched_prev_iter_basis_ft : np.ndarray,
                                                 n_true_frequencies: int,
                                                 device: torch.device,
                                                 norm_cutoff: float=1e-2,
                                                 observation_loss_weight: Optional[np.ndarray] = None,
                                                 sobolev_lambda: Optional[float] = None) \
        -> np.ndarray:
    '''
    Computes the Fourier-domain waveform optimization in batch, for cases where we have multiple sets of
        basis waveforms

    Note that because this is typically used to run the decomposition on each cell individually, where each
        cell has its own set of basis waveforms, we are virtually guaranteed to have singular matrices because of
        the sparsity in the decomposition. In particular:

        (1) We explicitly regularize the amplitudes matrix to be sparse with respect to the basis components.
            Therefore, for cells that don't appear on that many electrodes, it is quite probable that the cell
            will have 0 or near-0 amplitudes for a given basis waveform (i.e. the cell has no axonal component)
        (2) This results in 0 or near-0 rows and columns in the frequency-domain matrix.

    This function attempts to identify these near-0 amplitude components, and where identified, the function
        solves a lower-rank version of the problem rather than the full-rank version to avoid singular matrices.
        In the cases where a lower-rank matrix is solved for, the basis waveforms that are "missing" are not updated,
        since there is no data to update them with.

    Algorithm:

    :param batched_amplitudes_real: real-valued amplitudes for each observation, each shifted basis waveform,
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_phase_delays: integer sample delays for each basis waveform, for each observation
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_ft_observations: complex-valued Fourier transform of the observed data,
        shape (batch, n_observations, n_rfft_frequencies)
    :param batch_valid_mat: boolean matrix marking which entries of the above matrices correspond to real data,
        and which entries correspond to padding.
        shape (batch, n_observations), boolean valued
    :param batched_prev_iter_basis_ft: rFFT of the basis waveforms from the previous iteration. We need this matrix
        in the low-rank case, since in the low rank case, where we don't solve for some of the basis waveforms, we
        want to write-through those basis waveforms from the previous iteration (i.e. a no-change update)

        shape (batch, n_basis_waveforms, n_rfft_frequencies)
    :param n_true_frequencies : int, number of frequencies = n_samples for the normal FFT
        (not the number of rFFT frequencies)
    :param device: torch.device
    :param observation_loss_weight: Vector of weights, for weighting the contribution to the loss
        of each individual waveform. shape (batch, n_observations, )
    :return: Fourier transform of least-squares basis waveforms, shape (batch, n_basis_waveforms, n_rfft_frequencies)
        Complex-valued np.ndarray
    '''

    batch, n_observations, n_basis_waveforms = batched_amplitudes_real.shape
    _, _, n_rfft_frequencies = batched_ft_observations.shape

    # shape (batch, n_basis_waveforms), boolean-valued
    rank_include = _batch_rank_deficient_identifier(batched_amplitudes_real, batch_valid_mat, norm_cutoff=norm_cutoff)

    # shape (batch, ), positive integer-valued. If an entry is 0 something is wrong
    rank_values = np.sum(rank_include, axis=1)

    # shape (n_possible_ranks, ), positive integer-valued. Should typically be <= n_basis_waveforms, since
    # 0-rank is not allowed for obvious reasons
    unique_ranks_sorted = np.unique(rank_values)

    solved_waveforms = batched_prev_iter_basis_ft.copy()
    for rank in unique_ranks_sorted:

        # group the stuff within each batch by rank

        # shape (batch, )
        of_this_rank = (rank_values == rank)
        num_of_this_rank = np.sum(of_this_rank)

        # shape (batch, n_basis_waveforms)
        basis_selector = np.logical_and(of_this_rank[:, None], rank_include[:, :])

        # each of these has shape (num_of_this_rank * rank, )
        batch_sel, rank_sel = np.nonzero(basis_selector) # FIXME SOMEWHAT UNDEFINED BEHAVIOR HERE
        # We use the fact that batch_sel is sorted in non-decreasing order
        # to play some games with shapes
        # each unique value in batch_sel should be repeated rank number of times in a row

        batch_sel_reshape = batch_sel.reshape(num_of_this_rank, rank)
        rank_sel_reshape = rank_sel.reshape(num_of_this_rank, rank)

        selected_amplitudes = batched_amplitudes_real[batch_sel_reshape, :, rank_sel_reshape].transpose(0, 2, 1)
        selected_phases = batched_phase_delays[batch_sel_reshape, :, rank_sel_reshape].transpose(0, 2, 1)

        # solve equations with the same rank in parallel
        # shape (batch, rank <= n_basis_waveforms, n_rfft_frequencies)
        waveforms_of_rank = _batch_assemble_coefficients_and_solve(
            selected_amplitudes,
            selected_phases,
            batched_ft_observations[of_this_rank, :, :],
            batch_valid_mat[of_this_rank, :],
            n_true_frequencies,
            device,
            observation_loss_weight=None if observation_loss_weight is None else observation_loss_weight[of_this_rank, :],
            sobolev_lambda=sobolev_lambda
        )

        # now we have to reassemble the solutions
        solved_waveforms[batch_sel_reshape, rank_sel_reshape, :] = waveforms_of_rank

    return solved_waveforms


def fourier_complex_least_squares_optimize_waveforms3(amplitude_matrix_real_np: np.ndarray,
                                                      phase_delays_np: np.ndarray,
                                                      ft_complex_observations_np: np.ndarray,
                                                      n_true_frequencies: int,
                                                      device: torch.device,
                                                      sobolev_lambda: Optional[float] = None,
                                                      observation_loss_weight: Optional[
                                                          np.ndarray] = None) -> np.ndarray:
    '''

    :param amplitude_matrix_real_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param phase_delays_np: integer sample delays for each canonical waveform, for each observation
        shape (n_observations, n_canonical_waveforms)
    :param ft_complex_observations_np: complex-valued Fourier transform of the observed data,
        shape (n_observations, n_rfft_frequencies)
    :param n_true_frequencies : int, number of frequencies = n_samples for the normal FFT
        (not the number of rFFT frequencies)
    :param device: torch.device
    :param sobolev_lambda: scalar lambda for second derivative penalty for regularizing smoothness for
        the waveforms
    :param observation_loss_weight: lambda vector of weights, for weighting the contribution to the loss
        of each individual waveform. shape (n_observations, )
    :return: tuple of real component, imaginary component of canonical waveform Fourier transform
        each has shape (n_canonical_waveforms, n_rfft_frequencies)
    '''

    if observation_loss_weight is not None:
        amplitude_matrix_real_np = amplitude_matrix_real_np * observation_loss_weight[:, None]
        ft_complex_observations_np = ft_complex_observations_np * observation_loss_weight[:, None]

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
    soln = torch.linalg.solve(joint_coeff, joint_rhs[:, :, None])

    # shape (2 * n_canonical_waveforms, n_rfft_frequencies)
    soln_perm = soln.squeeze(2).permute(1, 0)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    soln_real_seg = soln_perm[:n_canonical_waveforms, :].cpu().numpy()
    soln_imag_seg = soln_perm[n_canonical_waveforms:, :].cpu().numpy()

    return soln_real_seg + 1j * soln_imag_seg
