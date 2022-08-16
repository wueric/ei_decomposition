from typing import Optional, List, Tuple, Union

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
    tot_power[tot_power == 0.0] = 1.0

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
                                           observation_loss_weight: Optional[np.ndarray] = None):
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
                                                 batched_prev_iter_basis_ft: np.ndarray,
                                                 n_true_frequencies: int,
                                                 device: torch.device,
                                                 norm_cutoff: float = 1e-2,
                                                 observation_loss_weight: Optional[np.ndarray] = None) \
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
        batch_sel, rank_sel = np.nonzero(basis_selector)  # FIXME SOMEWHAT UNDEFINED BEHAVIOR HERE
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
            observation_loss_weight=None if observation_loss_weight is None else observation_loss_weight[of_this_rank,
                                                                                 :],
        )

        # now we have to reassemble the solutions
        solved_waveforms[batch_sel_reshape, rank_sel_reshape, :] = waveforms_of_rank

    return solved_waveforms


def _pack_complex_to_real_imag(complex_valued_matrix: np.ndarray,
                               n_timepoints: int,
                               axis: int = -1) -> np.ndarray:
    '''
    Packs arbitrary complex-valued matrix into the stacked real-imaginary form,
        with the correct frequency order
    :param complex_valued_matrix: shape (... )
    :return:
    '''

    orig_shape = complex_valued_matrix.shape
    target_shape = list(orig_shape)
    target_shape[axis] = n_timepoints
    real_imag_matrix = np.zeros(target_shape, dtype=np.float32)

    real_values = np.real(complex_valued_matrix)
    imag_values = np.imag(complex_valued_matrix)

    # handle the real component; the real component has the full ((n_timepoints // 2) + 1) number of values
    # floor division here doesn't matter since n even
    real_indices_flat = np.r_[0:(n_timepoints // 2) + 1]
    expanded_real_indices_shape = [1 for _ in orig_shape]
    expanded_real_indices_shape[axis] = real_indices_flat.shape[0]
    real_indices = real_indices_flat.reshape(expanded_real_indices_shape)
    print('n_timepoints', n_timepoints)
    print(real_imag_matrix.shape, real_indices.shape, real_values.shape, axis)
    np.put_along_axis(real_imag_matrix, real_indices, real_values, axis=axis)

    # handle the imaginary component
    # we need to skip the first and last entries along axis in the even case
    # and only the first entry in the odd case
    if n_timepoints % 2 == 0:
        imag_indices_sel_flat = np.r_[1:(n_timepoints // 2)]
    else:
        imag_indices_sel_flat = np.r_[1:(n_timepoints // 2) + 1]
    selected_imag_values = np.take(imag_values, imag_indices_sel_flat, axis=axis)

    imag_indices_put_flat = np.r_[(n_timepoints // 2) + 1: n_timepoints]
    expanded_imag_indices_shape = [1 for _ in orig_shape]
    expanded_imag_indices_shape[axis] = imag_indices_put_flat.shape[0]
    imag_indices_put = imag_indices_put_flat.reshape(expanded_imag_indices_shape)

    np.put_along_axis(real_imag_matrix, imag_indices_put, selected_imag_values, axis=axis)

    return real_imag_matrix


def _unpack_real_imag_to_complex(stacked_real_imag: np.ndarray,
                                 n_timepoints: int,
                                 axis: int = -1) -> np.ndarray:
    '''
    Unpacks the stacked real-imaginary form back into a complex-valued matrix

    Inverse operation of _pack_complex_to_real_imag
    :return:
    '''
    n_rfft_freqs = (n_timepoints // 2) + 1

    orig_shape = stacked_real_imag.shape
    target_shape = list(orig_shape)
    target_shape[axis] = n_rfft_freqs

    complex_matrix = np.zeros(target_shape, dtype=np.csingle)

    # handle the imaginary component, since the imaginary component requires
    # use of put, and put is easier to use when starting with an emtpy destination
    imag_index_sel = np.r_[(n_timepoints // 2) + 1:n_timepoints]
    imag_component = np.take(stacked_real_imag, imag_index_sel, axis=axis) * 1j

    if n_timepoints % 2 == 0:
        imag_index_put_flat = np.r_[1:(n_timepoints // 2)]
    else:
        imag_index_put_flat = np.r_[1:(n_timepoints // 2) + 1]

    imag_index_put_shape = [1 for _ in orig_shape]
    imag_index_put_shape[axis] = imag_index_put_flat.shape[0]

    imag_index_put = imag_index_put_flat.reshape(imag_index_put_shape)

    np.put_along_axis(complex_matrix, imag_index_put, imag_component, axis=axis)

    # handle the real component
    # the real component consists of the first ((n_timepoints // 2) + 1) entries along axis
    # which can just be added to the final result
    real_index_sel = np.r_[0:(n_timepoints // 2) + 1]
    real_component = np.take(stacked_real_imag, real_index_sel, axis=axis)
    complex_matrix += real_component

    return complex_matrix


def construct_rfft_covariance_matrix(time_domain_covariance_matrix: np.ndarray) -> np.ndarray:
    '''
    Constructs NxN (n_timepoints x n_timepoints) real-imaginary covariance matrix
        from a NxN time-domain covariance matrix

    Facts about the rFFT matrix:
        * For an N-sequence, the rFFT matrix has shape (N // 2 + 1, N) ,
        * The 0th row, i.e. F[0, :] is the DC component, and is entirely real-valued
        * For N even, the last row, i.e. F[-1, :] is the maximum frequency component,
            and is also entirely real-valued

    Therefore, we have two cases for the rank of the real/imaginary stacked transform matrix:

    Case 1, N is even: The real matrix has rank (N / 2) + 1, and the imaginary matrix has rank (N/2) - 1,
        so the total system has rank N
    Case 2, N is odd: The real matrix has rank (N // 2 + 1), and the imaginary matrix has rank (N // 2)
        so the total system has rank N

    We need to construct both the real and imaginary matrices separately from the full rFFT matrix,
        and then stack the components

    Our variable ordering convention will be:
        * First, (N // 2 ) + 1 real-valued variables, same frequency order as rFFT
        * Imaginary-valued variables, same frequency order as rFFT

    :param time_domain_covariance_matrix: shape (..., n_timepoints, n_timepoints)
    :return: shape (..., n_timepoints, n_timepoints)
    '''
    # first we have to make the RFFT matrix
    n_timepoints = time_domain_covariance_matrix.shape[-1]

    # shape (n_rfft_freqs, n_timepoints)
    rfft_matrix = np.fft.rfft(np.eye(n_timepoints), axis=0)

    # shape (N, N), no matter if N is even or odd
    # this is G-matrix in the writeup
    real_imag_stacked_ft_matrix = _pack_complex_to_real_imag(rfft_matrix,
                                                             n_timepoints,
                                                             axis=0)

    # shape (1, N, N)
    cov_matrix = real_imag_stacked_ft_matrix[None, :, :] @ time_domain_covariance_matrix \
                 @ (real_imag_stacked_ft_matrix.T)[None, :, :]

    return cov_matrix


def DEBUG_identity_prior_optimize(
        ri_stack_ft_domain_cov_matrix: np.ndarray,
        ri_stack_basis_prior_mean_ft: np.ndarray,
        device: torch.device) -> np.ndarray:
    '''
    Goal here is to lay out the optimization in the same order as we do for
        the real thing, but to not include the real data pieces of the problem
        so that we simply get the identity

    This means we also have to figure out how to arrange the variables
        correctly. Adding the prior has the effect of coupling all of the frequencies
        of a waveform together. Since the original optimization solved separate
        systems for equations for each frequency over all of the waveforms, this results
        in one big system of equations (K waveforms * N timepoints x K * N) system

    The variable order is (real components, imaginary components) stacked by basis waveform

    (This should effectively be the computation that occurs in the no-data case,
    i.e. if a basis waveform is outright missing for a cell)
    :param ri_stack_ft_domain_cov_matrix: shape (n_basis, N, N)
    :param ri_stack_basis_prior_mean_ft: shape (n_basis, N)
    :return:
    '''

    n_basis, n_timepoints = ri_stack_basis_prior_mean_ft.shape

    # First compute the prior matrix
    ri_ft_prior_cov_mat_torch = torch.tensor(ri_stack_ft_domain_cov_matrix, dtype=torch.float32, device=device)
    ri_ft_prior_mean_torch = torch.tensor(ri_stack_basis_prior_mean_ft, dtype=torch.float32, device=device)

    with torch.no_grad():
        # first construct the prior coefficients
        # shape (n_basis, N, N) and shape (n_basis, N)
        prior_coeffs, prior_rhs_contrib = construct_prior_coefficients_and_rhs(ri_ft_prior_cov_mat_torch,
                                                                               ri_ft_prior_mean_torch)

        # then use torch put operations to pack these coefficients into something
        # that can be added to the MSE coefficients

        # variable ordering is (real components, imaginary components) stacked by basis waveform

        # shape (n_basis * N, n_basis * N)
        prior_block_diag = torch.block_diag(*[x.squeeze(0) for x in torch.split(prior_coeffs, 1, dim=0)])

        # shape (n_basis * N, )
        prior_coeffs_stacked = prior_rhs_contrib.reshape(-1)

        soln = torch.linalg.solve(prior_block_diag, prior_coeffs_stacked)

        return soln.reshape(n_basis, -1).detach().cpu().numpy()


def construct_prior_coefficients_and_rhs(prior_waveforms_ft_cov_mat_ri_stack: torch.Tensor,
                                         prior_waveforms_mean_ft_ri_stack: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    (Should return a system of equations that recovers the Fourier coefficients of the prior
        waveform basis if solved directly)
        
    :param prior_waveforms_ft_cov_mat_ri_stack: real-imag stack of the covariance matrices
        for the prior distribution,
        shape (n_basis, N, N)

        This must be a valid covariance matrix, i.e. last two dimensions should be square,
            full rank, symmetric
    :param prior_waveforms_mean_ft_ri_stack: real-imag stack of the Fourier coefficients of
        the prior mean for each basis waveform,
        shape (n_basis, N)

    :return:
    '''

    # shape (n_basis, N, N)
    with torch.no_grad():
        inv_cov_mat = torch.linalg.inv(prior_waveforms_ft_cov_mat_ri_stack)

        # shape (n_basis, N, N) @ (n_basis, N, 1)
        # -> (n_basis, N, 1) -> (n_basis, N)
        rhs = (inv_cov_mat @ prior_waveforms_mean_ft_ri_stack[:, :, None]).squeeze(2)

        return inv_cov_mat, rhs


def compute_variable_indices(frequency_num: torch.Tensor,
                             basis_num: torch.Tensor,
                             n_timepoints: int,
                             coeffs_are_imag: bool) -> torch.Tensor:
    '''
    The variable ordering is (basis 1 real, basis 1 imag; basis 2 real, basis 2 imag; ...)

    :param frequency_num: shape (...), integer-valued
    :param basis_num: shape (...), integer-valued
    :param real_or_imag:
    :param n_timepoints: int
    :return: shape (...), integer-valued
    '''

    basis_offset = n_timepoints * basis_num
    if coeffs_are_imag:
        basis_offset += (n_timepoints // 2)
    return frequency_num + basis_offset


def compute_all_eqn_real_var_placement_indices(n_timepoints: int,
                                               n_basis: int,
                                               device: torch.device) -> torch.Tensor:
    '''
    Computes the placement coefficients (for either torch.scatter or torch.scatter_add)
        for all equations (d / d_real AND d / d_imag), but only placing
        the real variables
        
    Needs a second call to compute_all_eqn_imag_var_placement_indices to be able to
        form a full rank system
        
    Indices and equation ordering correspond to the ordering
    X marks the variables whose indices are computed using this function call
                                        VARIABLES 
                           (basis 1 real, basis 1 imag, basis 2 real, basis 2 imag, ...)    
        (d / basis 1 real) |     X       |             |      X      |             |
    E   (d / basis 1 imag) |     X       |             |      X      |             |
    Q   (d / basis 2 real) |     X       |             |      X      |             |
    N   (d / basis 2 imag) |     X       |             |      X      |             |
             ...
    
    @param n_timepoints: int, number of samples = number of Fourier domain unknowns
    @param n_basis: int, number of basis waveforms
    @param device: 
    @return: torch.Tensor, shape (n_basis * n_timepoints, n_basis)
    '''

    n_real_eqns = (n_timepoints // 2) + 1
    n_imag_eqns = (n_timepoints // 2) - 1 if n_timepoints % 2 == 0 else (n_timepoints // 2)

    n_real_coeffs = n_real_eqns
    n_imag_coeffs = n_imag_eqns

    with torch.no_grad():
        d_real_freq_ix = torch.arange(0, n_real_coeffs,
                                      dtype=torch.long, device=device)[:, None].expand(-1, n_basis)
        d_real_basis_ix = torch.arange(0, n_basis,
                                       dtype=torch.long, device=device)[None, :].expand(n_real_coeffs, -1)

        # shape (n_real_eqns = n_real_coeffs, n_basis)
        d_real_real_var_ix = compute_variable_indices(d_real_freq_ix,
                                                      d_real_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=False)

        d_imag_freq_ix = torch.arange(1, n_imag_coeffs + 1, dtype=torch.long, device=device)[:, None].expand(-1,
                                                                                                             n_basis)
        d_imag_basis_ix = torch.arange(0, n_basis, dtype=torch.long, device=device)[None, :].expand(n_imag_coeffs, -1)

        # shape (n_imag_eqns = n_imag_coeffs, n_basis)
        d_imag_real_var_ix = compute_variable_indices(d_imag_freq_ix,
                                                      d_imag_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=False)

        # shape (n_timepoints, n_basis)
        all_eqns_real_placement_indices = torch.cat([d_real_real_var_ix, d_imag_real_var_ix], dim=0)

        # shape (n_basis * n_timepoints, n_basis)
        return all_eqns_real_placement_indices.repeat(n_basis, 1)


def _make_real_coeff_eqn_split_size_list(n_timepoints: int,
                                         n_basis: int) -> List[int]:
    '''

    @param n_timepoints:
    @param n_basis:
    @return:
    '''

    n_real_eqns = (n_timepoints // 2) + 1
    n_imag_eqns = (n_timepoints // 2) - 1 if n_timepoints % 2 == 0 else (n_timepoints // 2)

    output = []
    for b in range(n_basis):
        output.extend([n_real_eqns, n_imag_eqns])
    return output


def _make_imag_coeff_eqn_split_size_list(n_timepoints: int,
                                         n_basis: int) -> List[int]:
    '''
    
    @param n_timepoints: 
    @param n_basis: 
    @return: 
    '''
    n_real_eqns = (n_timepoints // 2) + 1
    n_imag_eqns = (n_timepoints // 2) - 1 if n_timepoints % 2 == 0 else (n_timepoints // 2)

    output = []
    if n_timepoints % 2 == 0:
        # even case, we need to throw away both the \omega=0 equation
        # and the \omega=2\pi equation, i.e. the first and last real equations
        for b in range(n_basis):
            output.extend([1, n_real_eqns - 2, 1, n_imag_eqns])
    else:
        # odd case, we only need to throw away the \omega=0 equation
        # i.e. the first real equation
        for b in range(n_basis):
            output.extend([1, n_real_eqns - 1, n_imag_eqns])
    return output


def chop_real_coeff_eqn_split(target_tensor: torch.Tensor,
                              n_timepoints: int,
                              n_basis: int,
                              dim: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''
    Produces views corresponding to the real entries from target_tensor,
        as well as the imaginary entries from target_tensor

    IMPORTANT: must return a bunch of VIEWS from target_tensor, not
        new tensors, otherwise we will not be able to use the output
        of this function to make in-place modifications

    @param target_tensor: shape (..., n_basis * n_timepoints, ...)
    @param n_timepoints: int
    @param n_basis: int
    @param dim: int, dimension that we want to apply the split operation on
        Note that target_tensor must have size (n_timepoints * n_basis) along
            this dimension
    @return:
    '''

    check_dim = target_tensor.shape[dim]
    if check_dim != (n_timepoints * n_basis):
        raise ValueError(f'target_tensor dim {dim} must have size {n_timepoints * n_dim}, received {check_dim}')

    slice_list = _make_real_coeff_eqn_split_size_list(n_timepoints, n_basis)
    if sum(slice_list) != (n_timepoints * n_basis):
        raise ValueError(f'Something wrong with slice_list')
    split_tensors = torch.split(target_tensor, slice_list, dim=dim)

    real_splits_by_basis = []  # type: List[torch.Tensor]
    imag_splits_by_basis = []  # type: List[torch.Tensor]
    for ii, split_tensor in enumerate(split_tensors):
        if ii % 2 == 0:
            real_splits_by_basis.append(split_tensor)
        else:
            imag_splits_by_basis.append(split_tensor)

    return real_splits_by_basis, imag_splits_by_basis


def chop_imag_coeff_eqn_split(target_tensor: torch.Tensor,
                              n_timepoints: int,
                              n_basis: int,
                              dim: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''
    Produces views corresponding to the non-zero subset of the real entries from target_tensor,
        as well as the imaginary entries from target_tensor
        
    IMPORTANT: must return a bunch of VIEWS from target_tensor, not
        new tensors, otherwise we will not be able to use the output
        of this function to make in-place modifications
        
    @param target_tensor: shape (..., n_basis * n_timepoints, ...)
    @param n_timepoints: int
    @param n_basis: int
    @param dim: int, dimension that we want to apply the split
        operation on. Note that target_tensor must have size (n_timepoints * n_basis)
        along this dimension
    @return: 
    '''

    check_dim = target_tensor.shape[dim]
    if check_dim != (n_timepoints * n_basis):
        raise ValueError(f'target_tensor dim {dim} must have size {n_timepoints * n_dim}, received {check_dim}')

    slice_list = _make_imag_coeff_eqn_split_size_list(n_timepoints, n_basis)
    if sum(slice_list) != (n_timepoints * n_basis):
        raise ValueError(f'Something wrong with slice_list')
    split_tensors = torch.split(target_tensor, slice_list, dim=dim)

    real_splits_by_basis = []  # type: List[torch.Tensor]
    imag_splits_by_basis = []  # type: List[torch.Tensor]
    if n_timepoints % 2 == 0:
        for ii, split_tensor in enumerate(split_tensors):
            if ii % 4 == 1:
                real_splits_by_basis.append(split_tensor)
            elif ii % 4 == 3:
                imag_splits_by_basis.append(split_tensor)
    else:
        for ii, split_tensor in enumerate(split_tensors):
            if ii % 3 == 1:
                real_splits_by_basis.append(split_tensor)
            elif ii % 3 == 2:
                imag_splits_by_basis.append(split_tensor)

    return real_splits_by_basis, imag_splits_by_basis


def place_all_eqn_imag_var_coeffs(all_eqns_imag_coeffs_by_basis_flat: torch.Tensor,
                                  dest_tensor: torch.Tensor,
                                  n_timepoints: int,
                                  n_basis: int,
                                  device: torch.device) -> None:
    '''
    Places coefficients for the imaginary variables only by copying the relevant
        values from imag_coeffs_by_basis_flat to dest_tensor
        
    Indices and equation ordering correspond to the ordering
    X marks the variables whose indices are computed using this function call
                                        VARIABLES
                           (basis 1 real, basis 1 imag, basis 2 real, basis 2 imag, ...)
        (d / basis 1 real) |             |      X      |             |      X      |
    E   (d / basis 1 imag) |             |      X      |             |      X      |
    Q   (d / basis 2 real) |             |      X      |             |      X      |
    N   (d / basis 2 imag) |             |      X      |             |      X      |
             ...
    
    NOTE: relies on the fact that torch.split returns a VIEW of the original tensor
    
    @param all_eqns_imag_coeffs_by_basis_flat: shape (batch, n_basis * N, n_basis)
        Tensor containing coefficients for the imaginary variables, where dim 1 corresponds
        to different equations, already in the correct order
        
        Note that for the imaginary case, we have to discard the omega=0 coefficient
        for N odd, and we have to dicard the omega=0 and omega=2\pi coefficients for N even
    @param dest_tensor: shape (batch, n_basis * N, n_basis * N), the batched square destination matrix
    @param n_timepoints: int
    @param n_basis: int
    @return: None. This function should modify dest_tensor in place
    '''

    batch = all_eqns_imag_coeffs_by_basis_flat.shape[0]

    source_split_real_block, source_split_imag_block = chop_imag_coeff_eqn_split(all_eqns_imag_coeffs_by_basis_flat,
                                                                                 n_timepoints,
                                                                                 n_basis,
                                                                                 dim=1)
    dest_split_real_block, dest_split_imag_block = chop_imag_coeff_eqn_split(dest_tensor,
                                                                             n_timepoints,
                                                                             n_basis,
                                                                             dim=1)

    # first copy the imaginary coefficients of the real equations
    for real_eqn_source, real_eqn_dest in zip(source_split_real_block, dest_split_real_block):
        # real_eqn_source has shape (batch, n_useful_real_eqns = n_imag_coeffs, n_basis)
        # real_eqn_dest has shape (batch, n_useful_real_eqns = n_imag_coeffs, n_basis * N)
        d_real_freq_ix = torch.arange(1, 1 + real_eqn_source.shape[1],
                                      dtype=torch.long, device=device)[:, None].expand(-1, n_basis)
        d_real_basis_ix = torch.arange(0, n_basis,
                                       dtype=torch.long, device=device)[None, :].expand(d_real_freq_ix.shape[0], -1)

        # shape (n_useful_real_eqns, n_basis) -> (batch, n_useful_real_eqns, n_basis)
        d_real_imag_var_ix = compute_variable_indices(d_real_freq_ix,
                                                      d_real_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=True)[None, :, :].expand(batch, -1, -1)

        real_eqn_dest.scatter_add_(2, d_real_imag_var_ix, real_eqn_source)

    # then copy the imaginary coefficients of the imaginary equations
    for imag_eqn_source, imag_eqn_dest in zip(source_split_imag_block, dest_split_imag_block):
        # real_eqn_source has shape (batch, n_imag_coeffs, n_basis)
        # real_eqn_dest has shape (batch, n_imag_coeffs, n_basis * N)
        d_imag_freq_ix = torch.arange(1, 1 + imag_eqn_source.shape[1],
                                      dtype=torch.long, device=device)[:, None].expand(-1, n_basis)
        d_imag_basis_ix = torch.arange(0, n_basis,
                                       dtype=torch.long, device=device)[None, :].expand(d_imag_freq_ix.shape[0], -1)

        # shape (n_useful_real_eqns, n_basis) -> (batch, n_useful_real_eqns, n_basis)
        d_imag_imag_var_ix = compute_variable_indices(d_imag_freq_ix,
                                                      d_imag_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=True)[None, :, :].expand(batch, -1, -1)

        imag_eqn_dest.scatter_add_(2, d_imag_imag_var_ix, imag_eqn_source)

    return


def place_all_eqn_real_var_coeffs(all_eqns_real_coeffs_by_basis_flat: torch.Tensor,
                                  dest_tensor: torch.Tensor,
                                  n_timepoints: int,
                                  n_basis: int,
                                  device: torch.device) -> None:
    '''
    Places coefficients for the real variables only by copying the relevant
        values from all_eqns_real_coeffs_by_basis_flat to dest_tensor

    Indices and equation ordering correspond to the ordering
    X marks the variables whose indices are computed using this function call
                                            VARIABLES
                           (basis 1 real, basis 1 imag, basis 2 real, basis 2 imag, ...)
        (d / basis 1 real) |     X       |             |      X      |             |
    E   (d / basis 1 imag) |     X       |             |      X      |             |
    Q   (d / basis 2 real) |     X       |             |      X      |             |
    N   (d / basis 2 imag) |     X       |             |      X      |             |
             ...

    NOTE: relies on the fact that torch.split returns a VIEW of the original tensor

    @param all_eqns_real_coeffs_by_basis_flat: shape (batch, n_basis * N, n_basis)
        Tensor containing coefficients for the imaginary variables, where dim 1 corresponds
        to different equations, already in the correct order
    @param dest_tensor: shape (batch, n_basis * N, n_basis * N), the batched square destination matrix
    @param n_timepoints: int
    @param n_basis: int
    @return: None. This function should modify dest_tensor in place
    '''

    batch = all_eqns_real_coeffs_by_basis_flat.shape[0]

    source_split_real_block, source_split_imag_block = chop_real_coeff_eqn_split(all_eqns_real_coeffs_by_basis_flat,
                                                                                 n_timepoints,
                                                                                 n_basis,
                                                                                 dim=1)

    dest_split_real_block, dest_split_imag_block = chop_real_coeff_eqn_split(dest_tensor,
                                                                             n_timepoints,
                                                                             n_basis,
                                                                             dim=1)

    # first copy the real coefficients of the real equations
    for real_eqn_source, real_eqn_dest in zip(source_split_real_block, dest_split_real_block):
        # real_eqn_source has shape (batch, n_useful_real_eqns = n_imag_coeffs, n_basis)
        # real_eqn_dest has shape (batch, n_useful_real_eqns = n_imag_coeffs, n_basis * N)
        d_real_freq_ix = torch.arange(0, real_eqn_source.shape[1],
                                      dtype=torch.long, device=device)[:, None].expand(-1, n_basis)
        d_real_basis_ix = torch.arange(0, n_basis,
                                       dtype=torch.long, device=device)[None, :].expand(d_real_freq_ix.shape[0], -1)

        # shape (n_useful_real_eqns, n_basis) -> (batch, n_useful_real_eqns, n_basis)
        d_real_imag_var_ix = compute_variable_indices(d_real_freq_ix,
                                                      d_real_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=False)[None, :, :].expand(batch, -1, -1)

        real_eqn_dest.scatter_add_(2, d_real_imag_var_ix, real_eqn_source)

    # then copy the imag coefficients of the real equations
    for imag_eqn_source, imag_eqn_dest in zip(source_split_imag_block, dest_split_imag_block):
        # real_eqn_source has shape (batch, n_imag_coeffs, n_basis)
        # real_eqn_dest has shape (batch, n_imag_coeffs, n_basis * N)
        d_imag_freq_ix = torch.arange(1, 1 + imag_eqn_source.shape[1],
                                      dtype=torch.long, device=device)[:, None].expand(-1, n_basis)
        d_imag_basis_ix = torch.arange(0, n_basis,
                                       dtype=torch.long, device=device)[None, :].expand(d_imag_freq_ix.shape[0], -1)

        # shape (n_useful_real_eqns, n_basis) -> (batch, n_useful_real_eqns, n_basis)
        d_imag_imag_var_ix = compute_variable_indices(d_imag_freq_ix,
                                                      d_imag_basis_ix,
                                                      n_timepoints,
                                                      coeffs_are_imag=False)[None, :, :].expand(batch, -1, -1)

        imag_eqn_dest.scatter_add_(2, d_imag_imag_var_ix, imag_eqn_source)

    return


def rearrange_mse_grouped_coefficients(eq1_group_real_coeffs: torch.Tensor,
                                       eq1_group_imag_coeffs: torch.Tensor,
                                       eq1_group_rhs: torch.Tensor,
                                       eq2_group_real_coeffs: torch.Tensor,
                                       eq2_group_imag_coeffs: torch.Tensor,
                                       eq2_group_rhs: torch.Tensor,
                                       n_timepoints: int) -> Tuple[torch.Tensor, torch.Tensor]:
    '''

    Rows are different equations, corresponding to derivative of MSE objective
        w.r.t. the corresponding Fourier coefficients
    Columns correspond to the variables, which are the Fourier coefficients

    variable ordering is (real components, imaginary components) stacked by basis waveform

    :param eq1_group_real_coeffs: shape (batch, n_basis, n_basis, n_rfft_freqs)
        These are the real coefficients that come from differentiating the MSE loss w.r.t. the real component

        Note that we form eq1 group by catting eq1_group_real_coeffs and eq1_group_imag_coeffs along dim 2
    :param eq1_group_imag_coeffs: shape (batch, n_basis, n_basis, n_rfft_freqs)
        These are the imag coefficients that come from differentiating the MSE loss w.r.t. the real component

        Note that we form eq1 group by catting eq1_group_real_coeffs and eq1_group_imag_coeffs along dim 2
    :param eq1_group_rhs: shape (batch, n_basis, n_rfft_frequencies)
    :param eq2_group_real_coeffs: shape (batch, n_basis, n_basis, n_rfft_freqs)
        These are the real coefficients that come from differentiating the MSE loss w.r.t. the imag component

        Note that we form eq2 group by catting eq2_group_real_coeffs and eq2_group_imag_coeffs along dim 2
    :param eq2_group_imag_coeffs: shape (batch, n_basis, n_basis, n_rfft_freqs)
        These are the imag coefficients that come from differentiating the MSE loss w.r.t. the imag component

        Note that we form eq2 group by catting eq2_group_real_coeffs and eq2_group_imag_coeffs along dim 2
    :param eq2_group_rhs: shape (batch, n_basis, n_rfft_frequencies)
    :param n_timepoints: int, number of time samples
    :return:

    Comments on variable ordering, conjugate symmetry of the Fourier transform, number of variables
    
    FACT: rfft gives back (N // 2) + 1 complex-valued coefficients. 
        * If N is even, the 0^th coefficient and the last coefficient must be real-valued
        
        This means that that there are N / 2 + 1  real coefficients, and N / 2 - 1 imaginary
        coefficients if we split real and imaginary, for a total of N coefficients
        
        * If N is odd, the 0^th coefficient must be real-valued
        
        This means there are (N+1) / 2 real coefficients, and (N+1) / 2 - 1 imaginary coefficients,
        so if we split real and imaginary, we have a total of N coefficients
        
    This means that the first imaginary coefficient, and possibly the last imaginary coefficient
        from the original set of equations must be handled as special cases
        
    FACT: Delay in the Fourier domain corresponds to multiplication by $e^{-j \omega n}$ where n is
        the number of delay samples. This is generally uninteresting, except in the case of the 0^th coefficient
        where $\omega = 0$ or in the last case of the last coefficient of $N$ is even, where $\omega = 2 \pi$.

        * If $\omega = 0$, $e^{-j \omega n} = e^{0} = 1$ and therefore the delay factor is real-valued.
            
            This means that the coefficients corresponding to imaginary variables of equation group 1 
            (differentiating w.r.t. the real component) for $\omega=0$ are zero.
            
            This also means that the coefficients corresponding to the imaginary variables of equatino group 2
            (differentiating w.r..t the imag component) for $\omega=0$ are purely real-valued and in fact
            are all the same, i.e. this ends up requiring that the solution be 0 for those variables.
            
        * If $\omega = 2\pi$, $e^{-j \omega n} = e^{-2 \pi \omega n} = 1$, and the same thing as above happens.
    '''

    with torch.no_grad():
        batch, n_basis = eq1_group_real_coeffs.shape[:2]
        device = eq1_group_real_coeffs.device

        ##########################################################################
        # Reimplementation here
        output_tensor = torch.zeros((batch, n_timepoints * n_basis, n_timepoints * n_basis),
                                    dtype=eq1_group_real_coeffs.dtype, device=device)

        # all shape (batch, n_basis, n_rfft_freqs, n_basis)
        eq1_group_real_perm = eq1_group_real_coeffs.permute(0, 1, 3, 2)
        eq1_group_imag_perm = eq1_group_imag_coeffs.permute(0, 1, 3, 2)
        eq2_group_real_perm = eq2_group_real_coeffs.permute(0, 1, 3, 2)
        eq2_group_imag_perm = eq2_group_imag_coeffs.permute(0, 1, 3, 2)

        if n_timepoints % 2 == 0:
            # all shape (batch, n_basis, n_rfft_freqs - 2, n_basis)
            eq2_group_real_perm_relev = eq2_group_real_perm[:, :, 1:-1, :]
            eq2_group_imag_perm_relev = eq2_group_imag_perm[:, :, 1:-1, :]

            # shape (batch, n_basis, n_rfft_freqs - 2)
            eq2_group_rhs_relev = eq2_group_rhs[:, :, 1:-1]
        else:
            # all shape (batch, n_basis, n_rfft_freqs - 1, n_basis)
            eq2_group_real_perm_relev = eq2_group_real_perm[:, :, 1:, :]
            eq2_group_imag_perm_relev = eq2_group_imag_perm[:, :, 1:, :]

            # shape (batch, n_basis, n_rfft_freqs - 1)
            eq2_group_rhs_relev = eq2_group_rhs[:, :, 1:]

        # all shape (batch, n_basis, N, n_basis)
        real_coeff_grouped_by_basis_stack = torch.cat([eq1_group_real_perm, eq2_group_real_perm_relev], dim=2)
        imag_coeff_grouped_by_basis_stack = torch.cat([eq1_group_imag_perm, eq2_group_imag_perm_relev], dim=2)

        # all shape (batch, n_basis * N, n_basis)
        real_coeff_grouped_by_basis_flat = real_coeff_grouped_by_basis_stack.reshape(batch, -1, n_basis)
        imag_coeff_grouped_by_basis_flat = imag_coeff_grouped_by_basis_stack.reshape(batch, -1, n_basis)

        place_all_eqn_real_var_coeffs(real_coeff_grouped_by_basis_flat,
                                      output_tensor,
                                      n_timepoints,
                                      n_basis,
                                      device)
        place_all_eqn_imag_var_coeffs(imag_coeff_grouped_by_basis_flat,
                                      output_tensor,
                                      n_timepoints,
                                      n_basis,
                                      device)

        # shape (batch, n_basis, N)
        rhs_grouped_by_basis_stack = torch.cat([eq1_group_rhs, eq2_group_rhs_relev], dim=2)

        # shape (batch, n_basis * N)
        rhs_tensor = rhs_grouped_by_basis_stack.reshape(batch, -1)

        return output_tensor, rhs_tensor


def rearrange_prior_grouped_coefficients(ri_stack_ft_domain_cov_matrix: torch.Tensor,
                                         ri_stack_basis_prior_mean_ft: torch.Tensor,
                                         prior_lambdas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Restructures the equations into the full-size form for
        the contributions from the Gaussian prior

    Should be full rank, and block diagonal

    :param ri_stack_ft_domain_cov_matrix: real-imag stack of the covariance matrices
        for the prior distribution,
        shape (n_basis, N, N)

    :param ri_stack_basis_prior_mean_ft: real-imag stack of the Fourier coefficients of
        the prior mean for each basis waveform,
        shape (n_basis, N)

        This must be a valid covariance matrix, i.e. last two dimensions should be square,
            full rank, symmetric
    :param prior_lambdas: lambda weights on the Gaussian prior term
        shape (n_basis, ), allowing each cell and basis waveform to have an individually-controlled
        prior

    :return: (n_basis * N, n_basis * N) , corresponding to the coefficients of the
        linear system of equations for each cell,

        and
            (n_basis * N), corresponding to the RHS of the linear systems for each cell
    '''

    with torch.no_grad():
        # first construct the prior coefficients
        # shape (n_basis, N, N) and shape (n_basis, N)
        prior_coeffs, prior_rhs_contrib = construct_prior_coefficients_and_rhs(ri_stack_ft_domain_cov_matrix,
                                                                               ri_stack_basis_prior_mean_ft)

        # shape (n_basis, N, N) and shape (n_basis, N)
        prior_coeffs = prior_coeffs * prior_lambdas[:, None, None]
        prior_rhs_contrib = prior_rhs_contrib * prior_lambdas[:, None]

        # then use torch put operations to pack these coefficients into something
        # that can be added to the MSE coefficients

        # variable ordering is (real components, imaginary components) stacked by basis waveform

        # shape (n_basis * N, n_basis * N)
        prior_block_diag = torch.block_diag(*[x.squeeze(0) for x in torch.split(prior_coeffs, 1, dim=0)])

        # shape (n_basis * N)
        prior_coeffs_stacked = prior_rhs_contrib.reshape(-1, )

        return prior_block_diag, prior_coeffs_stacked


def batch_fourier_complex_least_square_with_prior_optimize(
        batched_amplitudes_real: np.ndarray,
        batched_phase_delays: np.ndarray,
        batched_ft_observations: np.ndarray,
        batched_valid_mat: np.ndarray,
        ri_stack_ft_domain_cov_matrix: np.ndarray,
        ri_stack_basis_prior_mean_ft: np.ndarray,
        n_true_frequencies: int,
        regularization_lambda: Union[np.ndarray, float],
        device: torch.device,
        observation_loss_weight: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Used to solve basis waveform optimization problems in parallel,
        where we have a prior on the mean and have a covariance matrix describing
        the temporal correlation structure between samples in time domain

    Note that once we include the prior, the problem is no longer separable by component
        in frequency domain. Rather than solve a bunch of 2K x 2K systems, we solve a
        single NK x NK system where N is the number of time samples.
        
    Reminder about the original equation grouping and variable ordering:
        * Eqn group 1 refers to the set of K equations in 2K variables that are produced
            by differentiating the MSE objective w.r.t. the real component of the F^th Fourier
            coefficient of each basis waveform
        * Eqn group 2 refers to the set of K equations in 2K variables that are produced
            by differentiating the MSE objective w.r.t. the imaginary component of the F^th Fourier
            coefficient of each basis waveform

        * The variable ordering is all real coefficients, followed by all imag coefficients


    :param batched_amplitudes_real: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_phase_delays: integer sample delays for each canonical waveform, for each observation
        shape (batch, n_observations, n_basis_waveforms)
    :param batched_ft_observations: complex-valued Fourier transform of the observed data,
        shape (batch, n_observations, n_rfft_frequencies)
    :param batched_valid_mat: boolean matrix marking which entries of the above matrices correspond to real data,
        and which entries correspond to padding.
        shape (batch, n_observations), boolean valued
    :param ri_stack_ft_domain_inv_cov_matrix: shape (N = n_timepoints, N = n_timepoints)
    :param ri_stack_basis_prior_mean_ft: shape (n_basis_waveforms, N = n_timepoints)
    :param regularization_lambda: float, lambda scale factor for the waveform, or np.array, shape (batch, n_basis)
        if we want to specify a different regularization coefficient for each basis (for example, if we're
        much more certain about one type of basis waveform vs another)
    :param device:
    :param observation_loss_weight:
    :return:
    '''

    batch, n_observations, n_basis = batched_amplitudes_real.shape
    _, _, n_rfft_frequencies = batched_ft_observations.shape

    if isinstance(regularization_lambda, float):
        regularization_lambda = np.ones((n_basis,), dtype=np.float32) * regularization_lambda

    # coefficient assembly strategy (i.e. how do we compute and arrange the coefficients of the linear system to
    # avoid confusing ourselves?)
    # In the real-imag stacked version of covariance matrices and vectors, the frequencies are all-over-the-place
    # so we need to use _pack and _unpack to arrange the frequency coefficients of the phases and waveform
    # Fourier transforms

    if observation_loss_weight is not None:
        batched_amplitudes_real = batched_amplitudes_real * observation_loss_weight[:, :, None]
        batched_ft_observations = batched_ft_observations * observation_loss_weight[:, :, None]

    # First compute the prior matrix
    ri_ft_prior_cov_mat_torch = torch.tensor(ri_stack_ft_domain_cov_matrix, dtype=torch.float32, device=device)
    ri_ft_prior_mean_torch = torch.tensor(ri_stack_basis_prior_mean_ft, dtype=torch.float32, device=device)
    regularization_lambda_torch = torch.tensor(regularization_lambda, dtype=torch.float32, device=device)

    # prior_coeffs has shape (n_basis * N, n_basis * N)
    # prior_Rhs has shape (n_basis * N)
    prior_coeffs, prior_rhs = rearrange_prior_grouped_coefficients(ri_ft_prior_cov_mat_torch,
                                                                   ri_ft_prior_mean_torch,
                                                                   regularization_lambda_torch)

    # shape (batch, n_observations)
    valid_one_matrix = torch.tensor(batched_valid_mat.astype(np.float32), dtype=torch.float32, device=device)

    # shape (batch, )
    n_valid_per_batch = torch.sum(valid_one_matrix, dim=1)

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

    # shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    ################# (l, k, f) ############### (l, k, None) ########################### (l, None, f)
    eq1_rhs_re = real_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * real_observe_ft_torch[:, :,
                                                                                         None, :]
    eq1_rhs_im = imag_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * imag_observe_ft_torch[:, :,
                                                                                         None, :]

    # shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    eq1_rhs = torch.sum((eq1_rhs_re + eq1_rhs_im) * valid_one_matrix[:, :, None, None], dim=1)

    # shape (batch, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs_p = real_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * imag_observe_ft_torch[:, :,
                                                                                        None, :]
    eq2_rhs_m = imag_phase_mat_torch[:, :, :, :] * amplitude_mat_torch[:, :, :, None] * real_observe_ft_torch[:, :,
                                                                                        None, :]

    # shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    eq2_rhs = torch.sum((eq2_rhs_p - eq2_rhs_m) * valid_one_matrix[:, :, None, None], dim=1)

    mse_matrix_coeffs, mse_rhs = rearrange_mse_grouped_coefficients(
        eq1_group_real_coeff,
        eq1_group_imag_coeff,
        eq1_rhs,
        -eq1_group_imag_coeff,
        eq1_group_real_coeff,
        eq2_rhs,
        n_true_frequencies
    )

    # full_coeffs has shape (batch, n_basis * N, n_basis * N)
    # full_rhs has shape (batch, n_basis * N)
    full_coeffs = mse_matrix_coeffs + prior_coeffs[None, :, :]
    full_rhs = mse_rhs + prior_rhs[None, :]

    # shape (batch, n_basis * N) -> (batch, n_basis, N)
    soln_ri = torch.linalg.solve(full_coeffs, full_rhs[:, :, None]).squeeze(2).reshape(batch, n_basis, -1)

    soln_ri_np = soln_ri.detach().cpu().numpy()

    # shape (batch, n_basis, n_rfft_frequencies), complex-valued
    soln_rfft = _unpack_real_imag_to_complex(soln_ri_np, n_true_frequencies, axis=2)

    return soln_rfft


def fourier_complex_least_squares_optimize_waveforms3(amplitude_matrix_real_np: np.ndarray,
                                                      phase_delays_np: np.ndarray,
                                                      ft_complex_observations_np: np.ndarray,
                                                      n_true_frequencies: int,
                                                      device: torch.device,
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
