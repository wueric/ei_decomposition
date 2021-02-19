import torch
import torch.nn as nn

import numpy as np

import tqdm

import electrode_map as el_map

from typing import Dict, Tuple, List, Optional, Union, Callable, Any

from lib.util_fns import EIDecomposition, bspline_upsample_waveforms_padded_by_cell, \
    make_electrode_padded_ei_data_matrix, make_spatial_neighbors_mean_matrix, \
    pack_by_cell_into_flat, unpack_flat_into_by_cell, \
    pack_by_cell_amplitudes_and_phases_into_ei_shape, one_pad_disused_by_cell, \
    grab_above_threshold_electrodes_and_order, pack_full_by_cell_into_matrix_by_cell, get_neighbor_indices_from_adj_mat

from lib.ei_decomposition import debug_evaluate_error
from lib.joint_amplitude_time_optimization import coarse_to_fine_time_shifts_and_amplitudes, \
    make_by_cell_weighted_l1_regularizer
from lib.frequency_domain_optimization import fourier_complex_least_squares_optimize_waveforms3


def make_sparse_coord_descent_mean_connectivity_regularize_fn(adjacency_mean_mat_by_cell: np.ndarray,
                                                              dividable_data_waveform_magnitudes: np.ndarray,
                                                              all_amplitude_matrix: np.ndarray,
                                                              lambda_spatial: float,
                                                              electrode_idx: int,
                                                              electrode_neighbor_indices_by_cell: np.array,
                                                              device: torch.device) \
        -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    '''
    Makes lambda functions to calculate both the value of |AM-A|_F^2 spatial continuity penalty term (but in terms
        of A' instead of A), as well as the gradient of the penalty term with respect to A', where A' is
        the normalized amplitude matrix rather than the amplitude matrix

    Idea here is that the adjacency matrix is sparse (each electrode has only a few nearest neighbors), and so we
        can perform most of the calculation with a small submatrix of the adjacency matrix and get the same result

    :param adjacency_mean_mat_by_cell: Adjacency mean matrix for each cell,
        shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param dividable_data_waveform_magnitudes: L2 norms of the data waveforms, arranged in by-cell format. Entries
        corresponding to disused electrodes have value 1, because we use this matrix for multiplication and division
        only. Shape (n_cells, max_n_electrodes)
    :param all_amplitude_matrix: previously fitted normalized amplitudes A' for each cell (with rescaling, these amplitudes
        will not fit the raw EI). shape (n_cells, max_n_electrodes, n_canonical_waveforms). Entries corresponding to
        unused electrodes must be zero
    :param lambda_spatial: scalar multiple lambda for the spatial continuity penalty
    :param electrode_idx: the index of the center electrode that we are working with
    :param electrode_neighbor_indices_by_cell: shape (n_cells, ?), ragged np.ndarray, indices of the relevant neighbors of this
        electrode. Note that the neighbors will be different for every cell, since we're considering
        different electrodes for each cell in parallel. Some of the entries may be None, if the electrode is disused
    :param device:
    :return:
    '''

    n_cells, max_n_electrodes, _ = adjacency_mean_mat_by_cell.shape
    _, _, n_canonical_waveforms = all_amplitude_matrix.shape

    max_n_submatrix_electrodes = -1  # type: int
    for cell_idx in range(n_cells):
        max_submatrix_electrodes = max(len(electrode_neighbor_indices_by_cell[cell_idx]) + 1,
                                       max_n_submatrix_electrodes)

    ###### Calculate 1 / diag |X|_2 #####################################################################
    ###### and pack the M-I submatrices #################################################################
    ###### also select the subset of electrode ampltiudes that matter for the calculation ###############
    # this is 1 / diag |X|_2
    # shape (n_cells, max_n_submatrix_electrodes)
    one_over_mag_diag_by_cell = np.zeros((n_cells, max_n_submatrix_electrodes), dtype=np.float32)

    # this is M
    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices = np.zeros((n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes),
                                dtype=np.float32)

    # this is M-I
    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices_minus_i = np.zeros((n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes),
                                        dtype=np.float32)

    # this is the relevant amplitude submatrix
    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_submatrices = np.zeros((n_cells, max_n_submatrix_electrodes, n_canonical_waveforms),
                                     dtype=np.float32)

    for cell_idx in range(n_cells):
        neighbor_indices = electrode_neighbor_indices_by_cell[cell_idx]

        if neighbor_indices is not None:
            included_els = [electrode_idx, ] + neighbor_indices
            n_included_els = len(included_els)

            mean_submatrix = adjacency_mean_mat_by_cell[cell_idx, np.ix_(included_els, included_els)].copy()
            mean_submatrices[cell_idx, :n_included_els, :n_included_els] = mean_submatrix

            mean_submatrix_minus_ident = mean_submatrix - np.eye(n_included_els, dtype=np.float32)
            mean_submatrices_minus_i[cell_idx, :n_included_els, :n_included_els] = mean_submatrix_minus_ident

            one_over_mag_diag_by_cell[cell_idx, :n_included_els] = 1.0 / dividable_data_waveform_magnitudes[
                cell_idx, included_els]

            amplitude_submatrices[cell_idx, :n_included_els, :] = all_amplitude_matrix[cell_idx, included_els, :]

    ###### Transfer stuff over to GPU and calculate shared quantities ##########################

    # shape (n_cells, max_n_submatrix_electrodes)
    one_over_mag_diag_torch = torch.tensor(one_over_mag_diag_by_cell, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices_minus_i_torch = torch.tensor(mean_submatrices_minus_i, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    m_i_t_m_i = mean_submatrices_minus_i_torch.permute(0, 2, 1) @ mean_submatrices_minus_i_torch

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    diag_m_i_t_m_i_diag = one_over_mag_diag_torch[:, None, :] * m_i_t_m_i * one_over_mag_diag_torch[:, :, None]

    # shape (n_cells, max_n_submatrix_electrodes)
    diag_m_i_t_m_i_diag_relevant = diag_m_i_t_m_i_diag[:, :, 0]

    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_mat_torch = torch.tensor(amplitude_submatrices.transpose((0, 2, 1)), dtype=torch.float32, device=device)

    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_mat_torch_without_electrode = amplitude_mat_torch.clone()
    amplitude_mat_torch_without_electrode[:, :, 0] = 0.0

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
    mean_submatrices_torch = torch.tensor(mean_submatrices, dtype=torch.float32, device=device)

    def gradient_callable(batched_normalized_electrode_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Gradient w.r.t. amplitudes of the spatial continuity penalty, with lambda included

        :param batched_normalized_amplitudes: shape (n_cells, batch_size, n_canonical_waveforms), with normalization
            This is A', which if used directly will not fit the raw EIs
        :return: gradient, shape (n_cells, batch_size, n_canonical_waveforms)
        '''

        _, batch_size, _ = batched_normalized_electrode_amplitudes.shape

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_submatrix_electrodes)
        batched_amplitude_mat = amplitude_mat_torch_without_electrode[:, None, :, :].repeat(1, batch_size, 1, 1)
        batched_amplitude_mat[:, :, :, 0] += batched_normalized_electrode_amplitudes

        # shape (n_cells, batch_size, n_canonical_waveforms)
        all_grad = (batched_amplitude_mat @ diag_m_i_t_m_i_diag_relevant[:, None, :, None]).squeeze(3)

        # shape (n_cells, batch_size, n_canonical_waveforms)
        return all_grad * lambda_spatial

    def loss_callable(batched_normalized_electrode_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Value of the spatial continuity penalty, with lambda included

        Note that the implementation of this needs to careful to avoid including the unused electrodes at the
            end of the data

        :param batched_normalized_amplitudes: shape (n_cells, batch_size, n_canonical_waveforms), with normalization
            This is A', which if used directly will not fit the raw EIs
        :return: loss value, shape (n_cells, ...)
        '''
        _, batch_size, _ = batched_normalized_electrode_amplitudes.shape

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_submatrix_electrodes)
        batched_amplitude_mat = amplitude_mat_torch_without_electrode[:, None, :, :].repeat(1, batch_size, 1, 1)
        batched_amplitude_mat[:, :, :, 0] += batched_normalized_electrode_amplitudes

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_submatrix_electrodes)
        a_prime_diag = batched_amplitude_mat * one_over_mag_diag_torch[:, None, None, :]

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_submatrix_electrodes)
        a_prime_diag_m = a_prime_diag @ mean_submatrices_torch[:, None, :, :]

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_submatrix_electrodes)
        diff_matrix = a_prime_diag_m - a_prime_diag

        # shpae (n_cells, batch_size)
        frob_norm = torch.sum(diff_matrix * diff_matrix, dim=(3, 4))

        return lambda_spatial * frob_norm / 2.0

    return gradient_callable, loss_callable


def make_coord_descent_mean_connectivity_regularize_fn3(adjacency_mean_mat_by_cell: np.ndarray,
                                                        dividable_data_waveform_magnitudes: np.ndarray,
                                                        all_amplitude_matrix: np.ndarray,
                                                        last_valid_indices: np.ndarray,
                                                        lambda_spatial: float,
                                                        electrode_idx: int,
                                                        device: torch.device) \
        -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    '''
    Makes lambda functions to calculate both the value of |AM-A|_F^2 spatial continuity penalty term (but in terms
        of A' instead of A), as well as the gradient of the penalty term with respect to A', where A' is
        the normalized amplitude matrix rather than the amplitude matrix

    :param adjacency_mean_mat_by_cell: Adjacency mean matrix for each cell,
        shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param dividable_data_waveform_magnitudes: L2 norms of the data waveforms, arranged in by-cell format. Entries
        corresponding to disused electrodes have value 1, because we use this matrix for multiplication and division
        only. Shape (n_cells, max_n_electrodes)
    :param all_amplitude_matrix: previously fitted normalized amplitudes A' for each cell (with rescaling, these amplitudes
        will not fit the raw EI). shape (n_cells, max_n_electrodes, n_canonical_waveforms). Entries corresponding to
        unused electrodes must be zero
    :param last_valid_indices: shape (n_cells, ), number of valid electrodes for each cell
    :param lambda_spatial: scalar multiple lambda for the spatial continuity penalty
    :param electrode_idx: the index of the center electrode that we are working with
    :param device:
    :return:
    '''

    n_cells, max_n_electrodes, _ = adjacency_mean_mat_by_cell.shape
    _, _, n_canonical_waveforms = all_amplitude_matrix.shape

    # shape (n_cells, max_n_electrodes)
    one_over_mag_diag_by_cell = np.zeros((n_cells, max_n_electrodes), dtype=np.float32)

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
    mean_mat_by_cell_minus_ident = np.zeros_like(adjacency_mean_mat_by_cell, dtype=np.float32)
    for cell_idx in range(n_cells):
        n_used_electrodes = last_valid_indices[cell_idx]
        one_over_mag_diag_by_cell[
        cell_idx, :n_used_electrodes] = 1.0 / dividable_data_waveform_magnitudes[cell_idx, :n_used_electrodes]

        mean_mat_by_cell_minus_ident[cell_idx, :n_used_electrodes, :n_used_electrodes] = adjacency_mean_mat_by_cell[
                                                                                         cell_idx, :n_used_electrodes,
                                                                                         :n_used_electrodes]
        mean_mat_by_cell_minus_ident[cell_idx, np.r_[:n_used_electrodes], np.r_[:n_used_electrodes]] -= 1.0

    # shape (n_cells, max_n_electrodes)
    one_over_mag_diag_torch = torch.tensor(one_over_mag_diag_by_cell, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
    mean_mat_by_cell_minus_ident_torch = torch.tensor(mean_mat_by_cell_minus_ident, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
    m_i_t_m_i = mean_mat_by_cell_minus_ident_torch.permute(0, 2, 1) @ mean_mat_by_cell_minus_ident_torch

    diag_m_i_t_m_i_diag = one_over_mag_diag_torch[:, None, :] * m_i_t_m_i * one_over_mag_diag_torch[:, :, None]

    # shape (n_cells, n_canonical_waveforms, max_n_electrodes)
    amplitude_mat_torch = torch.tensor(all_amplitude_matrix.transpose((0, 2, 1)), dtype=torch.float32, device=device)

    # shape (n_cells, n_canonical_waveforms, max_n_electrodes)
    amplitude_mat_torch_without_electrode = amplitude_mat_torch.clone()
    amplitude_mat_torch_without_electrode[:, :, electrode_idx] = 0.0

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
    mean_mat_torch = torch.tensor(adjacency_mean_mat_by_cell, dtype=torch.float32, device=device)

    def gradient_callable(batched_normalized_electrode_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Gradient w.r.t. amplitudes of the spatial continuity penalty, with lambda included

        :param batched_normalized_amplitudes: shape (n_cells, batch_size, n_canonical_waveforms), with normalization
            This is A', which if used directly will not fit the raw EIs
        :return: gradient, shape (n_cells, batch_size, n_canonical_waveforms)
        '''

        _, batch_size, _ = batched_normalized_electrode_amplitudes.shape

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_electrodes
        batched_amplitude_mat = amplitude_mat_torch_without_electrode[:, None, :, :].repeat(1, batch_size, 1, 1)
        batched_amplitude_mat[:, :, :, electrode_idx] += batched_normalized_electrode_amplitudes

        diag_m_i_t_m_i_diag_relevant = diag_m_i_t_m_i_diag[:, :, electrode_idx]

        # shape (n_cells, batch_size, n_canonical_waveforms)
        all_grad = (batched_amplitude_mat @ diag_m_i_t_m_i_diag_relevant[:, None, :, None]).squeeze(3)

        # shape (n_cells, batch_size, n_canonical_waveforms)
        return all_grad * lambda_spatial

    def loss_callable(batched_normalized_electrode_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Value of the spatial continuity penalty, with lambda included

        Note that the implementation of this needs to careful to avoid including the unused electrodes at the
            end of the data

        :param batched_normalized_amplitudes: shape (n_cells, batch_size, n_canonical_waveforms), with normalization
            This is A', which if used directly will not fit the raw EIs
        :return: loss value, shape (n_cells, ...)
        '''
        _, batch_size, _ = batched_normalized_electrode_amplitudes.shape

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_electrodes
        batched_amplitude_mat = amplitude_mat_torch_without_electrode[:, None, :, :].repeat(1, batch_size, 1, 1)
        batched_amplitude_mat[:, :, :, electrode_idx] += batched_normalized_electrode_amplitudes

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_electrodes)
        a_prime_diag = batched_amplitude_mat * one_over_mag_diag_torch[:, None, None, :]
        print(a_prime_diag.shape, mean_mat_torch.shape)

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_electrodes)
        a_prime_diag_m = a_prime_diag @ mean_mat_torch[:, None, :, :]

        # shape (n_cells, batch_size, n_canonical_waveforms, max_n_electrodes)
        diff_matrix = a_prime_diag_m - a_prime_diag

        # shpae (n_cells, batch_size)
        frob_norm = torch.sum(diff_matrix * diff_matrix, dim=(3, 4))

        return lambda_spatial * frob_norm / 2.0

    return gradient_callable, loss_callable


def search_with_coordinate_descent_projected(observed_ft_by_cell: np.ndarray,
                                             ft_canonical: np.ndarray,
                                             n_true_frequencies: int,
                                             last_valid_indices: np.ndarray,
                                             neighborhood_mean_mat: np.ndarray,
                                             prev_iter_amplitudes_by_cell: np.ndarray,
                                             normalization_scale_factor: np.ndarray,
                                             spatial_regularization_lambda: float,
                                             valid_phase_shift_range: Tuple[int, int],
                                             first_pass_step_size: int,
                                             second_pass_best_n: int,
                                             second_pass_width: int,
                                             device: torch.device,
                                             l1_regularization_lambda: Optional[float] = None,
                                             amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                                             least_squares_converge_epsilon: float = 1e-3,
                                             max_batch_size: Optional[int] = 8192) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Does one pass of coordinate descent on the amplitudes and time shifts, parallelizing by cell
        The core algorithm is grid search over a series of nonnegative least squares problems with
        regularization, solved using projected gradient descent

    :param observed_ft_by_cell: observed waveforms by cell in Fourier domain, complex valued,
        shape (n_cells, max_n_electrodes, n_rfft_frequencies). May contain zero-valued vectors.
        Potentially scaled, if it is scaled then normalization_scale_factor cannot None
    :param ft_canonical: basis waveforms in Fourier domain, unshifted, complex valued,
        shape (n_canonical_waveforms, n_rfft_frequencies)
    :param n_true_frequencies: number of actual (not rFFT) FFT frequencies
    :param last_valid_indices: shape (n_cells, ), integer-valued, number of valid electrodes for each cell in
        observed_ft_by_cell
    :param neighborhood_mean_mat: mean adjacency matrix for each cell, shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param prev_iter_amplitudes_by_cell: initial amplitudes, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param normalization_scale_factor: Scale factor that was applied to the data waveforms,
        shape (n_cells, max_n_electrodes). Cannot contain any zeros for disused electrodes, so need to take care of
        this in the caller of this function
    :param valid_phase_shift_range: range of sample shifts to consider
    :param first_pass_step_size: int, step size for the first pass grid search
    :param second_pass_best_n: int, algorithm expands upon the second_pass_best_n best results from the first pass
        grid search to do the second pass fine search
    :param second_pass_width: int, one-sided width around the second_pass_best_n best results from the first pass girid
        search that the algorithm searches in the second pass
    :param spatial_regularization_lambda: spatial regularization penalty weight
    :param device: torch.device
    :param l1_regularization_lambda: L1 basis waveform weight penalty
    :param least_squares_converge_epsilon:
    :return:
    '''

    n_cells, max_n_electrodes, n_rfft_frequencies = observed_ft_by_cell.shape
    n_canonical_waveforms, _ = ft_canonical.shape

    # shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    amplitudes_cd_matrix = prev_iter_amplitudes_by_cell.copy()

    # shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    shifts_cd_matrix = np.zeros((n_cells, max_n_electrodes, n_canonical_waveforms), dtype=np.int32)

    # solve the problems for each cell in parallel by electrode index
    pbar = tqdm.tqdm(total=max_n_electrodes, leave=False, desc='Optimization by electrode')
    for electrode_idx in range(max_n_electrodes):

        kill_problems = (electrode_idx >= last_valid_indices) # shape (n_cells, )

        electrode_nn_mat = get_neighbor_indices_from_adj_mat(neighborhood_mean_mat, electrode_idx)
        spatial_regularizer_callable = make_sparse_coord_descent_mean_connectivity_regularize_fn(neighborhood_mean_mat,
                                                                                                 normalization_scale_factor,
                                                                                                 amplitudes_cd_matrix,
                                                                                                 spatial_regularization_lambda,
                                                                                                 electrode_idx,
                                                                                                 electrode_nn_mat,
                                                                                                 device)

        l1_regularizer_callable = None
        if l1_regularization_lambda is not None:
            l1_scale_factor = 1.0 / (
                        normalization_scale_factor[:, electrode_idx] * normalization_scale_factor[:, electrode_idx])
            l1_regularizer_callable = make_by_cell_weighted_l1_regularizer(l1_scale_factor,
                                                                           l1_regularization_lambda,
                                                                           device)

        parallel_amplitudes_el, parallel_shifts_el = coarse_to_fine_time_shifts_and_amplitudes(
            observed_ft_by_cell[:, electrode_idx, :],
            ft_canonical,
            n_true_frequencies,
            valid_phase_shift_range,
            first_pass_step_size,
            second_pass_best_n,
            second_pass_width,
            device,
            converge_epsilon=least_squares_converge_epsilon,
            kill_problems=kill_problems,
            amplitude_initialize_range=amplitude_initialize_range,
            l1_regularization_callable=l1_regularizer_callable,
            spatial_continuity_regularizer=spatial_regularizer_callable,
            max_batch_size=max_batch_size
        )

        amplitudes_cd_matrix[:, electrode_idx, :] = parallel_amplitudes_el
        shifts_cd_matrix[:, electrode_idx, :] = parallel_amplitudes_el

        pbar.update(1)
    pbar.close()

    return amplitudes_cd_matrix, shifts_cd_matrix


def shifted_fourier_nmf_iterative_optimization_spatial(waveforms_by_cell: np.ndarray,
                                                       last_valid_indices: np.ndarray,
                                                       initialized_canonical_waveforms: np.ndarray,
                                                       neighborhood_mean_mat: np.ndarray,
                                                       initial_amplitudes_by_cell: np.ndarray,
                                                       spatial_regularization_lambda: float,
                                                       valid_shift_range: Tuple[int, int],
                                                       shift_grid_step: int,
                                                       fine_search_top_n: int,
                                                       fine_search_width: int,
                                                       amplitude_init_range: Tuple[float, float],
                                                       n_iter: int,
                                                       device: torch.device,
                                                       l1_regularization_lambda: Optional[float] = None,
                                                       normalization_scale_factor: Optional[np.ndarray] = None,
                                                       waveform_observation_loss_weight: Optional[np.ndarray] = None,
                                                       sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    '''

    :param waveforms_by_cell: data matrix, shape (n_cells, max_n_electrodes, n_timepoints)
    :param last_valid_indices: shape (n_cells, ), integer-valued, contains the index of the last valid electrode
        for each cell in waveforms_by_cell
    :param initialized_canonical_waveforms: basis waveforms, shape (n_canonical_waveforms, n_timepoints)
    :param neighborhood_mean_mat: mean adjacency matrix for each cell, shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param initial_amplitudes_by_cell : intialized amplitudes, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes during gradient descent
    :param n_iter: int, number of iteration steps
    :param device: torch.device
    :param l1_regularization_lambda: float, L1 regularization penalty on the compartment weights
    :param normalization_scale_factor: Scale factor that was applied to the data waveforms,
        shape (n_cells, max_n_electrodes). Cannot contain any zeros for disused electrodes, so need to take care of
        this in the caller of this function
    :param waveform_observation_loss_weight: vecctor of weights, for weighting the contribution of each indivudal
        waveform to the overall loss when fitting basis waveform shapes. Shape (n_cells, max_n_electrodes)
    :param sobolev_regularization_lambda: scalar weight for second derivative penalty for regularizing the smoothness
        of the basis waveforms
    :return:
    '''

    n_cells, max_n_electrodes, n_timepoints = waveforms_by_cell.shape
    n_frequencies_not_rfft = n_timepoints

    # compute Fourier transform of the observed data once, ahead of time
    # shape (n_cells, max_n_electrodes, n_rfft_frequencies)
    data_ft = np.fft.rfft(waveforms_by_cell, axis=2)
    iter_canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

    # also want to keep a flattened representation of the data Fourier transforms
    # shape (n_observations, n_rfft_frequencies)
    flattened_data_ft = pack_by_cell_into_flat(data_ft, last_valid_indices)

    if waveform_observation_loss_weight is not None:
        # shape (n_cells, max_n_electrodes)
        flattened_waveform_weights = pack_by_cell_into_flat(waveform_observation_loss_weight,
                                                            last_valid_indices)
    else:
        flattened_waveform_weights = np.ones((flattened_data_ft.shape[0],), dtype=np.float32)

    print("Beginning optimization loop")
    pbar = tqdm.tqdm(total=n_iter, desc='Spatially regularized optimization')
    for iter_count in range(n_iter):
        # within each iteration, we have a two step optimization
        # (1) Given fixed canonical waveforms, solve for real-valued
        #   amplitudes and time shifts using coordinate descent
        # (2) Given fixed amplitudes and shifts, solve for the waveforms
        #   in frequency domain using unconstrained complex-valued linear least squares

        iter_real_amplitudes, iter_delays = search_with_coordinate_descent_projected(
            data_ft,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            last_valid_indices,
            neighborhood_mean_mat,
            initial_amplitudes_by_cell,
            normalization_scale_factor,
            spatial_regularization_lambda,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            amplitude_initialize_range=amplitude_init_range,
            l1_regularization_lambda=l1_regularization_lambda,
        )

        # iter_real_amplitudes and iter_delays have shape (n_cells, max_n_electrodes, n_canonical_waveforms)
        # now we have to reshape iter_real_amplitudes and iter_delays
        # so that we can reuse the Fourier code

        # shape (n_waveforms_total, n_canonical_waveforms)
        flattened_amplitude_mat = pack_by_cell_into_flat(iter_real_amplitudes, last_valid_indices)
        flattened_delays_mat = pack_by_cell_into_flat(iter_delays, last_valid_indices)

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        # print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms3(
            flattened_amplitude_mat,
            flattened_delays_mat,
            flattened_data_ft,
            n_frequencies_not_rfft,
            device,
            sobolev_lambda=sobolev_regularization_lambda,
            observation_loss_weight=flattened_waveform_weights
        )

        # shape (n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = np.fft.irfft(iter_canonical_waveform_ft, n=n_timepoints, axis=1)

        # now rescale the waveforms and amplitudes so that the basis waveforms each have L2 norm 1
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=1)
        iter_canonical_waveform_ft = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude[:, None]
        iter_canonical_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]
        flattened_amplitude_mat = flattened_amplitude_mat * raw_optimized_waveform_magnitude[None, :]

        mse = debug_evaluate_error(flattened_data_ft,
                                   flattened_amplitude_mat,
                                   iter_canonical_waveform_ft,
                                   flattened_delays_mat,
                                   n_frequencies_not_rfft)

        pbar.set_postfix({'MSE': mse})
        pbar.update(1)

    # now we want to unflatten the final result
    fitted_amplitudes_by_cell = unpack_flat_into_by_cell(flattened_amplitude_mat, last_valid_indices)
    return fitted_amplitudes_by_cell, iter_canonical_waveform_td, iter_delays, mse


def spatial_cont_time_optimization(eis_by_cell_id: Dict[int, np.ndarray],
                                   electrode_array_raw_adj_mat: np.ndarray,
                                   spatial_regularization_lambda: float,
                                   initial_decomposition: Dict[str, Any],
                                   device: torch.device,
                                   snr_abs_threshold: float = 5.0,
                                   amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
                                   supersample_factor: int = 5,
                                   shifts: Tuple[int, int] = (-100, 100),
                                   grid_search_step: int = 5,
                                   grid_search_top_n: int = 4,
                                   fine_search_width: int = 2,
                                   maxiter_spatial_reg_decomp: int = 10,
                                   l1_regularize_lambda: Optional[float] = None,
                                   sobolev_regularize_lambda: Optional[float] = None,
                                   output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray, float],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, float, Dict[str, np.ndarray]]]:
    '''
    Main optimization loop for the two-step optimization process, with spatial continuity regularization.
        Optimization steps are
        (0) Initialize with first pass solution using no-spatial two-step optimization process.

        for fixed number of iterations, iterate over
            (1) With fixed waveforms, solve for both the amplitudes and the timeshifts together by solving a series of
                nonnegative least squares L1 regularized problems, with spatial continuity penalty. We couple the solutions
                between the different electrodes of the same cell in a coordinate-descent manner (i.e. do the search for
                the first electrode, update the amplitudes for the purpose of calculating the penalty, then move on to
                the next electrode)
            (2) With fixed amplitudes and timeshifts, solve for the waveforms with complex-valued least squares
                with optional Sobolev second difference regularization


    :param padded_eis:
    :param electrode_array_raw_adj_mat:
    :param neighbors_mean_mat:
    :param device:
    :param n_basis_vectors:
    :param initialized_basis_vectors:
    :param snr_abs_threshold:
    :param amplitude_random_init_range:
    :param supersample_factor:
    :param shifts:
    :param grid_search_step:
    :param grid_search_top_n:
    :param fine_search_width:
    :param grid_search_batch_size:
    :param maxiter_decomp:
    :param renormalize_data_waveforms_amplitude_fit:
    :param renormalize_data_waveforms_waveform_fit:
    :param l1_regularize_lambda:
    :param sobolev_regularize_lambda:
    :param output_debug_dict:
    :return:
    '''

    n_electrodes_total = -1
    for cell_id, ei_matrix in eis_by_cell_id.items():
        n_electrodes_total = ei_matrix.shape[0]
        break

    temp_cell_order = list(eis_by_cell_id.keys())

    max_n_electrodes, selected_above_threshold_els = grab_above_threshold_electrodes_and_order(eis_by_cell_id,
                                                                                               snr_abs_threshold)

    ei_data_mat, matrix_indices_by_cell_id, last_valid_indices = make_electrode_padded_ei_data_matrix(
        eis_by_cell_id,
        temp_cell_order,
        max_n_electrodes,
        selected_above_threshold_els
    )

    prefit_decomp = initial_decomposition['decomposition']
    amplitudes_initialized_unflattened, phases = pack_full_by_cell_into_matrix_by_cell(prefit_decomp,
                                                                                       temp_cell_order,
                                                                                       max_n_electrodes,
                                                                                       selected_above_threshold_els)
    waveforms = initial_decomposition['waveforms']

    neighbor_mean_matrices_by_cell = make_spatial_neighbors_mean_matrix(electrode_array_raw_adj_mat,
                                                                        matrix_indices_by_cell_id,
                                                                        last_valid_indices)

    # shape (n_cells, max_n_electrodes, n_timepoints_unpadded)
    bspline_supersampled_by_cell = bspline_upsample_waveforms_padded_by_cell(ei_data_mat,
                                                                             last_valid_indices,
                                                                             supersample_factor)

    # now zero pad before and after
    # shape (n_cells, max_n_electrodes, n_timepoints)
    padded_channels_by_cell = np.pad(bspline_supersampled_by_cell,
                                     [(0, 0), (0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                     mode='constant')

    # shape (n_cells, max_n_electrodes), may contain zeros for disused electrodes
    # on a cell-by-cell basis, need to set those to one because we only use this
    # for multiplication and division
    mag_by_cell = np.linalg.norm(padded_channels_by_cell, axis=2)
    mag_by_cell_one_padded = one_pad_disused_by_cell(mag_by_cell, last_valid_indices)
    waveform_observation_weights_by_cell = (mag_by_cell_one_padded * mag_by_cell_one_padded)

    # shape (n_waveforms_total, n_timepoints)
    padded_channels_flattened = pack_by_cell_into_flat(padded_channels_by_cell,
                                                       last_valid_indices)
    # shape (n_waveforms_total, )
    mag_flattened = np.linalg.norm(padded_channels_flattened, axis=1)

    # shape (n_waveforms_total, n_timepoints)
    padded_channels_flattened = padded_channels_flattened / mag_flattened[:, None]

    # shape (n_waveforms_total, )
    waveform_observation_weights = 1.0 / (mag_flattened * mag_flattened)

    n_waveforms_total, _ = padded_channels_flattened.shape

    # amplitudes has shape (n_total_waveforms, n_basis_vectors)
    # waveforms has shape (n_basis_vectors, n_timepoints)
    # delays has shape (n_total_waveforms, n_basis_vectors) and is integer-valued
    # mse has shape (n_total_waveforms, )

    # shape (n_cells, max_n_electrodes, n_timepoints)
    normalized_unflattened_data_matrix = unpack_flat_into_by_cell(padded_channels_flattened,
                                                                  last_valid_indices)

    amplitudes_by_cell, canonical_waveforms, delays_by_cell, mse = shifted_fourier_nmf_iterative_optimization_spatial(
        normalized_unflattened_data_matrix,
        last_valid_indices,
        waveforms,
        neighbor_mean_matrices_by_cell,
        amplitudes_initialized_unflattened,
        spatial_regularization_lambda,
        shifts,
        grid_search_step,
        grid_search_top_n,
        fine_search_width,
        amplitude_random_init_range,
        maxiter_spatial_reg_decomp,
        device,
        l1_regularization_lambda=l1_regularize_lambda,
        normalization_scale_factor=mag_by_cell_one_padded,
        sobolev_regularization_lambda=sobolev_regularize_lambda,
        waveform_observation_loss_weight=waveform_observation_weights_by_cell
    )

    result_dict = pack_by_cell_amplitudes_and_phases_into_ei_shape(amplitudes_by_cell,
                                                                   delays_by_cell,
                                                                   matrix_indices_by_cell_id,
                                                                   last_valid_indices,
                                                                   temp_cell_order,
                                                                   n_electrodes_total)

    if output_debug_dict:
        debug_dict = {
            'amplitudes': amplitudes_by_cell,
            'waveforms': canonical_waveforms,
            'delays': delays_by_cell,
            'raw_data': padded_channels_by_cell
        }

        return result_dict, canonical_waveforms, mse, debug_dict

    return result_dict, canonical_waveforms, mse
