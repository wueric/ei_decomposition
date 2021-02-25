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
    grab_above_threshold_electrodes_and_order, pack_full_by_cell_into_matrix_by_cell, \
    get_neighborhood_indices_from_adj_mat_dfs

from lib.losseval import by_cell_evaluate_loss, evaluate_mse_by_cell
from lib.joint_amplitude_time_optimization import coarse_to_fine_time_shifts_and_amplitudes, \
    make_by_cell_weighted_l1_regularizer
from lib.frequency_domain_optimization import fourier_complex_least_squares_optimize_waveforms3


def make_sparse_coord_descent_mean_connectivity_regularize_fn(adjacency_mean_mat_by_cell: np.ndarray,
                                                              data_waveform_scaling: np.ndarray,
                                                              all_amplitude_matrix: np.ndarray,
                                                              lambda_spatial: float,
                                                              electrode_idx: int,
                                                              electrode_neighbor_indices_by_cell: np.array,
                                                              device: torch.device,
                                                              use_scaled_regularization: bool = False) \
        -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    '''
    Makes lambda functions to calculate both the value of |AM-A|_F^2 spatial continuity penalty term (but in terms
        of A' instead of A), as well as the gradient of the penalty term with respect to A', where A' is
        the normalized amplitude matrix rather than the amplitude matrix

    Idea here is that the adjacency matrix is sparse (each electrode has only a few nearest neighbors), and so we
        can perform most of the calculation with a small submatrix of the adjacency matrix and get the same result

    :param adjacency_mean_mat_by_cell: Adjacency mean matrix for each cell,
        shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param data_waveform_scaling: Scaling factor needed to apply to the data waveforms such that I recover the original
        raw EI waveform, arranged in by-cell format. For example, if we scale the input data waveforms such that the
        input to coordinate descent all has L2 norm 1, then then entries of data_waveform_scaling are the L2 norms
        of the original raw EI waveforms.
        Entries corresponding to disused electrodes have value 1
        Shape (n_cells, max_n_electrodes)
    :param all_amplitude_matrix: previously fitted normalized amplitudes A' for each cell (rescaled, these amplitudes
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
        if electrode_neighbor_indices_by_cell[cell_idx] is not None:
            max_n_submatrix_electrodes = max(len(electrode_neighbor_indices_by_cell[cell_idx]) + 1,
                                             max_n_submatrix_electrodes)

    # determine what the scalar multiples need to be ####################################################
    # we can precalculate these because they are the same for every interation ##########################
    gradient_unscaler_torch = None
    loss_unscaler_torch = None
    if not use_scaled_regularization:
        gradient_unscaler = 1.0 / np.power(data_waveform_scaling[:, electrode_idx], 2)
        loss_unscaler = 1.0 / np.power(data_waveform_scaling[:, electrode_idx], 2)

        gradient_unscaler_torch = torch.tensor(gradient_unscaler, device=device, dtype=torch.float32)
        loss_unscaler_torch = torch.tensor(loss_unscaler, device=device, dtype=torch.float32)

    ###### Calculate the magnitude outer product ########################################################
    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mag_outer_product = np.zeros((n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes), dtype=np.float32)

    ###### and pack the M-I submatrices #################################################################
    ###### also select the subset of electrode ampltiudes that matter for the calculation ###############
    diag_magnitude_for_neighborhood = np.zeros((n_cells, max_n_submatrix_electrodes),
                                               dtype=np.float32)

    # this is M
    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices = np.zeros((n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes),
                                dtype=np.float32)

    # this is M-I
    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices_minus_i = np.zeros((n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes),
                                        dtype=np.float32)

    # this is the relevant amplitude submatrix (i.e. the relevant piece of A')
    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_submatrices = np.zeros((n_cells, max_n_submatrix_electrodes, n_canonical_waveforms),
                                     dtype=np.float32)

    for cell_idx in range(n_cells):
        neighbor_indices = electrode_neighbor_indices_by_cell[cell_idx]
        mean_matrix_for_cell = adjacency_mean_mat_by_cell[cell_idx, :, :]

        if len(neighbor_indices) > 1:
            n_included_els = len(neighbor_indices)

            diag_magnitude_for_neighborhood[cell_idx, :n_included_els] = data_waveform_scaling[
                cell_idx, neighbor_indices]

            mean_submatrix = mean_matrix_for_cell[np.ix_(neighbor_indices, neighbor_indices)]
            mean_submatrices[cell_idx, :n_included_els, :n_included_els] = mean_submatrix

            mean_submatrix_minus_ident = mean_submatrix - np.eye(n_included_els, dtype=np.float32)
            mean_submatrices_minus_i[cell_idx, :n_included_els, :n_included_els] = mean_submatrix_minus_ident

            amplitude_submatrices[cell_idx, :n_included_els, :] = all_amplitude_matrix[cell_idx, neighbor_indices, :]

            outer_product = data_waveform_scaling[cell_idx, :n_included_els, None] @ data_waveform_scaling[
                                                                                     cell_idx, None, :n_included_els]
            mag_outer_product[cell_idx, :n_included_els, :n_included_els] = outer_product

    ###### Transfer stuff over to GPU and calculate shared quantities ##########################
    # shape (n_cells, max_n_submatrix_electrodes)
    diag_magnitude_for_nbd_torch = torch.tensor(diag_magnitude_for_neighborhood,
                                                dtype=torch.float32,
                                                device=device)

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mag_outer_product_torch = torch.tensor(mag_outer_product, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    mean_submatrices_minus_i_torch = torch.tensor(mean_submatrices_minus_i, dtype=torch.float32, device=device)

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    m_i_t_m_i = mean_submatrices_minus_i_torch.permute(0, 2, 1) @ mean_submatrices_minus_i_torch

    # shape (n_cells, max_n_submatrix_electrodes, max_n_submatrix_electrodes)
    outer_product_m_i_prod = mag_outer_product_torch @ m_i_t_m_i

    # shape (n_cells, max_n_submatrix_electrodes)
    outer_product_m_i_prod_relevant = outer_product_m_i_prod[:, :, 0]

    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_mat_torch = torch.tensor(amplitude_submatrices.transpose((0, 2, 1)), dtype=torch.float32, device=device)

    # shape (n_cells, n_canonical_waveforms, max_n_submatrix_electrodes)
    amplitude_mat_torch_without_electrode = amplitude_mat_torch.clone().detach()
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
        batched_amplitude_mat[:, :, :, 0] = batched_normalized_electrode_amplitudes

        # shape (n_cells, batch_size, n_canonical_waveforms)
        all_grad = (batched_amplitude_mat @ outer_product_m_i_prod_relevant[:, None, :, None]).squeeze(3)

        if not use_scaled_regularization:
            # shape (n_cells, batch_size, n_canonical_waveforms)
            return all_grad * lambda_spatial * gradient_unscaler_torch[:, None, None]
        else:
            return all_grad * lambda_spatial

    def loss_callable(batched_normalized_electrode_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Value of the spatial continuity penalty, with lambda included

        Note that the implementation of this needs to careful to avoid including the unused electrodes at the
            end of the data

        :param batched_normalized_amplitudes: shape (n_cells, batch_size, n_basis_waveforms), with normalization
            This is A' for the center electrode, which if used directly will not fit the raw EIs
        :return: loss value, shape (n_cells, ...)
        '''
        _, batch_size, _ = batched_normalized_electrode_amplitudes.shape

        # shape (n_cells, batch_size, n_basis_waveforms)
        center_aprime_diag = diag_magnitude_for_nbd_torch[:, 0, None, None] * batched_normalized_electrode_amplitudes

        # shape (n_cells, batch_size, n_basis_waveforms, max_submatrix_electrodes)
        unshared_center_mean = center_aprime_diag[:, :, :, None] @ mean_submatrices_torch[:, None, 0, None, :]

        # shape (n_cells, n_basis_waveforms, max_n_submatrix_electrodes - 1)
        nbd_aprime_diag = amplitude_mat_torch_without_electrode[:, :, 1:] * diag_magnitude_for_nbd_torch[:, None, 1:]

        # shape (n_cells, n_basis_waveforms, max_n_submatrix_electrodes - 1) @
        #       shape (n_cells, max_n_submatrix_electrodes - 1, max_n_submatrix_electrodes)
        # which has shape (n_cells, n_basis_waveforms, max_n_submatrix_electrodes)
        shared_neighborhood_mean = nbd_aprime_diag @ mean_submatrices_torch[:, 1:, :]

        # shape (n_cells, batch_size, n_basis_waveforms, max_n_submatrix_electrodes)
        aprime_diag_m_product = unshared_center_mean + shared_neighborhood_mean[:, None, :, :]

        # shape (n_cells, batch_size, n_basis_waveforms, max_n_submatrix_electrodes)
        aprime_diag_m_minus_aprime = aprime_diag_m_product.clone().detach()
        aprime_diag_m_minus_aprime[:, :, :, 0] -= center_aprime_diag
        aprime_diag_m_minus_aprime[:, :, :, 1:] -= nbd_aprime_diag[:, None, :, :]

        frob_norm = torch.sum(aprime_diag_m_minus_aprime * aprime_diag_m_minus_aprime, dim=(2, 3))

        if not use_scaled_regularization:
            return lambda_spatial * frob_norm * loss_unscaler_torch[:, None] / 2.0
        else:
            return lambda_spatial * frob_norm / 2.0

    return gradient_callable, loss_callable


def search_with_coordinate_descent_projected(observed_ft_by_cell: np.ndarray,
                                             observed_data_scaling: np.ndarray,
                                             ft_canonical: np.ndarray,
                                             n_true_frequencies: int,
                                             last_valid_indices: np.ndarray,
                                             neighborhood_mean_mat: np.ndarray,
                                             prev_iter_amplitudes_by_cell: np.ndarray,
                                             spatial_regularization_lambda: float,
                                             valid_phase_shift_range: Tuple[int, int],
                                             first_pass_step_size: int,
                                             second_pass_best_n: int,
                                             second_pass_width: int,
                                             device: torch.device,
                                             use_scaled_regularization_terms: bool = False,
                                             l1_regularization_lambda: Optional[float] = None,
                                             amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                                             least_squares_converge_epsilon: float = 1e-3,
                                             max_batch_size: Optional[int] = 4096) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Does one pass of coordinate descent on the amplitudes and time shifts, parallelizing by cell
        The core algorithm is grid search over a series of nonnegative least squares problems with
        regularization, solved using projected gradient descent

    Optimizes the objective function |B^{\tau_e} A_{:,e}' - X_{:,e}'|_2^2 with optional regularization

        Note that A' is the scaled amplitude defined as A \diag |{ 1 / |X_{:,}|_2 \},
        and X' is the scaled data waveform defined as X \diag \{ 1 / |X_{:, }|_2 \}

    :param observed_ft_by_cell: Fourier transform of the scaled data X', complex valued,
        shape (n_cells, max_n_electrodes, n_rfft_frequencies). May contain zero-valued vectors.
    :param observed_data_scaling: L2 norm of the original X waveforms (unscaled)
        shape (n_cells, max_n_electrodes), where entries corresponding to unused electrodes
        have value 1.0 so we can perform division
    :param ft_canonical: basis waveforms in Fourier domain, unshifted, complex valued,
        shape (n_canonical_waveforms, n_rfft_frequencies)
    :param n_true_frequencies: number of actual (not rFFT) FFT frequencies
    :param last_valid_indices: shape (n_cells, ), integer-valued, number of valid electrodes for each cell in
        observed_ft_by_cell
    :param neighborhood_mean_mat: mean adjacency matrix for each cell, shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param prev_iter_amplitudes_by_cell: initial amplitudes, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param valid_phase_shift_range: range of sample shifts to consider
    :param first_pass_step_size: int, step size for the first pass grid search
    :param second_pass_best_n: int, algorithm expands upon the second_pass_best_n best results from the first pass
        grid search to do the second pass fine search
    :param second_pass_width: int, one-sided width around the second_pass_best_n best results from the first pass girid
        search that the algorithm searches in the second pass
    :param spatial_regularization_lambda: spatial regularization penalty weight
    :param device: torch.device
    :param l1_regularization_lambda: L1 basis waveform weight penalty
    :param mse_scale_factor: Scale factor that we want to apply to the MSE fit term to control the relative
        weight between the MSE fit problem and the regularization terms
        Shape (n_cells, max_n_electrodes), entries corresponding to unused electrodes must be 1 to facilitate
        easy multiplication and division
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

        kill_problems = (electrode_idx >= last_valid_indices)  # shape (n_cells, )

        electrode_nn_mat = get_neighborhood_indices_from_adj_mat_dfs(neighborhood_mean_mat, electrode_idx, 2)
        spatial_regularizer_callable = make_sparse_coord_descent_mean_connectivity_regularize_fn(neighborhood_mean_mat,
                                                                                                 observed_data_scaling,
                                                                                                 amplitudes_cd_matrix,
                                                                                                 spatial_regularization_lambda,
                                                                                                 electrode_idx,
                                                                                                 electrode_nn_mat,
                                                                                                 device)

        l1_regularizer_callable = None
        if l1_regularization_lambda is not None:
            if use_scaled_regularization_terms:
                l1_scale_factor = 1.0 / observed_data_scaling[:, electrode_idx]
                l1_regularizer_callable = make_by_cell_weighted_l1_regularizer(l1_scale_factor,
                                                                               l1_regularization_lambda,
                                                                               device)
            else:
                l1_scale_factor = np.ones((observed_data_scaling.shape[0],), dtype=np.float32)
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
            # spatial_continuity_regularizer=spatial_regularizer_callable,
            max_batch_size=max_batch_size
        )

        amplitudes_cd_matrix[:, electrode_idx, :] = parallel_amplitudes_el
        shifts_cd_matrix[:, electrode_idx, :] = parallel_shifts_el

        pbar.update(1)
    pbar.close()

    return amplitudes_cd_matrix, shifts_cd_matrix


def shifted_fourier_nmf_iterative_optimization_spatial(raw_waveforms_by_cell: np.ndarray,
                                                       last_valid_indices: np.ndarray,
                                                       initialized_canonical_waveforms: np.ndarray,
                                                       neighborhood_mean_mat: np.ndarray,
                                                       prev_iter_amplitudes_by_cell: np.ndarray,
                                                       prev_iter_delays_by_cell: np.ndarray,
                                                       spatial_regularization_lambda: float,
                                                       valid_shift_range: Tuple[int, int],
                                                       shift_grid_step: int,
                                                       fine_search_top_n: int,
                                                       fine_search_width: int,
                                                       amplitude_init_range: Tuple[float, float],
                                                       n_iter: int,
                                                       device: torch.device,
                                                       l1_regularization_lambda: Optional[float] = None,
                                                       use_scaled_mse_penalty: bool = False,
                                                       use_scaled_regularization_terms: bool = False,
                                                       sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    '''
    High level wrapper to perform iterative coordinate descent with spatial penalty

    :param raw_waveforms_by_cell: data matrix, shape (n_cells, max_n_electrodes, n_timepoints)
    :param last_valid_indices: shape (n_cells, ), integer-valued, contains the number of valid electrodes for each cell
        in raw_waveforms_by_cell
    :param initialized_canonical_waveforms: basis waveforms in time domain, shape (n_canonical_waveforms, n_timepoints)
        Should have L2 norm 1
    :param neighborhood_mean_mat: mean adjacency matrix for each cell, shape (n_cells, max_n_electrodes, max_n_electrodes)
        This matrix computes the spatial mean of the nearest neighbors of an electrode
    :param prev_iter_amplitudes_by_cell : intialized amplitudes, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
            Amplitudes from a previous fit. These amplitudes should fit raw_waveforms_by_cell without any
            additional scaling
    :param prev_iter_delays_by_cell: sample delays for each cell/electrode, integer valued
        shape (n_cells, max_n_electrodes, n_canonical_waveforms)
        Calculated form a previous fit.
    :param spatial_regularization_lambda: lambda scalar multiple for the spatial regularization term
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes during gradient descent
    :param n_iter: int, number of iteration steps
    :param device: torch.device
    :param l1_regularization_lambda: float, L1 regularization penalty on the compartment weights
    :param use_scaled_mse_penalty: Whether or not to scale the MSE loss by the L2 norm of the data waveforms
        when performing the waveform Fourier domain optimization. True if we want the MSE loss to be about
        the same for every electrode regardless of L2 norm, False if we want the large amplitude electrodes
        to contribute more heavily to the MSE loss
    :param use_scaled_regularization_terms: Whether or not to scale the regularization terms according to the L2 norm
        of the data waveforms. If True, we will calculate the penalty terms using the rescaled basis amplitudes. If
        False, we will calculate the penalty terms according to the unscaled basis amplitudes
    :param normalization_scale_factor: Scale factor that was applied to the data waveforms,
        shape (n_cells, max_n_electrodes). Cannot contain any zeros for disused electrodes, so need to take care of
        this in the caller of this function
    :param waveform_observation_loss_weight: vector of weights, for weighting the contribution of each indivudal
        waveform to the overall loss when fitting basis waveform shapes. Shape (n_cells, max_n_electrodes)
    :param sobolev_regularization_lambda: scalar weight for second derivative penalty for regularizing the smoothness
        of the basis waveforms
    :return:
    '''

    n_cells, max_n_electrodes, n_timepoints = raw_waveforms_by_cell.shape
    n_frequencies_not_rfft = n_timepoints

    raw_waveform_norms = np.linalg.norm(raw_waveforms_by_cell, axis=2)
    raw_waveform_norms_one_padded = one_pad_disused_by_cell(raw_waveform_norms,
                                                            last_valid_indices)
    normalized_raw_waveforms_by_cell = raw_waveforms_by_cell / raw_waveform_norms_one_padded[:, :, None]

    normalized_prev_iter_amplitudes = prev_iter_amplitudes_by_cell / raw_waveform_norms_one_padded[:, :, None]

    # compute Fourier transform of the observed data once, ahead of time
    # shape (n_cells, max_n_electrodes, n_rfft_frequencies)
    data_ft = np.fft.rfft(normalized_raw_waveforms_by_cell, axis=2)
    iter_canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

    # also want to keep a flattened representation of the data Fourier transforms
    # shape (n_observations, n_rfft_frequencies)
    flattened_data_ft = pack_by_cell_into_flat(data_ft, last_valid_indices)

    flattened_waveform_weights = None
    if not use_scaled_mse_penalty:
        # shape (n_observations, )
        flattened_waveform_weights = pack_by_cell_into_flat(raw_waveform_norms,
                                                            last_valid_indices)
    if not use_scaled_regularization_terms:
        # shape (n_cells, max_n_electrodes)
        waveform_weights_one_padded = raw_waveform_norms_one_padded
    else:
        waveform_weights_one_padded = np.ones((n_cells, max_n_electrodes), dtype=np.float32)

    before_mse = by_cell_evaluate_loss(data_ft,
                                       normalized_prev_iter_amplitudes,
                                       last_valid_indices,
                                       waveform_weights_one_padded,
                                       iter_canonical_waveform_ft,
                                       prev_iter_delays_by_cell,
                                       n_frequencies_not_rfft,
                                       neighborhood_mean_matrices=neighborhood_mean_mat,
                                       lambda_l1=l1_regularization_lambda,
                                       lambda_spatial=spatial_regularization_lambda,
                                       use_scaled_mse=use_scaled_mse_penalty,
                                       use_scaled_reg_penalty=use_scaled_regularization_terms)

    print("Before MSE: {0}".format(before_mse))

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
            waveform_weights_one_padded,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            last_valid_indices,
            neighborhood_mean_mat,
            normalized_prev_iter_amplitudes,
            spatial_regularization_lambda,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            use_scaled_regularization_terms=use_scaled_regularization_terms,
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
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_frequencies_not_rfft, axis=1))

        # now rescale the waveforms and amplitudes so that the basis waveforms each have L2 norm 1
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=1)
        iter_canonical_waveform_ft = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude[:, None]
        iter_canonical_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]
        iter_real_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude[None, None, :]

        orig_MSE = evaluate_mse_by_cell(data_ft,
                                        iter_real_amplitudes,
                                        last_valid_indices,
                                        iter_canonical_waveform_ft,
                                        iter_delays,
                                        n_frequencies_not_rfft,
                                        norm_scale_factor_by_cell=raw_waveform_norms_one_padded,
                                        use_scaled_mse=True,
                                        take_mean_over_valid_electrodes=True)

        true_MSE = evaluate_mse_by_cell(data_ft,
                                        iter_real_amplitudes,
                                        last_valid_indices,
                                        iter_canonical_waveform_ft,
                                        iter_delays,
                                        n_frequencies_not_rfft,
                                        norm_scale_factor_by_cell=raw_waveform_norms_one_padded,
                                        use_scaled_mse=False,
                                        take_mean_over_valid_electrodes=False)

        mse_component = evaluate_mse_by_cell(data_ft,
                                             iter_real_amplitudes,
                                             last_valid_indices,
                                             iter_canonical_waveform_ft,
                                             iter_delays,
                                             n_frequencies_not_rfft,
                                             norm_scale_factor_by_cell=raw_waveform_norms_one_padded,
                                             use_scaled_mse=use_scaled_mse_penalty,
                                             take_mean_over_valid_electrodes=False)

        loss_with_penalty = by_cell_evaluate_loss(data_ft,
                                                  iter_real_amplitudes,
                                                  last_valid_indices,
                                                  waveform_weights_one_padded,
                                                  iter_canonical_waveform_ft,
                                                  iter_delays,
                                                  n_frequencies_not_rfft,
                                                  neighborhood_mean_matrices=neighborhood_mean_mat,
                                                  lambda_l1=l1_regularization_lambda,
                                                  lambda_spatial=spatial_regularization_lambda,
                                                  use_scaled_mse=use_scaled_mse_penalty,
                                                  use_scaled_reg_penalty=use_scaled_regularization_terms)

        loss_dict = {
            'MSE equalized by electrode' : orig_MSE,
            'true MSE': true_MSE,
            'Loss MSE component' : mse_component,
            'Loss': loss_with_penalty

        }

        pbar.set_postfix(loss_dict)
        pbar.update(1)

    # now we want to unflatten the final result
    iter_real_amplitudes_orig_scaling = iter_real_amplitudes * raw_waveform_norms[:, :, None]
    return iter_real_amplitudes_orig_scaling, iter_canonical_waveform_td, iter_delays, loss_dict


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
                                   use_scaled_mse_penalty: bool = False,
                                   use_scaled_regularization_terms: bool = False,
                                   output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float]],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float], Dict[str, np.ndarray]]]:
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

    Definitions:
        * The "Scaled MSE term" is defined as \frac{1}{|X_{:,e}|_2^2} |B^{(\tau_e} A_{:,e} - X_{:,e}|_2^2
            which is identical in value to |B^{(\tau_e} A_{:,e}' - X_{:,e}'|_2^2
        * The "Scaled regularization terms" refer to
            1. The scaled L1 regularization terms, \lambda_{L1} \sum_{e \in E} \frac{1]{|X_:,e|_2} 1^T A_{:,e}'
            2. The scaled spatial continuity penalty

    There are four distinct objective functions that we can solve for here, depending on if
        use_scaled_mse_term is True and if use_scaled_regularization_terms is True
        (1) The "unnormalized" objective function, which uses the unscaled MSE Term and the unscaled regularization
            terms
        (2) The "normalized by electrode" objective function, which uses the scaled MSE term and the scaled
            regularization terms. This one has a weaker L1 penalty for large-amplitude electrodes relative to the
            (1) "unnormalized" objection function, since the A' amplitudes used for in the normalized calculation will
            be substantially smaller than the original A
        (3) The "mixed case" objective function, which uses the scaled MSE term and the unscaled regularization terms.
            This one will tend to over-regularize in L1, since the L1 penalty for large amplitude electrodes is unscaled
            while the MSE fit penalty will be scaled down
        (4) The "weird case" objective function, which uses the unscaled MSE term and the scaled regularization terms.
            This one will tend to under-regularize in L1, since the L1 penalty for the large amplitude electrodes is
            scaled down while the MSE fit penalty is not scaled at all

    :param eis_by_cell_id: cell_id (int) -> EI (np.ndarray). Each np.ndarray has shape (n_electrodes, n_timepoints)
        These EIs are unpadded and at the original sample resolution. This function will automatically take care
        of the upsampling and left/right time padding
    :param electrode_array_raw_adj_mat: raw adjacency list representation of nearest neighbors for the electrode array
        Has n_electrode rows, the i^{th} row contains a List[int] corresponding to the neighbors of electrode i
        Everything is 0-indexed
    :param spatial_regularization_lambda: regularization weight for the spatial regularization term
    :param device: torch.device
    :param snr_abs_threshold: float, maximum deviation from 0 on channel must exceed this value in order for the
        channel to be included in the calculation
    :param amplitude_random_init_range: Parameters of uniform distribution used to initialize the amplitudes for
        the proximal gradient step. Will affect convergence speed, but should not affect final result
    :param supersample_factor: int, factor at which we supersample in time domain
    :param shifts: Tuple[int, int], range of shifts that the algorithm will consider
    :param grid_search_step:
    :param grid_search_top_n:
    :param fine_search_width:
    :param grid_search_batch_size:
    :param maxiter_spatial_reg_decomp:
    :param l1_regularize_lambda:
    :param sobolev_regularize_lambda:
    :param use_scaled_regularization_terms: whether or not to use the scaled regularization terms in the loss function.
        If False, use the unscaled regularization terms
    :param output_debug_dict:
    :return:
    '''

    # determine how many electrodes there are in a complete EI
    n_electrodes_total = -1
    for cell_id, ei_matrix in eis_by_cell_id.items():
        n_electrodes_total = ei_matrix.shape[0]
        break

    # temporary ordering of cell_id, used to determine order of the rows of all tensors
    temp_cell_order = list(eis_by_cell_id.keys())

    max_n_electrodes, selected_above_threshold_els = grab_above_threshold_electrodes_and_order(eis_by_cell_id,
                                                                                               snr_abs_threshold)
    # ei_data_mat has shape (n_cells, max_n_electrodes, n_raw_timepoints)
    # last_valid_indices has shape (n_cells, )
    ei_data_mat, matrix_indices_by_cell_id, last_valid_indices = make_electrode_padded_ei_data_matrix(
        eis_by_cell_id,
        temp_cell_order,
        max_n_electrodes,
        selected_above_threshold_els
    )

    prefit_decomp = initial_decomposition['decomposition']
    amplitudes_initialized_unflattened, phases_unflattened = pack_full_by_cell_into_matrix_by_cell(prefit_decomp,
                                                                                                   temp_cell_order,
                                                                                                   max_n_electrodes,
                                                                                                   selected_above_threshold_els)
    basis_waveforms = initial_decomposition['waveforms']

    # shape (n_cells, max_n_electrodes, max_n_electrodes)
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

    # amplitudes has shape (n_total_waveforms, n_basis_vectors)
    # waveforms has shape (n_basis_vectors, n_timepoints)
    # delays has shape (n_total_waveforms, n_basis_vectors) and is integer-valued
    # mse has shape (n_total_waveforms, )
    amplitudes_by_cell, canonical_waveforms, delays_by_cell, mse = shifted_fourier_nmf_iterative_optimization_spatial(
        padded_channels_by_cell,
        last_valid_indices,
        basis_waveforms,
        neighbor_mean_matrices_by_cell,
        amplitudes_initialized_unflattened,
        phases_unflattened,
        spatial_regularization_lambda,
        shifts,
        grid_search_step,
        grid_search_top_n,
        fine_search_width,
        amplitude_random_init_range,
        maxiter_spatial_reg_decomp,
        device,
        l1_regularization_lambda=l1_regularize_lambda,
        use_scaled_mse_penalty=use_scaled_mse_penalty,
        use_scaled_regularization_terms=use_scaled_regularization_terms,
        sobolev_regularization_lambda=sobolev_regularize_lambda,
    )

    result_dict = pack_by_cell_amplitudes_and_phases_into_ei_shape(amplitudes_by_cell,
                                                                   delays_by_cell,
                                                                   selected_above_threshold_els,
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
