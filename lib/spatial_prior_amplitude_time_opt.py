import torch
import torch.nn as nn

import numpy as np

import tqdm

import electrode_map as el_map

from typing import Dict, Tuple, List, Optional, Union, Callable

from lib.util_fns import EIDecomposition, bspline_upsample_waveforms_padded_by_cell, \
    make_electrode_padded_ei_data_matrix, make_spatial_neighbors_mean_matrix, \
    bspline_upsample_waveforms, pack_by_cell_into_flat, unpack_flat_into_by_cell, \
    pack_by_cell_amplitudes_and_phases_into_ei_shape, one_pad_disused_by_cell
from lib.ei_decomposition import shifted_fourier_nmf_iterative_optimization3, debug_evaluate_error
from lib.joint_amplitude_time_optimization import coarse_to_fine_time_shifts_and_amplitudes
from lib.frequency_domain_optimization import fourier_complex_least_squares_optimize_waveforms3


def make_coord_descent_mean_connectivity_regularize_fn2(adjacency_mean_mat_by_cell: np.ndarray,
                                                        all_amplitude_matrix: np.ndarray,
                                                        lambda_spatial: float,
                                                        electrode_idx: int,
                                                        device: torch.device) \
        -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    '''
    Makes lambda functions to calculate both the value of the |AM-A|_F^2 spatial continuity penalty term
        as well as its gradient

    :param adjacency_mean_mat_by_cell: Adjacency mean matrix for each cell,
        shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param all_amplitude_matrix: previously fitted amplitudes for each cell, with no rescaling applied (i.e. these
        amplitudes should fit the raw EI). shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param lambda_spatial: scalar multiple lambda for the spatial continuity penalty
    :param electrode_idx: the index of the center electrode that we are working with
    :param device:
    :return: gradient callable, takes batched amplitudes of center electrode as input, and
        penalty term callable, takes batched amplitudes of center electrode as input
    '''

    n_cells, max_n_electrodes, _ = adjacency_mean_mat_by_cell.shape
    _, _, n_canonical_waveforms = all_amplitude_matrix.shape

    # shape (n_cells, n_canonical_waveforms, max_n_electrodes)
    amplitude_mat_torch = torch.tensor(all_amplitude_matrix.transpose((0, 2, 1)), dtype=torch.float32, device=device)
    # shape (n_cells, max_n_electrodes)
    adj_mean_torch = torch.tensor(adjacency_mean_mat_by_cell[:, :, electrode_idx], dtype=torch.float32, device=device)

    # shape (n_cells, n_canonical_waveforms)
    relevant_am_product = (amplitude_mat_torch @ adj_mean_torch[:, :, None]).squeeze(2)

    def gradient_callable(batched_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Gradient w.r.t. amplitudes of the spatial continuity penalty, with lambda included

        :param batched_amplitudes: shape (n_cells, ..., n_canonical_waveforms), with no normalization whatsoever (i.e.
            proportional to the original raw EI waveform)
        :return: gradient, shape (n_cells, ..., n_canonical_waveforms)
        '''
        ndim_batched_amplitudes = batched_amplitudes.ndim
        slice_obj = [slice(None), ]
        for i in range(ndim_batched_amplitudes - 2):
            slice_obj.append(None)
        slice_obj.append(slice(None))

        return lambda_spatial * (batched_amplitudes - relevant_am_product[slice_obj])

    def loss_callable(batched_amplitudes: torch.Tensor) -> torch.Tensor:
        '''
        Value of the spatial continuity penalty, with lambda included

        :param batched_amplitudes: shape (n_cells, ..., n_canonical_waveforms), with no normalization whatsoever (i.e.
            proportional to the original raw EI waveform)
        :return: loss value, shape (n_cells, ...)
        '''
        ndim_batched_amplitudes = batched_amplitudes.ndim
        slice_obj = [slice(None), ]
        for i in range(ndim_batched_amplitudes - 2):
            slice_obj.append(None)
        slice_obj.append(slice(None))

        # shape (n_cells, ..., n_canonical_waveforms)
        difference_mat = batched_amplitudes - relevant_am_product[slice_obj]

        # shape (n_cells, ...)
        total_loss = torch.sum(difference_mat * difference_mat, dim=-1) * lambda_spatial / 2
        return total_loss

    return gradient_callable, loss_callable


def search_with_coordinate_descent_projected(observed_ft_by_cell: np.ndarray,
                                             ft_canonical: np.ndarray,
                                             n_true_frequencies: int,
                                             neighborhood_mean_mat: np.ndarray,
                                             initial_amplitudes_by_cell: np.ndarray,
                                             spatial_regularization_lambda: float,
                                             valid_phase_shift_range: Tuple[int, int],
                                             first_pass_step_size: int,
                                             second_pass_best_n: int,
                                             second_pass_width: int,
                                             device: torch.device,
                                             l1_regularization_lambda: Optional[float] = None,
                                             normalization_scale_factor: Optional[np.ndarray] = None,
                                             amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                                             least_squares_converge_epsilon: float = 1e-3) \
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
    :param neighborhood_mean_mat: mean adjacency matrix for each cell, shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param initial_amplitudes_by_cell: initial amplitudes, shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    :param valid_phase_shift_range: range of sample shifts to consider
    :param first_pass_step_size: int, step size for the first pass grid search
    :param second_pass_best_n: int, algorithm expands upon the second_pass_best_n best results from the first pass
        grid search to do the second pass fine search
    :param second_pass_width: int, one-sided width around the second_pass_best_n best results from the first pass girid
        search that the algorithm searches in the second pass
    :param spatial_regularization_lambda: spatial regularization penalty weight
    :param device: torch.device
    :param l1_regularization_lambda: L1 basis waveform weight penalty
    :param normalization_scale_factor: Scale factor that was applied to the data waveforms,
        shape (n_cells, max_n_electrodes). Cannot contain any zeros for disused electrodes, so need to take care of
        this in the caller of this function
    :param least_squares_converge_epsilon:
    :return:
    '''

    n_cells, max_n_electrodes, n_rfft_frequencies = observed_ft_by_cell.shape
    n_canonical_waveforms, _ = ft_canonical.shape

    # shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    amplitudes_cd_matrix = initial_amplitudes_by_cell.copy()

    # shape (n_cells, max_n_electrodes, n_canonical_waveforms)
    shifts_cd_matrix = np.zeros((n_cells, max_n_electrodes, n_canonical_waveforms), dtype=np.int32)

    # solve the problems for each cell in parallel by electrode index
    for electrode_idx in range(max_n_electrodes):
        spatial_regularizer_callable = make_coord_descent_mean_connectivity_regularize_fn2(neighborhood_mean_mat,
                                                                                           amplitudes_cd_matrix,
                                                                                           spatial_regularization_lambda,
                                                                                           electrode_idx,
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
            l1_regularization_lambda=l1_regularization_lambda,
            normalization_scale_factor=normalization_scale_factor,
            converge_epsilon=least_squares_converge_epsilon,
            amplitude_initialize_range=amplitude_initialize_range,
            spatial_continuity_regularizer=spatial_regularizer_callable
        )

        amplitudes_cd_matrix[:, electrode_idx, :] = parallel_amplitudes_el
        shifts_cd_matrix[:, electrode_idx, :] = parallel_amplitudes_el

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
        flattened_waveform_weights = one_pad_disused_by_cell(waveform_observation_loss_weight,
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
            neighborhood_mean_mat,
            initial_amplitudes_by_cell,
            spatial_regularization_lambda,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            amplitude_initialize_range=amplitude_init_range,
            l1_regularization_lambda=l1_regularization_lambda,
            normalization_scale_factor=normalization_scale_factor,
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
                                   flattened_delays_mat,
                                   iter_delays,
                                   n_frequencies_not_rfft)

        pbar.set_postfix({'MSE': mse})
        pbar.update(1)

    # now we want to unflatten the final result
    fitted_amplitudes_by_cell = unpack_flat_into_by_cell(flattened_amplitude_mat, last_valid_indices)
    return fitted_amplitudes_by_cell, iter_canonical_waveform_td, iter_delays, mse


def spatial_cont_time_optimization(eis_by_cell_id: Dict[int, np.ndarray],
                                   electrode_array_raw_adj_mat: np.ndarray,
                                   spatial_regularization_lambda: float,
                                   device: torch.device,
                                   n_basis_vectors: Optional[int] = None,
                                   initialized_basis_vectors: Optional[np.ndarray] = None,
                                   snr_abs_threshold: float = 5.0,
                                   amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
                                   supersample_factor: int = 5,
                                   shifts: Tuple[int, int] = (-100, 100),
                                   grid_search_step: int = 5,
                                   grid_search_top_n: int = 4,
                                   fine_search_width: int = 2,
                                   grid_search_batch_size: int = 8192,
                                   maxiter_intialization_decomp: int = 2,
                                   maxiter_spatial_reg_decomp: int = 10,
                                   renormalize_data_waveforms_waveform_fit: bool = True,
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

    # check the inputs for correctness
    # must either specify the number of basis waveforms, or specify initial basis waveforms outright
    if n_basis_vectors is None and initialized_basis_vectors is None:
        raise ValueError('Must specify either n_basis_vectors or initialized_basis_vectors')
    elif n_basis_vectors is not None and initialized_basis_vectors is not None:
        raise ValueError('Can specify only one of n_basis_vectors and initialized_basis_vectors')

    n_electrodes_total = -1
    for cell_id, ei_matrix in eis_by_cell_id:
        n_electrodes_total = ei_matrix.shape[0]
        break

    temp_cell_order = list(eis_by_cell_id.keys())

    ei_data_mat, matrix_indices_by_cell_id, last_valid_indices = make_electrode_padded_ei_data_matrix(eis_by_cell_id,
                                                                                                      temp_cell_order,
                                                                                                      snr_abs_threshold)
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

    n_cells, max_n_electrodes, n_samples = padded_channels_by_cell.shape

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

    # init_basis will have shape (n_basis_vectors, n_timepoints)
    if n_basis_vectors is not None:
        # have to randomly initialize basis waveforms
        init_basis = np.zeros((n_basis_vectors, n_samples),
                              dtype=np.float32)
        rand_choice_data_waveform = np.random.randint(0, max_n_electrodes, size=n_basis_vectors)
        init_basis[:, :] = padded_channels_flattened[rand_choice_data_waveform, :]
        init_basis = init_basis / np.linalg.norm(init_basis, axis=1, keepdims=True)

    else:
        # also need to supersample and pad the initial basis waveforms
        bspline_supersampled_basis = bspline_upsample_waveforms(initialized_basis_vectors, supersample_factor)
        init_basis = np.pad(bspline_supersampled_basis,
                            [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                            mode='constant')
        init_basis = init_basis / np.linalg.norm(init_basis, axis=1, keepdims=True)

    # now, we have to make a first pass solution to initialize the amplitudes and waveforms in a reasonable place

    amplitudes, waveforms, delays, mse = shifted_fourier_nmf_iterative_optimization3(
        padded_channels_flattened,
        init_basis,
        shifts,
        grid_search_step,
        grid_search_top_n,
        fine_search_width,
        amplitude_random_init_range,
        maxiter_intialization_decomp,
        device,
        max_batch_size=grid_search_batch_size,
        l1_regularization_lambda=l1_regularize_lambda,
        sobolev_regularization_lambda=sobolev_regularize_lambda,
        waveform_observation_loss_weight=(
            None if renormalize_data_waveforms_waveform_fit else waveform_observation_weights)
    )

    # amplitudes has shape (n_total_waveforms, n_basis_vectors)
    # waveforms has shape (n_basis_vectors, n_timepoints)
    # delays has shape (n_total_waveforms, n_basis_vectors) and is integer-valued
    # mse has shape (n_total_waveforms, )

    # shape (n_cells, max_n_electrodes, n_timepoints)
    normalized_unflattened_data_matrix = unpack_flat_into_by_cell(padded_channels_flattened,
                                                                  last_valid_indices)
    amplitudes_initialized_unflattened = unpack_flat_into_by_cell(amplitudes,
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
