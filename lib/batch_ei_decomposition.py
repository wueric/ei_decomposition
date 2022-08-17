import numpy as np
import torch
from sklearn.gaussian_process.kernels import Kernel

from typing import List, Dict, Tuple, Optional, Union, Callable

import tqdm

from lib.util_fns import bspline_upsample_waveforms, auto_prebatch_pack_significant_electrodes, \
    auto_unbatch_unpack_significant_electrodes
from lib.batch_joint_amplitude_time_opt import batched_coarse_to_fine_time_shifts_and_amplitudes, \
    make_batched_group_l2_l1_weighted_regularizer, make_batched_component_l1_unweighted_regularizer, \
    make_batched_component_l1_weighted_regularizer, make_batched_group_l2_l1_unweighted_regularizer, \
    make_batched_by_cell_weighted_l1_regularizer, make_batched_unweighted_l1_regularizer
from lib.frequency_domain_optimization import batch_fourier_complex_least_square_optimize3, \
    construct_rfft_covariance_matrix, batch_fourier_complex_least_square_with_prior_optimize, _pack_complex_to_real_imag
from lib.losseval import batch_evaluate_mse_flat


def select_batched_l1_regularizer_callable(l1_regularization_lambda: Optional[float],
                                           n_basis_waveforms: int,
                                           use_scaled_regularization_terms: bool,
                                           per_problem_weights: Optional[np.ndarray],
                                           use_grouped_l1l2_norm: bool,
                                           grouped_l1l2_groups: Optional[List[np.ndarray]],
                                           use_basis_weighted_l1_norm: bool,
                                           basis_weights: Optional[np.ndarray],
                                           device: torch.device) \
        -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    '''
    Selects the regularization loss function

    :param l1_regularization_lambda:
    :param use_scaled_regularization_terms:
    :param use_grouped_l1l2_norm:
    :param grouped_l1l2_groups:
    :param use_basis_weighted_l1_norm:
    :param basis_weights:
    :return:
    '''

    # Check correctness of inputs
    if use_scaled_regularization_terms and per_problem_weights is None:
        raise ValueError("must specify problem weights if using scaled regularization terms")
    if use_grouped_l1l2_norm and grouped_l1l2_groups is None:
        raise ValueError("must specify L1L2 groups if using L1L2 regularization")
    if use_basis_weighted_l1_norm and basis_weights is None:
        raise ValueError("must specify basis weights if using component-wise weighted L1 regularization")

    if use_scaled_regularization_terms:

        if use_grouped_l1l2_norm:
            l1_regularization_callable = make_batched_group_l2_l1_unweighted_regularizer(l1_regularization_lambda,
                                                                                         n_basis_waveforms,
                                                                                         grouped_l1l2_groups,
                                                                                         device)
        elif use_basis_weighted_l1_norm:
            l1_regularization_callable = make_batched_component_l1_unweighted_regularizer(l1_regularization_lambda,
                                                                                          basis_weights,
                                                                                          device)
        else:
            l1_regularization_callable = make_batched_unweighted_l1_regularizer(l1_regularization_lambda)

    else:
        l1_reg_weight = 1.0 / per_problem_weights

        # dispatch table for setting up the correct regularization scheme
        if use_grouped_l1l2_norm:
            l1_regularization_callable = make_batched_group_l2_l1_weighted_regularizer(l1_reg_weight,
                                                                                       l1_regularization_lambda,
                                                                                       n_basis_waveforms,
                                                                                       grouped_l1l2_groups,
                                                                                       device)
        elif use_basis_weighted_l1_norm:
            l1_regularization_callable = make_batched_component_l1_weighted_regularizer(l1_reg_weight,
                                                                                        l1_regularization_lambda,
                                                                                        basis_weights,
                                                                                        device)
        else:
            l1_regularization_callable = make_batched_by_cell_weighted_l1_regularizer(l1_reg_weight,
                                                                                      l1_regularization_lambda,
                                                                                      device)

    return l1_regularization_callable


def batched_shifted_fourier_nmf_iterative_opt_with_prior(raw_waveform_data_matrix: np.ndarray,
                                                         is_valid_matrix: np.ndarray,
                                                         prior_basis_waveform_means: np.ndarray,
                                                         prior_basis_covariances: np.ndarray,
                                                         prior_regularization_lambda: float,
                                                         valid_shift_range: Tuple[int, int],
                                                         shift_grid_step: int,
                                                         fine_search_top_n: int,
                                                         fine_search_width: int,
                                                         amplitude_init_range: Tuple[float, float],
                                                         n_iter: int,
                                                         device: torch.device,
                                                         converge_epsilon: float = 1e-3,
                                                         converge_step_cutoff: Optional[float] = None,
                                                         max_batch_size=8192,
                                                         use_scaled_mse_penalty: bool = False,
                                                         use_scaled_regularization_terms: bool = False,
                                                         use_grouped_l1l2_norm: bool = False,
                                                         grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                                         use_basis_weighted_l1_norm: bool = False,
                                                         basis_weights_for_l1: Optional[np.ndarray] = None,
                                                         l1_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:

    '''
    Batched version of the main iteration loop for the two-step (as opposed to three-step) optimization process,
        with a Gaussian prior on the shape of the waveforms (used only during the waveform shape optimization step)
    Optimization steps are

        (1) With fixed waveforms, solve for both the amplitudes and the timeshifts together by performing grid search
            over a bunch of nonnegative least squares problems, with optional L1 regularization
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with complex-valued least squares with
            a Gaussian prior on the waveform shapes

    Because of the addition of the Gaussian prior, the linear system in the waveform shape optimization step is
        guaranteed to be full-rank (block diagonal) and therefore we no longer have problems if a given basis waveform
        is not observed ony channel.

    :param raw_waveform_data_matrix: np.ndarray, time domain data matrix, shape (batch, n_observations, n_timepoints)
        Not normalized. This function takes care of the normalization
    :param is_valid_matrix: np.ndarray, 0-1 integer valued. shape (batch, n_observations). Each entry is 1 if the
        corresponding data waveform in raw_waveform_data_matrix corresponds to real data, and is 0 if the corresponding
        data waveform is nonsense / padding
    :param prior_basis_waveform_means: np.ndarray, time domain canonical waveforms,
        shape (batch, n_basis, n_timepoints)
    :param prior_basis_covariances: np.ndarray, time-domain covariance matrix for the prior
        shape (batch, n_basis, n_timepoints, n_timepoints)
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes
    :param n_iter: number of iteration steps
    :param device:
    :param l1_regularization_lambda: weight for L1 regularization
    :return:
    '''

    # check input flag correctness
    if use_grouped_l1l2_norm and grouped_l1l2_groups is None:
        assert False, 'must specify groups if using L1L2 group norm regularization'

    is_valid_bool = is_valid_matrix.astype(bool)

    batch, n_observations, n_samples = raw_waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples
    n_basis = prior_basis_waveform_means.shape[1]

    # shape (batch, n_observations, 1)
    raw_data_magnitude = np.linalg.norm(raw_waveform_data_matrix, axis=2, keepdims=True)
    raw_data_magnitude[~is_valid_bool, :] = 1.0  # set null magnitudes to 1 to avoid dividing by zero
    # shape (batch, n_observations, n_samples)
    scaled_raw_data = raw_waveform_data_matrix / raw_data_magnitude

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (batch, n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(scaled_raw_data, axis=2)

    # shape (batch, n_basis_waveforms, n_frequencies)
    iter_canonical_waveform_ft = np.fft.rfft(prior_basis_waveform_means, axis=2)

    l1_regularization_callable = select_batched_l1_regularizer_callable(l1_regularization_lambda,
                                                                        n_basis,
                                                                        use_scaled_regularization_terms,
                                                                        raw_data_magnitude,
                                                                        use_grouped_l1l2_norm,
                                                                        grouped_l1l2_groups,
                                                                        use_basis_weighted_l1_norm,
                                                                        basis_weights_for_l1,
                                                                        device)

    waveform_observation_loss_weight = None
    if not use_scaled_mse_penalty:
        # shape (batch, n_observations)
        waveform_observation_loss_weight = raw_data_magnitude.copy().squeeze(2)

    # compute the prior covariance matrix in stacked real-imag Fourier domain
    prior_rfft_stacked_cov_mat = construct_rfft_covariance_matrix(prior_basis_covariances)

    # compute the rFFT of the prior mean, and then rearrange to put it in stacked real-imag form

    # shape (batch, n_basis_waveforms, n_frequencies)
    prior_mean_rfft = np.fft.rfft(prior_basis_waveform_means, axis=2)
    print(f"prior_mean_rfft {prior_mean_rfft.shape}")
    print(f"prior_basis_waveform_means {prior_basis_waveform_means.shape}")
    print(f"n_frequencies_not_rfft {n_frequencies_not_rfft}")
    # shape (batch, n_basis_waveforms, N)
    prior_mean_rfft_stacked = _pack_complex_to_real_imag(prior_mean_rfft,
                                                         n_frequencies_not_rfft,
                                                         axis=2)



    pbar = tqdm.tqdm(total=n_iter, desc='Overall optimization', leave=False)
    for iter_count in range(n_iter):
        # within each iteration, we have a two step optimization
        # (1) Given fixed canonical waveforms,
        #       solve for real-valued amplitudes and time shifts with
        #       grid search over nonnegative linear least squares problems
        # (2) Given fixed amplitudes and shifts, solve for waveforms in
        #       frequency domain with unconstrained complex-valued
        #       linear least squares with the Gaussian prior

        # both have shape (batch, n_observations, n_canonical_waveforms)
        # iter_real_amplitudes is real floating-point, iter_delays is integer
        iter_real_amplitudes, iter_delays = batched_coarse_to_fine_time_shifts_and_amplitudes(
            observations_fourier_transform,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            converge_epsilon=converge_epsilon,
            converge_step_cutoff=converge_step_cutoff,
            l1_regularization_callable=l1_regularization_callable,
            amplitude_initialize_range=amplitude_init_range,
            max_batch_size=max_batch_size
        )

        iter_canonical_waveform_ft = batch_fourier_complex_least_square_with_prior_optimize(
            iter_real_amplitudes,
            iter_delays,
            observations_fourier_transform,
            is_valid_matrix,
            prior_rfft_stacked_cov_mat,
            prior_mean_rfft_stacked,
            n_frequencies_not_rfft,
            prior_regularization_lambda,
            device,
            observation_loss_weight=waveform_observation_loss_weight
        )

        # shape (batch, n_canonical_waveforms, n_samples), real-valued float
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_samples, axis=2))

        # real valued np.ndarray, shape (batch, n_canonical_waveforms, 1)
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=2, keepdims=True)
        # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
        iter_canonical_waveform_ft = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude
        # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude

        # shape (batch, n_observations, n_basis_waveforms) * (batch, 1, n_basis_waveforms)
        # -> (batch, n_observations, n_basis_waveforms)
        iter_real_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude.transpose((0, 2, 1))

        orig_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_canonical_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=True,
                                           batch_observed_norms=waveform_observation_loss_weight,
                                           take_mean_over_electrodes=True)

        true_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_canonical_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=False,
                                           batch_observed_norms=raw_data_magnitude.squeeze(2))

        mse_component = batch_evaluate_mse_flat(observations_fourier_transform,
                                                iter_real_amplitudes,
                                                iter_canonical_waveform_ft,
                                                iter_delays,
                                                is_valid_matrix,
                                                n_frequencies_not_rfft,
                                                use_scaled_mse=use_scaled_mse_penalty,
                                                batch_observed_norms=waveform_observation_loss_weight)

        # FIXME we probably also want to report the overall loss value at some point
        # but that requires a bit of a reorg for the loss code
        # so I'll wait for that

        loss_dict = {
            'MSE equalized by electrode': np.mean(orig_MSE),
            'true MSE': np.mean(true_MSE),
            'Loss MSE component': np.mean(mse_component),
        }

        pbar.set_postfix(loss_dict)
        pbar.update(1)
    pbar.close()

    # shape (batch, n_observations, n_basis_waveforms) * (batch, n_observations, 1)
    # -> (batch, n_observations, n_basis_waveforms)
    fit_amplitudes_rescaled = iter_real_amplitudes * raw_data_magnitude
    return fit_amplitudes_rescaled, iter_canonical_waveform_td, iter_delays, loss_dict


def batch_two_step_decompose_cells_with_prior(
        eis_by_cell_id: Dict[int, np.ndarray],
        basis_prior_mean_waveforms: np.ndarray,
        basis_prior_covariance: Union[Kernel, List[Kernel]],
        basis_prior_reg_lambda: float,
        device: torch.device,
        converge_epsilon: float = 1e-3,
        converge_step_cutoff: Optional[float] = None,
        snr_abs_threshold: float = 5.0,
        amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
        shifts: Tuple[int, int] = (-100, 100),
        grid_search_step: int = 5,
        grid_search_top_n: int = 4,
        fine_search_width: int = 2,
        grid_search_batch_size: int = 1024,
        maxiter_decomp: int = 25,
        l1_regularize_lambda: Optional[float] = None,
        use_scaled_mse_penalty: bool = False,
        use_scaled_regularization_terms: bool = False,
        use_grouped_l1l2_norm: bool = False,
        grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
        use_basis_weighted_l1_norm: bool = False,
        basis_weights_for_l1: Optional[np.ndarray] = None) \
        -> Union[
            Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, float]],
            Dict[int, Dict[str, np.ndarray]]
        ]:
    '''
    Top-level wrapper for EI decomposition iterative algorithm including Gaussian prior on waveform shape, and
        either L1 or L2,1 sparsity/group-sparsity regularization on the amplitudes

    Unlike the previous implementation without the waveform prior, this function assumes that all of the inputs
        have been appropriately upsampled and padded (both data matrices and initial basis waveform means).

    Like the previous implementation, this function automatically takes care of batching. It also generates the
        basis waveform covariance matrix.

    :param eis_by_cell_id: Dict mapping cell id to raw EIs. Each EI must have shape (n_electrodes, n_timepoints_raw)
    :param basis_prior_mean_waveforms: shape (n_basis, n_timepoints_raw)
        means for the basis waveforms in time domain
    :param basis_prior_covariance: either Kernel (sklearn.gaussian_process.kernels.Kernel) or list of Kernel (one entry
        element for each basis waveform) for a Gaussian process
        distances are in units of raw data samples (before upsampling/interpolation)
    :param basis_prior_reg_lambda: float, regularization lambda strength for the Gaussian process
    :param device: torch.device
    :param converge_epsilon:
    :param converge_step_cutoff:
    :param snr_abs_threshold:
    :param amplitude_random_init_range:
    :param shifts:
    :param grid_search_step:
    :param grid_search_top_n:
    :param fine_search_width:
    :param grid_search_batch_size:
    :param maxiter_decomp:
    :param l1_regularize_lambda: float, regularization constant for L1 or group-sparse L1
    :param use_scaled_mse_penalty:
    :param use_scaled_regularization_terms:
    :param use_grouped_l1l2_norm:
    :param grouped_l1l2_groups:
    :param use_basis_weighted_l1_norm:
    :param basis_weights_for_l1:
    :return:
    '''

    n_basis, n_timepoints_raw = basis_prior_mean_waveforms.shape
    upsample_time_indices = np.r_[0:n_timepoints_raw:1.0]

    # generate the covariance matrices
    full_cov_mat = np.zeros((n_basis, n_timepoints_raw, n_timepoints_raw), dtype=np.float32)
    if isinstance(basis_prior_covariance, Kernel):
        kernel_cov_mat = basis_prior_covariance(upsample_time_indices)
        for i in range(n_basis):
            full_cov_mat[i, ...] = kernel_cov_mat
    elif isinstance(basis_prior_covariance, list) and len(basis_prior_covariance) == n_basis:
        for i, kernel_cov in enumerate(basis_prior_covariance):
            full_cov_mat[i, ...] = kernel_cov(upsample_time_indices)
    else:
        raise ValueError('if basis_prior_covariance is List, must have same length as number of basis waveforms')

    # do the autobatching
    autobatched_list = auto_prebatch_pack_significant_electrodes(eis_by_cell_id,
                                                                 snr_abs_threshold)

    wip_decomp_list = []
    batch_pbar = tqdm.tqdm(total=len(autobatched_list), desc='Batch')
    for batched_data_mat, is_valid_mat, ind_sel_mat, cell_order in autobatched_list:
        # Tensors have the following shapes
        # batched_data_mat: shape (batch, max_n_sig_electrodes, n_timepoints), real-valued float
        # is_valid_mat: shape (batch, max_n_sig_electrodes), boolean valued
        # ind_sel_mat: shape (batch, n_electrodes_total), integer indices
        # cell_order: shape (batch, ) integer cell id

        batch, max_n_sig_electrodes, n_timepoints_raw = batched_data_mat.shape

        # shape (n_basis_waveforms, n_timeponts_upsampled)
        # bspline_supersampled_basis = bspline_upsample_waveforms(initialized_basis_vectors, supersample_factor)
        # shape (n_basis_waveforms, n_timepoints)
        # padded_basis_waveforms_init = np.pad(bspline_supersampled_basis,
        #                                     [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
        #                                     mode='constant')
        # shape (batch, n_basis_waveforms, n_timepoints)
        batched_basis_waveforms = np.tile(basis_prior_mean_waveforms, (batch, 1, 1))

        # shape (batch, n_basis_waveforms, n_timepoints, n_timepoints)
        batched_prior_covs = np.tile(full_cov_mat, (batch, 1, 1, 1))

        amplitudes, waveforms, delays, mse = batched_shifted_fourier_nmf_iterative_opt_with_prior(
            batched_data_mat,
            is_valid_mat,
            batched_basis_waveforms,
            batched_prior_covs,
            basis_prior_reg_lambda,
            shifts,
            grid_search_step,
            grid_search_top_n,
            fine_search_width,
            amplitude_random_init_range,
            maxiter_decomp,
            device,
            converge_epsilon=converge_epsilon,
            converge_step_cutoff=converge_step_cutoff,
            max_batch_size=grid_search_batch_size,
            l1_regularization_lambda=l1_regularize_lambda,
            use_scaled_mse_penalty=use_scaled_mse_penalty,
            use_scaled_regularization_terms=use_scaled_regularization_terms,
            use_grouped_l1l2_norm=use_grouped_l1l2_norm,
            grouped_l1l2_groups=grouped_l1l2_groups,
            use_basis_weighted_l1_norm=use_basis_weighted_l1_norm,
            basis_weights_for_l1=basis_weights_for_l1,
        )

        wip_decomp_list.append((amplitudes, delays, waveforms))
        batch_pbar.update(1)

    batch_pbar.close()

    # now unpack the results
    result_dict = auto_unbatch_unpack_significant_electrodes(wip_decomp_list, autobatched_list)

    return result_dict


def batched_shifted_fourier_nmf_iterative_optimization3(raw_waveform_data_matrix: np.ndarray,
                                                        is_valid_matrix: np.ndarray,
                                                        initialized_canonical_waveforms: np.ndarray,
                                                        valid_shift_range: Tuple[int, int],
                                                        shift_grid_step: int,
                                                        fine_search_top_n: int,
                                                        fine_search_width: int,
                                                        amplitude_init_range: Tuple[float, float],
                                                        n_iter: int,
                                                        device: torch.device,
                                                        converge_epsilon: float = 1e-3,
                                                        converge_step_cutoff: Optional[float] = None,
                                                        max_batch_size=8192,
                                                        use_scaled_mse_penalty: bool = False,
                                                        use_scaled_regularization_terms: bool = False,
                                                        use_grouped_l1l2_norm: bool = False,
                                                        grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                                        use_basis_weighted_l1_norm: bool = False,
                                                        basis_weights_for_l1: Optional[np.ndarray] = None,
                                                        l1_regularization_lambda: Optional[float] = None,
                                                        sobolev_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    '''
    Batched version of the main iteration loop for the two-step (as opposed to three-step) optimization process.
    Optimization steps are

        (1) With fixed waveforms, solve for both the amplitudes and the timeshifts together by performing grid search
            over a bunch of nonnegative least squares problems, with optional L1 regularization
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with complex-valued least squares

    :param raw_waveform_data_matrix: np.ndarray, time domain data matrix, shape (batch, n_observations, n_timepoints)
        Not normalized. This function takes care of the normalization
    :param is_valid_matrix: np.ndarray, 0-1 integer valued. shape (batch, n_observations). Each entry is 1 if the
        corresponding data waveform in raw_waveform_data_matrix corresponds to real data, and is 0 if the corresponding
        data waveform is nonsense / padding
    :param initialized_canonical_waveforms: np.ndarray, time domain canonical waveforms,
        shape (batch, n_canonical_waveforms, n_timepoints)
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes
    :param n_iter: number of iteration steps
    :param device:
    :param l1_regularization_lambda: weight for L1 regularization
    :return:
    '''

    # check input flag correctness
    if use_grouped_l1l2_norm and grouped_l1l2_groups is None:
        assert False, 'must specify groups if using L1L2 group norm regularization'

    is_valid_bool = is_valid_matrix.astype(bool)

    batch, n_observations, n_samples = raw_waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples
    n_canonical_waveforms = initialized_canonical_waveforms.shape[1]

    # shape (batch, n_observations, 1)
    raw_data_magnitude = np.linalg.norm(raw_waveform_data_matrix, axis=2, keepdims=True)
    raw_data_magnitude[~is_valid_bool, :] = 1.0  # set null magnitudes to 1 to avoid dividing by zero
    # shape (batch, n_observations, n_samples)
    scaled_raw_data = raw_waveform_data_matrix / raw_data_magnitude

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (batch, n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(scaled_raw_data, axis=2)

    # shape (batch, n_basis_waveforms, n_frequencies)
    iter_canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=2)

    l1_regularization_callable = select_batched_l1_regularizer_callable(l1_regularization_lambda,
                                                                        n_canonical_waveforms,
                                                                        use_scaled_regularization_terms,
                                                                        raw_data_magnitude,
                                                                        use_grouped_l1l2_norm,
                                                                        grouped_l1l2_groups,
                                                                        use_basis_weighted_l1_norm,
                                                                        basis_weights_for_l1,
                                                                        device)

    waveform_observation_loss_weight = None
    if not use_scaled_mse_penalty:
        # shape (batch, n_observations)
        waveform_observation_loss_weight = raw_data_magnitude.copy().squeeze(2)

    pbar = tqdm.tqdm(total=n_iter, desc='Overall optimization', leave=False)
    for iter_count in range(n_iter):
        # within each iteration, we have a two step optimization
        # (1) Given fixed canonical waveforms,
        #       solve for real-valued amplitudes and time shifts with
        #       grid search over nonnegative linear least squares problems
        # (2) Given fixed amplitudes and shifts, solve for waveforms in
        #       frequency domain with unconstrained complex-valued
        #       linear least squares

        # both have shape (batch, n_observations, n_canonical_waveforms)
        # iter_real_amplitudes is real floating-point, iter_delays is integer
        iter_real_amplitudes, iter_delays = batched_coarse_to_fine_time_shifts_and_amplitudes(
            observations_fourier_transform,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            converge_epsilon=converge_epsilon,
            converge_step_cutoff=converge_step_cutoff,
            l1_regularization_callable=l1_regularization_callable,
            amplitude_initialize_range=amplitude_init_range,
            max_batch_size=max_batch_size
        )

        # complex valued, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
        iter_canonical_waveform_ft = batch_fourier_complex_least_square_optimize3(
            iter_real_amplitudes,
            iter_delays,
            observations_fourier_transform,
            is_valid_matrix,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            device,
            observation_loss_weight=waveform_observation_loss_weight,
            sobolev_lambda=sobolev_lambda
        )

        # shape (batch, n_canonical_waveforms, n_samples), real-valued float
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_samples, axis=2))

        # real valued np.ndarray, shape (batch, n_canonical_waveforms, 1)
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=2, keepdims=True)
        # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
        iter_canonical_waveform_ft = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude
        # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude

        # shape (batch, n_observations, n_basis_waveforms) * (batch, 1, n_basis_waveforms)
        # -> (batch, n_observations, n_basis_waveforms)
        iter_real_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude.transpose((0, 2, 1))

        orig_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_canonical_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=True,
                                           batch_observed_norms=waveform_observation_loss_weight,
                                           take_mean_over_electrodes=True)

        true_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_canonical_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=False,
                                           batch_observed_norms=raw_data_magnitude.squeeze(2))

        mse_component = batch_evaluate_mse_flat(observations_fourier_transform,
                                                iter_real_amplitudes,
                                                iter_canonical_waveform_ft,
                                                iter_delays,
                                                is_valid_matrix,
                                                n_frequencies_not_rfft,
                                                use_scaled_mse=use_scaled_mse_penalty,
                                                batch_observed_norms=waveform_observation_loss_weight)

        # FIXME we probably also want to report the overall loss value at some point
        # but that requires a bit of a reorg for the loss code
        # so I'll wait for that

        loss_dict = {
            'MSE equalized by electrode': np.mean(orig_MSE),
            'true MSE': np.mean(true_MSE),
            'Loss MSE component': np.mean(mse_component),
        }

        pbar.set_postfix(loss_dict)
        pbar.update(1)
    pbar.close()

    # shape (batch, n_observations, n_basis_waveforms) * (batch, n_observations, 1)
    # -> (batch, n_observations, n_basis_waveforms)
    fit_amplitudes_rescaled = iter_real_amplitudes * raw_data_magnitude
    return fit_amplitudes_rescaled, iter_canonical_waveform_td, iter_delays, loss_dict


def batch_two_step_decompose_cells_by_fitted_compartments(
        eis_by_cell_id: Dict[int, np.ndarray],
        initialized_basis_vectors: np.ndarray,
        device: torch.device,
        converge_epsilon: float = 1e-3,
        converge_step_cutoff: Optional[float] = None,
        snr_abs_threshold: float = 5.0,
        amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
        supersample_factor: int = 5,
        shifts: Tuple[int, int] = (-100, 100),
        grid_search_step: int = 5,
        grid_search_top_n: int = 4,
        fine_search_width: int = 2,
        grid_search_batch_size: int = 1024,
        maxiter_decomp: int = 25,
        l1_regularize_lambda: Optional[float] = None,
        use_scaled_mse_penalty: bool = False,
        use_scaled_regularization_terms: bool = False,
        use_grouped_l1l2_norm: bool = False,
        grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
        use_basis_weighted_l1_norm: bool = False,
        basis_weights_for_l1: Optional[np.ndarray] = None,
        sobolev_reg: Optional[float] = None) \
        -> Union[
            Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, float]],
            Dict[int, Dict[str, np.ndarray]]
        ]:
    '''

    :param eis_by_cell_id: Dict mapping cell id to raw EIs. Each EI must have shape (n_electrodes, n_timepoints)
    :param initialized_basis_vectors: Initial value for basis waveforms in time domain. It helps to specify reasonable
        basis waveforms, for accelerated convergence and better fits

        Even though each EI will be fitted with its own separate basis waveforms, we use the same initialization
            for each EI, since this initialization corresponds to a generic reasonable initial guess

        shape (n_basis_waveforms, n_timepoints_raw)
    :param device: torch device
    :param l1_regularize_lambda: float, regularization constant for L1 or group-sparse L1
    :param snr_abs_threshold:
    :param supersample_factor:
    :param shifts:
    :param maxiter_decomp:
    :return:
    '''

    autobatched_list = auto_prebatch_pack_significant_electrodes(eis_by_cell_id,
                                                                 snr_abs_threshold)

    wip_decomp_list = []
    batch_pbar = tqdm.tqdm(total=len(autobatched_list), desc='Batch')
    for batched_data_mat, is_valid_mat, ind_sel_mat, cell_order in autobatched_list:
        # Tensors have the following shapes
        # batched_data_mat: shape (batch, max_n_sig_electrodes, n_timepoints), real-valued float
        # is_valid_mat: shape (batch, max_n_sig_electrodes), boolean valued
        # ind_sel_mat: shape (batch, n_electrodes_total), integer indices
        # cell_order: shape (batch, ) integer cell id

        batch, max_n_sig_electrodes, n_timepoints_raw = batched_data_mat.shape
        flat_data_mat = batched_data_mat.reshape(batch * max_n_sig_electrodes, n_timepoints_raw)

        # shape (batch * max_n_sig_electrodes, n_timepoints_upsampled)
        flat_bspline_supersampled = bspline_upsample_waveforms(flat_data_mat, supersample_factor)
        _, n_timepoints_upsampled = flat_bspline_supersampled.shape

        # shape (batch, max_n_sig_electrodes, n_timepoints_upsampled)
        batched_bspline_supersampled = flat_bspline_supersampled.reshape(batch, max_n_sig_electrodes,
                                                                         n_timepoints_upsampled)

        # now zero pad before and after
        # shape (batch, max_n_sig_electrodes, n_timepoints)
        padded_channels_sufficient_magnitude = np.pad(batched_bspline_supersampled,
                                                      [(0, 0), (0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                                      mode='constant')

        # shape (n_basis_waveforms, n_timeponts_upsampled)
        # bspline_supersampled_basis = bspline_upsample_waveforms(initialized_basis_vectors, supersample_factor)
        # shape (n_basis_waveforms, n_timepoints)
        # padded_basis_waveforms_init = np.pad(bspline_supersampled_basis,
        #                                     [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
        #                                     mode='constant')
        # shape (batch, n_basis_waveforms, n_timepoints)
        batched_basis_waveforms = np.tile(initialized_basis_vectors, (batch, 1, 1))

        # amplitudes has shape (batch, n_observations, n_basis_waveforms))
        # waveforms has shape (batch, n_basis_waveforms, n_timepoints)
        amplitudes, waveforms, delays, mse = batched_shifted_fourier_nmf_iterative_optimization3(
            padded_channels_sufficient_magnitude,
            is_valid_mat,
            batched_basis_waveforms,
            shifts,
            grid_search_step,
            grid_search_top_n,
            fine_search_width,
            amplitude_random_init_range,
            maxiter_decomp,
            device,
            converge_epsilon=converge_epsilon,
            converge_step_cutoff=converge_step_cutoff,
            max_batch_size=grid_search_batch_size,
            l1_regularization_lambda=l1_regularize_lambda,
            use_scaled_mse_penalty=use_scaled_mse_penalty,
            use_scaled_regularization_terms=use_scaled_regularization_terms,
            use_grouped_l1l2_norm=use_grouped_l1l2_norm,
            grouped_l1l2_groups=grouped_l1l2_groups,
            use_basis_weighted_l1_norm=use_basis_weighted_l1_norm,
            basis_weights_for_l1=basis_weights_for_l1,
            sobolev_lambda=sobolev_reg
        )

        wip_decomp_list.append((amplitudes, delays, waveforms))
        batch_pbar.update(1)

    batch_pbar.close()

    # now unpack the results
    result_dict = auto_unbatch_unpack_significant_electrodes(wip_decomp_list, autobatched_list)

    return result_dict
