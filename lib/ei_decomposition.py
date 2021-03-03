import numpy as np
import torch

from typing import List, Dict, Tuple, Sequence, Optional, Union, Callable

import tqdm

from lib.amplitude_optimization import nonnegative_least_squares_optimize_amplitudes
from lib.frequency_domain_optimization import fourier_complex_least_squares_optimize_waveforms3
from lib.joint_amplitude_time_optimization import coarse_to_fine_time_shifts_and_amplitudes, \
    make_unweighted_l1_regularizer, make_by_cell_weighted_l1_regularizer, make_group_l2_l1_unweighted_regularizer, \
    make_group_l2_l1_weighted_regularizer, make_component_l1_unweighted_regularizer, make_component_l1_weighted_regularizer
from lib.template_matching import greedy_template_match_time_shift, torch_fit_integer_shifts_all_but_one_template_match
from lib.util_fns import bspline_upsample_waveforms, generate_fourier_phase_shift_matrices, \
    EIDecomposition, pack_significant_electrodes_into_matrix, unpack_amplitudes_and_phases_into_ei_shape
from lib.losseval import evaluate_mse_flat, flat_pack_evaluate_loss


def shifted_fourier_nmf_iterative_optimization2(waveform_data_matrix: np.ndarray,
                                                initialized_canonical_waveforms: np.ndarray,
                                                initialized_amplitudes: np.ndarray,
                                                intialized_delays: np.ndarray,
                                                valid_sample_shifts: np.ndarray,
                                                n_iter: int,
                                                device: torch.device,
                                                l1_regularization_lambda: Optional[float] = None,
                                                sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
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
        canonical_waveforms_shifted = np.real(np.fft.irfft(canonical_waveform_shift_ft, n=n_samples, axis=2))

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

        mse = evaluate_mse_flat(observations_fourier_transform,
                                iter_real_amplitudes,
                                iter_canonical_waveform_ft,
                                intialized_delays,
                                n_frequencies_not_rfft,
                                use_scaled_mse=False)

        loss_dict = {"MSE": mse}

        pbar.set_postfix(loss_dict)
        pbar.update(1)

    return initialized_amplitudes, initialized_canonical_waveforms, intialized_delays, loss_dict


def select_l1_regularizer_callable(l1_regularization_lambda: Optional[float],
                                   n_basis_waveforms : int,
                                   use_scaled_regularization_terms: bool,
                                   per_problem_weights: Optional[np.ndarray],
                                   use_grouped_l1l2_norm: bool,
                                   grouped_l1l2_groups: Optional[List[np.ndarray]],
                                   use_basis_weighted_l1_norm: bool,
                                   basis_weights: Optional[np.ndarray],
                                   device : torch.device) \
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
        assert False, "must specify problem weights if using scaled regularization terms"
    if use_grouped_l1l2_norm and grouped_l1l2_groups is None:
        assert False, "must specify L1L2 groups if using L1L2 regularization"
    if use_basis_weighted_l1_norm and basis_weights is None:
        assert False, "must specify basis weights if using component-wise weighted L1 regularization"

    if use_scaled_regularization_terms:

        if use_grouped_l1l2_norm:
            l1_regularization_callable = make_group_l2_l1_unweighted_regularizer(l1_regularization_lambda,
                                                                                 n_basis_waveforms,
                                                                                 grouped_l1l2_groups,
                                                                                 device)
        elif use_basis_weighted_l1_norm:
            l1_regularization_callable = make_component_l1_unweighted_regularizer(l1_regularization_lambda,
                                                                                  basis_weights,
                                                                                  device)
        else:
            l1_regularization_callable = make_unweighted_l1_regularizer(l1_regularization_lambda)

    else:
        l1_reg_weight = 1.0 / per_problem_weights

        # dispatch table for setting up the correct regularization scheme
        if use_grouped_l1l2_norm:
            l1_regularization_callable = make_group_l2_l1_weighted_regularizer(l1_reg_weight,
                                                                               l1_regularization_lambda,
                                                                               n_basis_waveforms,
                                                                               grouped_l1l2_groups,
                                                                               device)
        elif use_basis_weighted_l1_norm:
            l1_regularization_callable = make_component_l1_weighted_regularizer(l1_reg_weight,
                                                                                l1_regularization_lambda,
                                                                                basis_weights,
                                                                                device)
        else:
            l1_regularization_callable = make_by_cell_weighted_l1_regularizer(l1_reg_weight,
                                                                              l1_regularization_lambda,
                                                                              device)

    return l1_regularization_callable


def one_iteration_amplitude_fit_only(raw_waveform_data_matrix: np.ndarray,
                                     basis_waveforms: np.ndarray,
                                     valid_shift_range: Tuple[int, int],
                                     shift_grid_step: int,
                                     fine_search_top_n: int,
                                     fine_search_width: int,
                                     amplitude_init_range: Tuple[float, float],
                                     device: torch.device,
                                     max_batch_size=8192,
                                     use_scaled_mse_penalty: bool = False,
                                     use_scaled_regularization_terms: bool = False,
                                     use_grouped_l1l2_norm: bool = False,
                                     grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                     use_basis_weighted_l1_norm: bool = False,
                                     basis_weights_for_l1: Optional[np.ndarray] = None,
                                     l1_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    n_observations, n_samples = raw_waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples

    n_basis_waveforms = basis_waveforms.shape[0]

    raw_data_magnitude = np.linalg.norm(raw_waveform_data_matrix, axis=1)
    scaled_raw_data = raw_waveform_data_matrix / raw_data_magnitude[:, None]

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(scaled_raw_data, axis=1)
    iter_canonical_waveform_ft = np.fft.rfft(basis_waveforms, axis=1)

    l1_regularization_callable = None
    if l1_regularization_lambda is not None:
        l1_regularization_callable = select_l1_regularizer_callable(l1_regularization_lambda,
                                                                    n_basis_waveforms,
                                                                    use_scaled_regularization_terms,
                                                                    raw_data_magnitude,
                                                                    use_grouped_l1l2_norm,
                                                                    grouped_l1l2_groups,
                                                                    use_basis_weighted_l1_norm,
                                                                    basis_weights_for_l1,
                                                                    device)

    iter_real_amplitudes, iter_delays = coarse_to_fine_time_shifts_and_amplitudes(
        observations_fourier_transform,
        iter_canonical_waveform_ft,
        n_frequencies_not_rfft,
        valid_shift_range,
        shift_grid_step,
        fine_search_top_n,
        fine_search_width,
        device,
        l1_regularization_callable=l1_regularization_callable,
        amplitude_initialize_range=amplitude_init_range,
        max_batch_size=max_batch_size
    )

    orig_MSE = evaluate_mse_flat(observations_fourier_transform,
                                 iter_real_amplitudes,
                                 iter_canonical_waveform_ft,
                                 iter_delays,
                                 n_frequencies_not_rfft,
                                 use_scaled_mse=True,
                                 observed_norms=raw_data_magnitude,
                                 take_mean_over_electrodes=True)

    true_MSE = evaluate_mse_flat(observations_fourier_transform,
                                 iter_real_amplitudes,
                                 iter_canonical_waveform_ft,
                                 iter_delays,
                                 n_frequencies_not_rfft,
                                 use_scaled_mse=False,
                                 observed_norms=raw_data_magnitude)

    mse_component = evaluate_mse_flat(observations_fourier_transform,
                                      iter_real_amplitudes,
                                      iter_canonical_waveform_ft,
                                      iter_delays,
                                      n_frequencies_not_rfft,
                                      use_scaled_mse=use_scaled_mse_penalty,
                                      observed_norms=raw_data_magnitude)

    loss_with_penalty = flat_pack_evaluate_loss(observations_fourier_transform,
                                                iter_real_amplitudes,
                                                raw_data_magnitude,
                                                iter_canonical_waveform_ft,
                                                iter_delays,
                                                n_frequencies_not_rfft,
                                                l1_regularization_lambda,
                                                use_scaled_reg_penalty=use_scaled_regularization_terms,
                                                use_scaled_mse=use_scaled_mse_penalty)

    loss_dict = {
        'MSE equalized by electrode': orig_MSE,
        'true MSE': true_MSE,
        'Loss MSE component': mse_component,
        'Loss': loss_with_penalty

    }

    fit_amplitudes_rescaled = iter_real_amplitudes * raw_data_magnitude[:, None]

    return fit_amplitudes_rescaled, basis_waveforms, iter_delays, loss_dict


def shifted_fourier_nmf_iterative_optimization3(raw_waveform_data_matrix: np.ndarray,
                                                initialized_canonical_waveforms: np.ndarray,
                                                valid_shift_range: Tuple[int, int],
                                                shift_grid_step: int,
                                                fine_search_top_n: int,
                                                fine_search_width: int,
                                                amplitude_init_range: Tuple[float, float],
                                                n_iter: int,
                                                device: torch.device,
                                                max_batch_size=8192,
                                                use_scaled_mse_penalty: bool = False,
                                                use_scaled_regularization_terms: bool = False,
                                                use_grouped_l1l2_norm: bool = False,
                                                grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                                use_basis_weighted_l1_norm: bool = False,
                                                basis_weights_for_l1: Optional[np.ndarray] = None,
                                                l1_regularization_lambda: Optional[float] = None,
                                                sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    '''
    Main iteration loop for the two-step (as opposed to three-step) optimization process. Optimization steps are

        (1) With fixed waveforms, solve for both the amplitudes and the timeshifts together by performing grid search
            over a bunch of nonnegative least squares problems, with optional L1 regularization
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with complex-valued least squares
            with optional Sobolev second difference regularization

    :param raw_waveform_data_matrix: np.ndarray, time domain data matrix, shape (n_observations, n_timepoints)
        Not normalized. This function takes care of the normalization
    :param initialized_canonical_waveforms: np.ndarray, time domain canonical waveforms,
        shape (n_canonical_waveforms, n_timepoints)
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes
    :param n_iter: number of iteration steps
    :param device:
    :param l1_regularization_lambda: weight for L1 regularization
    :param sobolev_regularization_lambda:
    :return:
    '''

    # check input flag correctness
    if use_grouped_l1l2_norm and grouped_l1l2_groups is None:
        assert False, 'must specify groups if using L1L2 group norm regularization'

    n_observations, n_samples = raw_waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples
    n_canonical_waveforms = initialized_canonical_waveforms.shape[0]

    raw_data_magnitude = np.linalg.norm(raw_waveform_data_matrix, axis=1)
    scaled_raw_data = raw_waveform_data_matrix / raw_data_magnitude[:, None]

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(scaled_raw_data, axis=1)
    iter_canonical_waveform_ft = np.fft.rfft(initialized_canonical_waveforms, axis=1)

    l1_regularization_callable = select_l1_regularizer_callable(l1_regularization_lambda,
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
        waveform_observation_loss_weight = raw_data_magnitude.copy()

    print("Beginning optimization loop")
    pbar = tqdm.tqdm(total=n_iter, desc='Overall optimization')
    for iter_count in range(n_iter):
        # within each iteration, we have a two step optimization
        # (1) Given fixed canonical waveforms,
        #       solve for real-valued amplitudes and time shifts with
        #       grid search over nonnegative linear least squares problems
        # (2) Given fixed amplitudes and shifts, solve for waveforms in
        #       frequency domain with unconstrained complex-valued
        #       linear least squares

        iter_real_amplitudes, iter_delays = coarse_to_fine_time_shifts_and_amplitudes(
            observations_fourier_transform,
            iter_canonical_waveform_ft,
            n_frequencies_not_rfft,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            device,
            l1_regularization_callable=l1_regularization_callable,
            amplitude_initialize_range=amplitude_init_range,
            max_batch_size=max_batch_size
        )

        # complex valued np.ndarray, shape (n_canonical_waveforms, n_frequencies)
        # print("Iter {0}, Waveform complex least squares".format(iter_count))
        iter_canonical_waveform_ft = fourier_complex_least_squares_optimize_waveforms3(
            iter_real_amplitudes,
            iter_delays,
            observations_fourier_transform,
            n_frequencies_not_rfft,
            device,
            sobolev_lambda=sobolev_regularization_lambda,
            observation_loss_weight=waveform_observation_loss_weight
        )

        # real valued np.ndarray, shape (n_canonical_waveforms, n_samples)
        iter_canonical_waveform_td = np.real(np.fft.irfft(iter_canonical_waveform_ft, n=n_samples, axis=1))

        # real valued np.ndarray, shape (n_canonical_waveforms, )
        raw_optimized_waveform_magnitude = np.linalg.norm(iter_canonical_waveform_td, axis=1)
        iter_canonical_waveform_ft = iter_canonical_waveform_ft / raw_optimized_waveform_magnitude[:, None]
        iter_canonical_waveform_td = iter_canonical_waveform_td / raw_optimized_waveform_magnitude[:, None]
        iter_real_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude[None, :]

        orig_MSE = evaluate_mse_flat(observations_fourier_transform,
                                     iter_real_amplitudes,
                                     iter_canonical_waveform_ft,
                                     iter_delays,
                                     n_frequencies_not_rfft,
                                     use_scaled_mse=True,
                                     observed_norms=raw_data_magnitude,
                                     take_mean_over_electrodes=True)

        true_MSE = evaluate_mse_flat(observations_fourier_transform,
                                     iter_real_amplitudes,
                                     iter_canonical_waveform_ft,
                                     iter_delays,
                                     n_frequencies_not_rfft,
                                     use_scaled_mse=False,
                                     observed_norms=raw_data_magnitude)

        mse_component = evaluate_mse_flat(observations_fourier_transform,
                                          iter_real_amplitudes,
                                          iter_canonical_waveform_ft,
                                          iter_delays,
                                          n_frequencies_not_rfft,
                                          use_scaled_mse=use_scaled_mse_penalty,
                                          observed_norms=raw_data_magnitude)

        loss_with_penalty = flat_pack_evaluate_loss(observations_fourier_transform,
                                                    iter_real_amplitudes,
                                                    raw_data_magnitude,
                                                    iter_canonical_waveform_ft,
                                                    iter_delays,
                                                    n_frequencies_not_rfft,
                                                    l1_regularization_lambda,
                                                    use_scaled_reg_penalty=use_scaled_regularization_terms,
                                                    use_scaled_mse=use_scaled_mse_penalty)

        loss_dict = {
            'MSE equalized by electrode': orig_MSE,
            'true MSE': true_MSE,
            'Loss MSE component': mse_component,
            'Loss': loss_with_penalty

        }

        pbar.set_postfix(loss_dict)
        pbar.update(1)

    fit_amplitudes_rescaled = iter_real_amplitudes * raw_data_magnitude[:, None]

    return fit_amplitudes_rescaled, iter_canonical_waveform_td, iter_delays, loss_dict


def shifted_fourier_nmf(waveform_data_matrix: np.ndarray,
                        n_canonical_waveforms: int,
                        valid_sample_shifts: np.ndarray,
                        n_iter: int,
                        device: torch.device,
                        l1_regularization_lambda: Optional[float] = None,
                        sobolev_regularization_lambda: Optional[float] = None,
                        amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0)) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
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


def optimize_initialized_waveforms_fourier_nmf(waveform_data_matrix: np.ndarray,
                                               initialized_canonical_waveforms: np.ndarray,
                                               valid_sample_shifts: np.ndarray,
                                               n_iter: int,
                                               device: torch.device,
                                               l1_regularization_lambda: Optional[float] = None,
                                               sobolev_regularization_lambda: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
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


def two_step_decompose_cells_by_fitted_compartments(eis_by_cell_id: Dict[int, np.ndarray],
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
                                                    maxiter_decomp: int = 25,
                                                    l1_regularize_lambda: Optional[float] = None,
                                                    sobolev_regularize_lambda: Optional[float] = None,
                                                    use_scaled_mse_penalty: bool = False,
                                                    use_scaled_regularization_terms: bool = False,
                                                    use_grouped_l1l2_norm: bool = False,
                                                    grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                                    use_basis_weighted_l1_norm: bool = False,
                                                    basis_weights_for_l1: Optional[np.ndarray] = None,
                                                    output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float]],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float], Dict[str, np.ndarray]]]:
    # check the inputs for correctness
    # must either specify the number of basis waveforms, or specify initial basis waveforms outright
    if n_basis_vectors is None and initialized_basis_vectors is None:
        raise ValueError('Must specify either n_basis_vectors or initialized_basis_vectors')
    elif n_basis_vectors is not None and initialized_basis_vectors is not None:
        raise ValueError('Can specify only one of n_basis_vectors and initialized_basis_vectors')

    temp_cell_order = list(eis_by_cell_id.keys())

    ei_data_mat, matrix_indices_by_cell_id = pack_significant_electrodes_into_matrix(eis_by_cell_id,
                                                                                     temp_cell_order,
                                                                                     snr_abs_threshold)

    bspline_supersampled = bspline_upsample_waveforms(ei_data_mat, supersample_factor)

    # now zero pad before and after
    # shpae (n_observations, n_samples)
    padded_channels_sufficient_magnitude = np.pad(bspline_supersampled,
                                                  [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                                  mode='constant')

    n_observations, n_samples = padded_channels_sufficient_magnitude.shape

    if n_basis_vectors is not None:
        # have to randomly initialize basis waveforms
        init_basis = np.zeros((n_basis_vectors, n_samples),
                              dtype=np.float32)
        rand_choice_data_waveform = np.random.randint(0, n_observations, size=n_basis_vectors)
        init_basis[:, :] = padded_channels_sufficient_magnitude[rand_choice_data_waveform, :]
        init_basis = init_basis / np.linalg.norm(init_basis, axis=1, keepdims=True)

    else:

        # also need to supersample and pad the initial basis waveforms
        bspline_supersampled_basis = bspline_upsample_waveforms(initialized_basis_vectors, supersample_factor)
        init_basis = np.pad(bspline_supersampled_basis,
                            [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                            mode='constant')
        init_basis = init_basis / np.linalg.norm(init_basis, axis=1, keepdims=True)

    amplitudes, waveforms, delays, mse = shifted_fourier_nmf_iterative_optimization3(
        padded_channels_sufficient_magnitude,
        init_basis,
        shifts,
        grid_search_step,
        grid_search_top_n,
        fine_search_width,
        amplitude_random_init_range,
        maxiter_decomp,
        device,
        max_batch_size=grid_search_batch_size,
        l1_regularization_lambda=l1_regularize_lambda,
        sobolev_regularization_lambda=sobolev_regularize_lambda,
        use_scaled_mse_penalty=use_scaled_mse_penalty,
        use_scaled_regularization_terms=use_scaled_regularization_terms,
        use_grouped_l1l2_norm=use_grouped_l1l2_norm,
        grouped_l1l2_groups=grouped_l1l2_groups,
        use_basis_weighted_l1_norm=use_basis_weighted_l1_norm,
        basis_weights_for_l1=basis_weights_for_l1
    )

    # now unpack the results
    result_dict = unpack_amplitudes_and_phases_into_ei_shape(amplitudes,
                                                             delays,
                                                             eis_by_cell_id,
                                                             temp_cell_order,
                                                             matrix_indices_by_cell_id)

    if output_debug_dict:
        debug_dict = {
            'amplitudes': amplitudes,
            'waveforms': waveforms,
            'delays': delays,
            'raw_data': padded_channels_sufficient_magnitude
        }

        return result_dict, waveforms, mse, debug_dict

    return result_dict, waveforms, mse


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

    temp_cell_order = list(eis_by_cell_id.keys())

    ei_data_mat, matrix_indices_by_cell_id = pack_significant_electrodes_into_matrix(eis_by_cell_id,
                                                                                     temp_cell_order,
                                                                                     snr_abs_threshold)

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

    if renormalize_data_waveforms:
        amplitudes = mag_padded[:, None] * amplitudes

    # now unpack the results
    result_dict = unpack_amplitudes_and_phases_into_ei_shape(amplitudes,
                                                             delays,
                                                             eis_by_cell_id,
                                                             temp_cell_order,
                                                             matrix_indices_by_cell_id)

    if output_debug_dict:
        debug_dict = {
            'amplitudes': amplitudes,
            'waveforms': waveforms,
            'delays': delays,
            'raw_data': padded_channels_sufficient_magnitude
        }

        return result_dict, waveforms, mse, debug_dict

    return result_dict, waveforms, mse


def decompose_cells_amplitudes_only(eis_by_cell_id: Dict[int, np.ndarray],
                                    device: torch.device,
                                    basis_vectors: np.ndarray,
                                    snr_abs_threshold: float = 5.0,
                                    amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
                                    supersample_factor: int = 5,
                                    shifts: Tuple[int, int] = (-100, 100),
                                    grid_search_step: int = 5,
                                    grid_search_top_n: int = 4,
                                    fine_search_width: int = 2,
                                    grid_search_batch_size: int = 8192,
                                    l1_regularize_lambda: Optional[float] = None,
                                    use_scaled_mse_penalty: bool = False,
                                    use_scaled_regularization_terms: bool = False,
                                    use_grouped_l1l2_norm: bool = False,
                                    grouped_l1l2_groups: Optional[List[np.ndarray]] = None,
                                    use_basis_weighted_l1_norm: bool = False,
                                    basis_weights_for_l1: Optional[np.ndarray] = None,
                                    output_debug_dict: bool = False) \
        -> Union[Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float]],
                 Tuple[Dict[int, EIDecomposition], np.ndarray, Dict[str, float], Dict[str, np.ndarray]]]:
    temp_cell_order = list(eis_by_cell_id.keys())

    ei_data_mat, matrix_indices_by_cell_id = pack_significant_electrodes_into_matrix(eis_by_cell_id,
                                                                                     temp_cell_order,
                                                                                     snr_abs_threshold)

    bspline_supersampled = bspline_upsample_waveforms(ei_data_mat, supersample_factor)

    # now zero pad before and after
    # shpae (n_observations, n_samples)
    padded_channels_sufficient_magnitude = np.pad(bspline_supersampled,
                                                  [(0, 0), (abs(shifts[0]), abs(shifts[1]))],
                                                  mode='constant')

    amplitudes, waveforms, delays, mse = one_iteration_amplitude_fit_only(
        padded_channels_sufficient_magnitude,
        basis_vectors,
        shifts,
        grid_search_step,
        grid_search_top_n,
        fine_search_width,
        amplitude_random_init_range,
        device,
        max_batch_size=grid_search_batch_size,
        l1_regularization_lambda=l1_regularize_lambda,
        use_scaled_mse_penalty=use_scaled_mse_penalty,
        use_scaled_regularization_terms=use_scaled_regularization_terms,
        use_grouped_l1l2_norm=use_grouped_l1l2_norm,
        grouped_l1l2_groups=grouped_l1l2_groups,
        use_basis_weighted_l1_norm=use_basis_weighted_l1_norm,
        basis_weights_for_l1=basis_weights_for_l1
    )

    # now unpack the results
    result_dict = unpack_amplitudes_and_phases_into_ei_shape(amplitudes,
                                                             delays,
                                                             eis_by_cell_id,
                                                             temp_cell_order,
                                                             matrix_indices_by_cell_id)

    if output_debug_dict:
        debug_dict = {
            'amplitudes': amplitudes,
            'waveforms': waveforms,
            'delays': delays,
            'raw_data': padded_channels_sufficient_magnitude
        }

        return result_dict, waveforms, mse, debug_dict

    return result_dict, waveforms, mse
