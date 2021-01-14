from typing import Optional, Tuple

import numpy as np
import torch
import tqdm


def build_at_a_matrix(ft_canonical: np.ndarray,
                      valid_phase_shifts: np.ndarray,
                      n_true_frequencies: int) -> np.ndarray:
    '''

    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param valid_phase_shifts: integer array, shape (n_canonical_waveforms, n_valid_phase_shifts)
    :param n_true_frequencies: int
    :return: batched A^T A matrix, shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    '''

    circular_corr_td = np.fft.irfft(ft_canonical[:, None, :] * np.conjugate(ft_canonical[None, :, :]),
                                    n=n_true_frequencies,
                                    axis=2)

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_timepoints), axis 1 corresponds to the shifted waveforms
    # relative to axis 0 fixed waveforms
    # (i,j,t)^{th} entry corresponds to cross correlation of i^{th} canonical waveform with j^{th} canonical waveform
    #   that has been delayed by t samples
    # This means that circular_conv_td is not symmetric for dims (0, 1)

    # now we have to build at_a matrix by grabbing the relevant pieces
    # not so straightforward, since we care about relative timing instead of absolute timing

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_valid_phase_shifts)
    relative_shifts = valid_phase_shifts[None, :, :] - valid_phase_shifts[:, None, :]

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_valid_phase_shifts)
    taken_piece = np.take_along_axis(circular_corr_td, relative_shifts, axis=2)
    at_a_matrix_np = taken_piece.transpose((2, 0, 1))

    return at_a_matrix_np


def build_at_b_vector(observed_ft: np.ndarray,
                      ft_canonical: np.ndarray,
                      valid_phase_shifts: np.ndarray,
                      n_true_frequencies: int) -> np.ndarray:
    '''

    :param observed_ft:
    :param ft_canonical:
    :param valid_phase_shifts:
    :param n_true_frequencies:
    :return:
    '''
    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    data_circ_conv_td = np.fft.irfft(observed_ft[:, None, :] * np.conjugate(ft_canonical[None, :, :]),
                                     n=n_true_frequencies,
                                     axis=2)
    # The (i,j,t)^{th} entry corresponds to cross correlation of the i^{th} data waveform with the j^{th} canonical
    #   waveform that has been delayed by t samples

    # we have to build A^T b from this matrix
    # shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    at_b_perm = np.take_along_axis(data_circ_conv_td, valid_phase_shifts[None, :, :], axis=2)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_np = at_b_perm.transpose((0, 2, 1))

    return at_b_np


def build_unshared_at_a_matrix(ft_canonical: np.ndarray,
                               unshared_phase_shifts: np.ndarray,
                               n_true_frequencies: int) -> np.ndarray:
    '''

    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer, shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    :param n_true_frequencies: integer, number of FFT frequencies
    :return: np.ndarray, shape (n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
        last two dimensions are A^T A matrices
    '''

    n_observations, n_canonical_waveforms, n_phase_shifts = unshared_phase_shifts.shape

    # shape (n_canonical_waveforms, n_canonical_waveforms, n_timepoints)
    circular_corr_td = np.fft.irfft(ft_canonical[:, None, :] * np.conjugate(ft_canonical[None, :, :]),
                                    n=n_true_frequencies,
                                    axis=2)

    at_a_matrix_np = np.zeros((n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms),
                              dtype=np.float32)

    for j in range(n_canonical_waveforms):
        # shape (n_observations, n_canonical_waveforms, n_phase_shifts)
        unshared_relative_shift = unshared_phase_shifts - unshared_phase_shifts[:, j, :][:, None, :]

        # shape (n_canonical_waveforms, n_observations * n_phase_shifts)
        unshared_relative_shift_flat = unshared_relative_shift.transpose(1, 2, 0).reshape((n_canonical_waveforms, -1))

        # shape (n_canonical_waveforms, n_observations * n_phase_shifts)
        taken_piece_flat = np.take_along_axis(circular_corr_td[j, :, :], unshared_relative_shift_flat, axis=1)

        # shape (n_observations, n_canonical_waveforms, n_phase_shifts)
        taken_piece = taken_piece_flat.reshape(n_canonical_waveforms, n_phase_shifts, n_observations).transpose(2, 0, 1)

        # shape (n_canonical_waveforms, n_phase_shifts)
        at_a_matrix_np[:, :, :, j] = taken_piece.transpose((0, 2, 1))

    return at_a_matrix_np


def build_unshared_at_b_vector(observed_ft: np.ndarray,
                               ft_canonical: np.ndarray,
                               unshared_phase_shifts: np.ndarray,
                               n_true_frequencies: int) -> np.ndarray:
    '''

    :param observed_ft: observed waveforms in Fourier domain, complex valued, shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer, shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    :param n_true_frequencies:
    :return: np.ndarray, shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    '''

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    data_circ_conv_td = np.fft.irfft(observed_ft[:, None, :] * np.conjugate(ft_canonical[None, :, :]),
                                     n=n_true_frequencies,
                                     axis=2)
    # The (i,j,t)^{th} entry corresponds to cross correlation of the i^{th} data waveform with the j^{th} canonical
    #   waveform that has been delayed by t samples

    # we have to build A^T b from this matrix
    # shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    at_b_perm = np.take_along_axis(data_circ_conv_td, unshared_phase_shifts[:, :, :], axis=2)

    # shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    at_b_np = at_b_perm.transpose((0, 2, 1))

    return at_b_np


def fast_time_shifts_and_amplitudes_unshared_shifts(observed_ft: np.ndarray,
                                                    ft_canonical: np.ndarray,
                                                    unshared_phase_shifts: np.ndarray,
                                                    amplitude_matrix_real_np: np.ndarray,
                                                    n_true_frequencies: int,
                                                    max_iter: int,
                                                    device: torch.device,
                                                    l1_regularization_lambda: Optional[float] = None,
                                                    converge_epsilon: float = 1e-3) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''

    :param observed_ft: observed waveforms in Fourier domain, complex valued, shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer, shape (n_observations, n_canonical_waveforms, n_phase_shifts)
    :param amplitude_matrix_real_np: shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    :param n_true_frequencies:
    :param max_iter:
    :param device:
    :param l1_regularization_lambda:
    :param converge_epsilon:
    :return:
    '''

    ##### Generate the appropriate A^T A matrices and A^T b vectors ###########################################

    # shape (n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    unshared_at_a_matrix_np = build_unshared_at_a_matrix(ft_canonical, unshared_phase_shifts, n_true_frequencies)
    unshared_at_a_matrix = torch.tensor(unshared_at_a_matrix_np, dtype=torch.float32, device=device)

    # shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    unshared_at_b_vector_np = build_unshared_at_b_vector(observed_ft,
                                                         ft_canonical,
                                                         unshared_phase_shifts,
                                                         n_true_frequencies)
    unshared_at_b_vector = torch.tensor(unshared_at_b_vector_np, dtype=torch.float32, device=device)

    ##### Set up nonnegative orthant minimization problem ####################################################
    eigenvalues_np, _ = np.linalg.eigh(unshared_at_a_matrix_np)

    # shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)

    max_eigenvalue, _ = torch.max(eigenvalues, dim=2)  # shape (n_observations, n_phase_shifts)
    min_eigenvalue, _ = torch.min(eigenvalues, dim=2)  # shape (n_observations, n_phase_shifts)

    convergence_factor = 0.5 * (max_eigenvalue - min_eigenvalue)  # (n_observations, n_phase_shifts)

    # boundaries for the step size
    # we have to have 0 < step_size <= 1/L where L is the largest eigenvalue
    # we make step_size smaller to be safe
    step_size = 1.0 / (2 * max_eigenvalue)  # has shape (n_observations, n_phase_shifts)

    ##### Step 4: do the projected gradient descent (no batching) ################################
    ##### This function assumes that the batching is taken care of by the caller #################

    # shape (n_observations, n_phase_shifts, n_canonical_waveforms)
    amplitudes = torch.tensor(amplitude_matrix_real_np, dtype=torch.float32, device=device)

    # at_a_x has shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_a_x = (unshared_at_a_matrix @ amplitudes[:, :, :, None]).squeeze(3)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    gradient = at_a_x - unshared_at_b_vector
    if l1_regularization_lambda is not None:
        gradient += l1_regularization_lambda

    for step_num in range(max_iter):

        # shape (n_observations, n_canonical_waveforms)
        next_amplitudes = torch.clamp(amplitudes - step_size[:, :, None] * gradient,
                                      min=0.0)

        # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
        at_a_x = (unshared_at_a_matrix @ next_amplitudes[:, :, :, None]).squeeze(3)

        # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
        gradient = at_a_x - unshared_at_b_vector
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
    xt_at_b = (amplitudes[:, :, None, :] @ unshared_at_b_vector[:, :, :, None]).squeeze()

    partial_objective = 0.5 * xt_at_a_x - xt_at_b

    return amplitudes.cpu().numpy(), partial_objective.cpu().numpy()


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

        objective fn is 1/2 |Ax-b|^2 = 1/2 (Ax-b)^T (Ax-b) = 1/2 x^T A^T A x - x^T A^T b - 1/2 b^T b
        gradient is A^T A x - A^T b

    Implementation notes:

    (A^T A)_{i,j}^{(z)} corresponds to the cross-correlation of the i^{th} canonical waveform, delayed by
        valid_phase_shifts[i,z] number of samples, with the j^{th} canonical waveform, delayed by
        valid_phase_shifts[j,z] number of samples

    :param observed_ft: observed waveforms in Fourier domain, complex valued, shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param amplitude_matrix_real_np: shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    :param valid_phase_shifts: integer array, shape (n_canonical_waveforms, n_valid_phase_shifts)
    :param n_true_frequencies: int
    :return:
    '''

    n_canonical_waveforms, n_valid_phase_shifts = valid_phase_shifts.shape

    #### Step 1: build A^T A from circular cross correlation #####################################

    # shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix_np = build_at_a_matrix(ft_canonical, valid_phase_shifts, n_true_frequencies)

    # shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix = torch.tensor(at_a_matrix_np, dtype=torch.float32, device=device)

    ##### Step 2: build A^T b from circular cross correlation with data matrix ##################
    # this one depends on absolute timing so it is much easier to pack
    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_np = build_at_b_vector(observed_ft, ft_canonical, valid_phase_shifts, n_true_frequencies)

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

    # at_a_x has shape (n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_a_x = (at_a_matrix[None, :, :, :] @ amplitudes[:, :, :, None]).squeeze(3)

    # shape (n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    gradient = at_a_x - at_b_torch
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
        convergence_bound = convergence_factor[None, :] * step_distance  # shape (n_observations, n_valid_phase_shifts)
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
                                              second_pass_best_n: int,
                                              second_pass_width: int,
                                              device: torch.device,
                                              l1_regularization_lambda: Optional[float] = None,
                                              converge_epsilon: float = 1e-3,
                                              amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
                                              max_batch_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:

    '''
    Coarse-to-fine joint fitting of amplitudes and time shifts. Algorithm has two distinct phases:

    Phase (1): Perform a coarse grid search over all possible time shifts for each waveform, solving the nonnegative
        least squares problem for each point on the grid, and taking the second_pass_best_n best grid points
    Phase (2): For each of the second_pass_best_n best grid points, perform a fine search with width second_pass_width
        centered on those best grid points

    Returns the best amplitude, shift

    :param observed_ft: observed waveforms in Fourier domain, complex valued, shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: canonical waveforms in Fourier domain, unshifted, complex valued,
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param n_true_frequencies: int, number of actual (not rFFT) FFT frequencies
    :param valid_phase_shift_range: range of sample shifts to consider
    :param first_pass_step_size: int, step size for the first pass grid search
    :param second_pass_best_n: int, algorithm expands upon the second_pass_best_n best results from the first pass grid
        search to do the second pass grid search
    :param second_pass_width: int, one-sided width around the second_pass_best_n best results from the first pass grid
        search that the algorithm searches for the second pass
    :param device: torch.device
    :param l1_regularization_lambda: float, lambda for L1 regularization of amplitudes
    :param converge_epsilon:
    :param amplitude_initialize_range: range from which we uniformly at random initialize the amplitudes. Should have
        no bearing on the result, but may affect convergence speed
    :param max_batch_size: int, maximum batch size to solve at once
    :return: amplitudes, shifts, each with shape (n_observations, n_canonical_waveforms)
    '''

    n_observations, _ = observed_ft.shape
    n_canonical_waveforms, n_rfft_frequencies = ft_canonical.shape

    ######### Step 1: first pass, perform nonnegative least squares minimization on a coarse ###############
    ######## grid of phase shifts, then pick the N best ####################################################
    low_shift, high_shift = valid_phase_shift_range
    shift_steps = np.r_[low_shift:high_shift:first_pass_step_size]
    mg = np.stack(np.meshgrid(*[shift_steps for _ in range(n_canonical_waveforms)]), axis=0)

    # shape (n_canonical_waveforms, (high_shift - low_shift)^n_canonical_waveforms)
    valid_phase_shifts_matrix = mg.reshape((n_canonical_waveforms, -1))

    _, n_valid_phase_shifts = valid_phase_shifts_matrix.shape

    amplitude_results = np.zeros((n_observations, n_valid_phase_shifts, n_canonical_waveforms),
                                 dtype=np.float32)
    objective_results = np.zeros((n_observations, n_valid_phase_shifts), dtype=np.float32)

    print("First pass coarse search")
    pbar = tqdm.tqdm(total=int(np.ceil(n_valid_phase_shifts / max_batch_size)), leave=False)
    for low in range(0, n_valid_phase_shifts, max_batch_size):
        high = min(n_valid_phase_shifts, low + max_batch_size)

        amplitudes_random_init = np.random.uniform(amplitude_initialize_range[0],
                                                   amplitude_initialize_range[1],
                                                   size=(n_observations, high - low, n_canonical_waveforms))

        amplitude_batch, objective_batch = fast_time_shifts_and_amplitudes_shared_shifts(
            observed_ft,
            ft_canonical,
            valid_phase_shifts_matrix[:, low:high],
            amplitudes_random_init,
            n_true_frequencies,
            10000,
            device,
            l1_regularization_lambda=l1_regularization_lambda,
            converge_epsilon=converge_epsilon
        )
        amplitude_results[:, low:high, :] = amplitude_batch
        objective_results[:, low:high] = objective_batch
        pbar.update(1)
    pbar.close()

    # pick the N best nodes to expand in detail
    # shape (n_observations, second_pass_best_n)
    partition_idx = np.argpartition(objective_results, second_pass_best_n, axis=1)[:, :second_pass_best_n]

    # shape (n_observations * second_pass_best_n, )
    partition_idx_flat = partition_idx.reshape((-1,))

    # shape (n_canonical_waveforms, n_observations * second_pass_best_n)
    best_phases_flat = valid_phase_shifts_matrix[:, partition_idx_flat]
    best_phases = best_phases_flat.reshape((n_canonical_waveforms, n_observations, second_pass_best_n)).transpose(
        (1, 0, 2))

    #### Do the fine search ###################################################

    second_pass_fine_steps = np.r_[-second_pass_width:second_pass_width + 1]

    # define n_fine_phases = (1 + 2 * second_pass_width)^n_canonical_waveforms
    # shape (n_canonical_waveforms, n_fine_phases)
    second_pass_mg = np.stack(np.meshgrid(*[second_pass_fine_steps for _ in range(n_canonical_waveforms)]), axis=0)
    second_pass_mg_flat = second_pass_mg.reshape((second_pass_mg.shape[0], -1))

    # shape (n_observations, n_canonical_waveforms, n_fine_phases, second_past_best_n)
    next_iter_phases = best_phases[:, :, None, :] + second_pass_mg_flat[None, :, :, None]

    # shape (n_observations, n_canonical_waveforms, second_pass_best_n * n_fine_phases)
    next_iter_phases_flat = next_iter_phases.reshape((n_observations, n_canonical_waveforms, -1))
    n_second_pass_shifts = next_iter_phases_flat.shape[2]

    amplitude_results = np.zeros((n_observations, n_second_pass_shifts, n_canonical_waveforms),
                                 dtype=np.float32)
    objective_results = np.zeros((n_observations, n_second_pass_shifts), dtype=np.float32)
    print("Second pass fine grained search")
    pbar = tqdm.tqdm(total=int(np.ceil(n_second_pass_shifts / max_batch_size)), leave=False)
    for low in range(0, n_second_pass_shifts, max_batch_size):
        high = min(n_second_pass_shifts, low + max_batch_size)

        amplitudes_random_init = np.random.uniform(amplitude_initialize_range[0],
                                                   amplitude_initialize_range[1],
                                                   size=(n_observations, high - low, n_canonical_waveforms))

        amplitude_batch, objective_batch = fast_time_shifts_and_amplitudes_unshared_shifts(
            observed_ft,
            ft_canonical,
            next_iter_phases_flat[:, :, low:high],
            amplitudes_random_init,
            n_true_frequencies,
            10000,
            device,
            l1_regularization_lambda=l1_regularization_lambda,
            converge_epsilon=converge_epsilon
        )
        amplitude_results[:, low:high, :] = amplitude_batch
        objective_results[:, low:high] = objective_batch
        pbar.update(1)
    pbar.close()

    #### Now pick the best objective value for each observed waveform ########################
    best_objective = np.argmin(objective_results, axis=1)  # shape (n_observations, )

    best_amplitudes = np.take_along_axis(amplitude_results, best_objective[:, None, None], axis=1).squeeze(1)
    best_shifts = np.take_along_axis(next_iter_phases_flat, best_objective[:, None, None], axis=2).squeeze(2)

    return best_amplitudes, best_shifts
