from typing import Optional

import numpy as np
import torch


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