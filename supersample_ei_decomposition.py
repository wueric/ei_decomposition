import visionloader as vl
import numpy as np
import scipy.interpolate as interpolate

import torch

import argparse

from typing import List, Dict, Tuple, Callable


def bspline_interpolate_waveforms(waveforms: np.ndarray,
                                  bspl_valid_shifts: np.ndarray) -> np.ndarray:
    '''

    :param waveform: shape (n_waveforms, n_samples) original waveforms
    :param bspl_valid_shifts : shape (n_shifts, ) possible shifts that we
        are interested in
    :return: upsampled waveforms, shape (n_waveforms, n_shifts, n_samples)
    '''

    n_waveforms, n_orig_samples = waveforms.shape
    n_shifts = bspl_valid_shifts.shape[0]
    bspl_shifted = np.zeros((n_waveforms, n_shifts, n_orig_samples), dtype=np.float32)

    orig_time_samples = np.r_[0:n_orig_samples]  # shape (n_samples, )
    all_shifted_timepoints = orig_time_samples[None, :] - bspl_valid_shifts[:, None]
    # shape (n_shifts, n_orig_samples)

    # note that we do have to worry a bit about including samples
    # that are outside the interpolation range
    # especially since we will be shifting by
    # more than one sample
    out_of_sample_timepoints = np.logical_or.reduce([
        all_shifted_timepoints < 0,
        all_shifted_timepoints > orig_time_samples[-1]
    ])

    for idx in range(n_waveforms):
        orig_waveform_1d = waveforms[idx, :]
        bspline = interpolate.splrep(orig_time_samples, orig_waveform_1d)

        waveform_shifted = interpolate.splev(all_shifted_timepoints, bspline)
        # shape (n_shifts, n_orig_samples)

        # now invalidate out-of-sample samples
        waveform_shifted[out_of_sample_timepoints] = 0.0
        bspl_shifted[idx, :, :] = waveform_shifted

    return bspl_shifted


def torch_fixed_step_size_waveform_nonneg_orthant_min(batched_targets: torch.Tensor,
                                                      shifted_basis_functions: torch.Tensor,
                                                      max_iter: int,
                                                      converge_epsilon: float,
                                                      device: torch.device,
                                                      x_unif_init_ceiling: float = 10.0,
                                                      batch_one_iter=8192) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Implementation with fixed step size

    The problem we want to solve:

        minimize g(x) = 1/2 * (Ax-w)^T (Ax-w)
        subject to x >= 0

    where w \in R^{n_samples}, w \in R^{n_canonical_waveforms}, and
        A \in R^{n_samples x n_canonical_waveforms}

    The gradient is \grad g(x) = A^T (Ax - w)

    The algorithm we implement is proximal gradient descent with fixed
        step size.

    Because the objective function is quadratic, we can easily find
        L (for L-Lipschitz continuity of the gradient) and m
        (for m-strong convexity) from the eigenvalues of A^T A,
        and we set step size t accordingly

    :param batched_targets: shape (n_cells, n_channels, n_samples) batched w vector
    :param shifted_basis_functions: shape (n_canonical_waveforms, n_shifts, n_samples) temporary
        tensor containing the waveforms that we build A matrices out of
    :param converge_epsilon:
    :param max_iter:
    :param device:
    :param x_unif_init_ceiling:
    :return:
    '''
    n_cells, n_channels, n_samples = batched_targets.shape
    n_waveforms, n_shifts, _ = shifted_basis_functions.shape  # temporary data structure

    # shape (n_cells * n_channels, n_samples)
    batched_targets_flattened = batched_targets.reshape((-1, n_samples))

    # shape (n_samples, n_cells * n_channels)
    batched_targest_flat_permute = batched_targets_flattened.permute(1, 0)

    # Construct batched A matrices for the problem
    #
    # We need variable dimension basis tensor depending on how many types
    #   of waveforms we want to include
    #
    # We are solving n_shifts^{n_canonical_waveforms} number of linear systems
    # Dimension of the output is (n_shifts, ..., n_shifts, n_samples, n_waveforms)

    basis_mat_shape = [n_shifts for _ in range(n_waveforms)]
    basis_mat_shape.extend([n_samples, n_waveforms])
    # batched A matrix, shape (n_shifts, ...,  n_shifts, n_samples, n_waveforms)
    all_basis_matrix = torch.zeros(tuple(basis_mat_shape), dtype=torch.float32, device=device)
    for i in range(n_waveforms):

        # we have to build the slice thing since we have a variable number of dimensions
        slice_list = [i, ]
        for _ in range(i):
            slice_list.append(None)
        slice_list.append(slice(None))
        for _ in range(i + 1, n_waveforms):
            slice_list.append(None)
        slice_list.append(slice(None))
        slice_tup = tuple(slice_list)

        all_basis_matrix[..., i] += shifted_basis_functions[slice_tup]

    # now want to use proximal gradient descent to solve
    # nonnegative orthant least squares problem
    # for a whole bunch of systems all at once

    # let's stick with a simple implementation where we first reshape
    #   all_basis_matrix into a 3D tensor (batch, n_samples, n_waveforms)
    #   which is much easier to handle with torch semantics
    #   batch = n_shifts * ... * n_shifts
    all_basis_shape = all_basis_matrix.shape

    batched_a_matrix = all_basis_matrix.reshape((-1, all_basis_shape[-2], all_basis_shape[-1]))
    # shape (batch, n_samples, n_waveforms)

    n_problems = batched_a_matrix.shape[0]

    # first calculate what the fixed step sizes are for each system
    # this requires calculating A^T A and finding the eigenvalues
    at_a = batched_a_matrix.permute((0, 2, 1)) @ batched_a_matrix  # shape (batch, n_waveforms, n_waveforms)
    print(at_a.shape)
    print("Calculating eigenvalues")
    at_a_numpy = at_a.cpu().numpy()
    eigenvalues_np, _ = np.linalg.eigh(at_a_numpy)
    print("Done calculating numpy eigenvalues")
    eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)
    print(eigenvalues.shape)

    # eigenvalues has shape (batch, n_waveforms)
    max_eigenvalue, _ = torch.max(eigenvalues, dim=1)  # shape (batch, )
    min_eigenvalue, _ = torch.min(eigenvalues, dim=1)  # shape (batch, )

    convergence_factor = 0.5 * (max_eigenvalue - min_eigenvalue)  # shape (batch, )

    # boundaries for the step size
    step_size = 1.0 / min_eigenvalue  # has shape (batch, )

    # Order of 1e6 3x3 systems is too much to fit on GPU
    # so we will need to batch solve the systems
    solved_objective_values = torch.zeros((n_problems, n_channels * n_cells), dtype=torch.float32, device=device)
    solved_weights = torch.zeros((n_problems, n_waveforms, n_channels * n_cells), dtype=torch.float32, device=device)

    for batch_low in range(0, n_problems, batch_one_iter):

        batch_high = min(batch_low + batch_one_iter, n_problems)
        batch_size = batch_high - batch_low

        batched_a_matrix_chunk = batched_a_matrix[batch_low:batch_high,:,:]

        # randomly initialize x
        batched_x_vector = torch.empty((batch_size, n_waveforms, n_cells * n_channels),
                                       dtype=torch.float32,
                                       device=device)
        torch.nn.init.uniform_(batched_x_vector, 0, x_unif_init_ceiling)

        ax_minus_b = batched_a_matrix_chunk @ batched_x_vector - batched_targest_flat_permute[None, :, :]
        # shape (batch, n_samples, n_cells * n_channels)

        gradient = batched_a_matrix_chunk.permute(0, 2, 1) @ ax_minus_b
        # has shape (batch, n_waveforms, n_samples) x (batch, n_samples, n_cells * n_channels) =
        #   (batch, n_waveforms, n_cells * n_channels)

        # main loop for the algorithm
        for step_num in range(max_iter):
            print(step_num)

            # apply the step and proximal operator
            next_x_step = batched_x_vector - step_size[:, None, None] * gradient
            next_x_step = torch.clamp_min(0.0, next_x_step)  # shape (batch, n_waveforms, n_cells * n_channels)

            ax_minus_b = batched_a_matrix_chunk @ batched_x_vector - batched_targest_flat_permute[None, :, :]
            # shape (batch, n_samples, n_cells * n_channels)

            gradient = batched_a_matrix_chunk.permute(0, 2, 1) @ ax_minus_b
            # has shape (batch, n_waveforms, n_samples) x (batch, n_samples, n_cells * n_channels) =
            #   (batch, n_waveforms, n_cells * n_channels)

            step_distance = torch.norm(next_x_step - batched_x_vector, dim=1)  # shape (batch, n_cells * n_channels)
            convergence_bound = convergence_factor[:, None] * step_distance
            worst_bound = torch.max(convergence_bound)[0, 0]

            batched_x_vector = next_x_step
            if worst_bound < converge_epsilon:
                break

        ax_minus_b = batched_a_matrix_chunk @ batched_x_vector - batched_targets_flattened[None, :, :]
        # shape (batch, n_samples, n_cells * n_channels)
        objective_value = 0.5 * torch.sum(ax_minus_b * ax_minus_b, dim=1)
        # shape (batch, n_cells * n_channels)

        solved_objective_values[batch_low:batch_high,:] = objective_value
        solved_weights[batch_low:batch_high,:,:] = batched_x_vector

    return solved_objective_values, solved_weights


if __name__ == '__main__':
    device = torch.device('cuda')

    # for now, don't bother with argparse since we still don't have an automatic way
    # to pick canonical waveforms
    dataset = vl.load_vision_data('/Volumes/Lab/Users/ericwu/yass-ei/2018-03-01-0/data001',
                                  'data001',
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()

    # hardcoded test data
    example_cell = 616
    dendritic_electrode = 78
    somatic_electrode = 61
    axonic_electrode = 300

    ei_example = dataset.get_ei_for_cell(example_cell).ei
    normalized_dendritic = ei_example[dendritic_electrode, :] / np.linalg.norm(ei_example[dendritic_electrode, :])
    normalized_somatic = ei_example[somatic_electrode, :] / np.linalg.norm(ei_example[somatic_electrode, :])
    normalized_axonic = ei_example[axonic_electrode, :] / np.linalg.norm(ei_example[axonic_electrode, :])

    canonical_waveforms_unshifted = np.array([normalized_dendritic, normalized_somatic, normalized_axonic],
                                             dtype=np.float32)  # shape (n_waveforms, n_samples)
    canonical_waveforms_with_shifts = bspline_interpolate_waveforms(canonical_waveforms_unshifted,
                                                                    np.r_[-10.0:10.0:1])

    canonical_waveforms_with_shifts_torch = torch.tensor(canonical_waveforms_with_shifts,
                                                         dtype=torch.float32,
                                                         device=device)

    batched_targets_torch = torch.tensor(ei_example[None, :, :], dtype=torch.float32, device=device)

    objective_fn, x_vals = torch_fixed_step_size_waveform_nonneg_orthant_min(batched_targets_torch,
                                                                             canonical_waveforms_with_shifts_torch,
                                                                             50,
                                                                             1e-3,
                                                                             device)
