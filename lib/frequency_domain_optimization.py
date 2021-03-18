from typing import Optional

import numpy as np
import torch

from lib.util_fns import generate_fourier_phase_shift_matrices


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
    soln, _ = torch.solve(joint_rhs[:, :, None], joint_coeff)

    # shape (2 * n_canonical_waveforms, n_rfft_frequencies)
    soln_perm = soln.squeeze(2).permute(1, 0)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    soln_real_seg = soln_perm[:n_canonical_waveforms, :].cpu().numpy()
    soln_imag_seg = soln_perm[n_canonical_waveforms:, :].cpu().numpy()

    return soln_real_seg + 1j * soln_imag_seg
