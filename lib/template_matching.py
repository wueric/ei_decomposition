import numpy as np
import torch

from lib.util_fns import generate_fourier_phase_shift_matrices


def greedy_template_match_time_shift(observed_ft: np.ndarray,
                                     ft_canonical: np.ndarray,
                                     valid_phase_shifts: np.ndarray,
                                     n_true_frequencies: int) -> np.ndarray:
    '''

    :param observed_ft: observed waveforms in Fourier domain, complex valued
            shape (n_observations, n_rfft_frequencies)
    :param ft_canonical: Fourier transform of unshifted canonical waveforms, complex valued
            shape (n_canonical_waveforms, n_rfft_frequencies)
    :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
    :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
    :return:
    '''
    n_observations, _ = observed_ft.shape
    n_canonical_waveforms, n_rfft_frequencies = ft_canonical.shape

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    ft_canonical_time_reversed = np.conjugate(ft_canonical)  # * single_phase_shift_matrix

    # shape (n_observations, n_rfft_Frequencies
    observed_ft_deconv = np.copy(observed_ft)

    deconv_already = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    deconv_time_shifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    for deconv_iter in range(n_canonical_waveforms):
        # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
        cross_corr_ft = observed_ft_deconv[:, None, :] * ft_canonical_time_reversed[None, :, :]
        cross_corr_td = np.fft.irfft(cross_corr_ft, axis=2, n=n_true_frequencies)

        cross_corr_td += (deconv_already[:, :, None] * (-1e9))

        # shape (n_observations, n_canonical_waveforms, n_valid_phase_shifts)
        valid_td_samples = np.take(cross_corr_td, valid_phase_shifts, axis=2)

        # shape (n_observations, n_canonical_waveforms)
        best_phase_shift_idx = np.argmax(valid_td_samples, axis=2)
        best_phase_shift_value = np.take_along_axis(valid_td_samples,
                                                    best_phase_shift_idx[:, :, None], axis=2).squeeze(2)

        # shape (n_observations, )
        best_canonical_waveform = np.argmax(best_phase_shift_value, axis=1)
        best_canonical_waveform_shift = np.take_along_axis(best_phase_shift_idx,
                                                           best_canonical_waveform[:, None], axis=1).squeeze(1)
        best_canonical_waveform_amplitude = np.take_along_axis(best_phase_shift_value,
                                                               best_canonical_waveform[:, None], axis=1).squeeze(1)

        # save outputs and update values for next iteration
        deconv_time_shifts[np.r_[0:n_observations], best_canonical_waveform] = valid_phase_shifts[
            best_canonical_waveform_shift]
        deconv_already[np.r_[0:n_observations], best_canonical_waveform] = 1

        observed_ft_deconv -= ft_canonical_time_reversed[best_canonical_waveform,
                              :] * best_canonical_waveform_amplitude[:, None]

    return deconv_time_shifts


def torch_fit_integer_shifts_all_but_one_template_match(observed_ft: np.ndarray,
                                                        real_amplitude_matrix_np: np.ndarray,
                                                        ft_canonical: np.ndarray,
                                                        previous_shifts_integer: np.ndarray,
                                                        valid_phase_shifts: np.ndarray,
                                                        n_true_frequencies: int,
                                                        device: torch.device) -> np.ndarray:
    '''
        Crude approximate all-but-one template match.

        Description of algorithm:

            For the i^{th} canonical waveform:
                Deconvolve all j != i canonical waveforms from the observed data
                    using fixed amplitude and shift from previous iterations
                Find the optimal shift for the i^{th} waveform with a zero-padded correlation

        :param observation_matrix_np: observed waveforms in Fourier domain, complex valued
            shape (n_observations, n_rfft_frequencies)
        :param real_amplitude_matrix_np: real-valued amplitudes for each observation, each shifted canonical waveform,
            shape (n_observations, n_canonical_waveforms)
        :param ft_canonical: Fourier transform of unshifted canonical waveforms, complex valued
            shape (n_canonical_waveforms, n_rfft_frequencies)
        :param previous_shifts_integer: previous canonical waveform shifts, shape (n_observations, n_canonical_waveforms)
        :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
        :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
        :param device: torch device
        :return: optimal timeshifts, shape (n_observations, n_canonical_waveforms)
        '''

    n_observations, _ = observed_ft.shape
    _, n_canonical_waveforms = real_amplitude_matrix_np.shape

    # component-wise complex representation, shape (2, n_observations, n_rfft_frequencies)
    observed_ft_torch = torch.tensor(np.stack([observed_ft.real, observed_ft.imag], axis=0),
                                     dtype=torch.float32,
                                     device=device)

    # real-valued, shape (n_observations, n_canonical_waveforms)
    real_amplitude_matrix_torch = torch.tensor(real_amplitude_matrix_np, dtype=torch.float32, device=device)

    # component-wise complex representation, shape (2, n_canonical_waveforms, n_rfft_frequencies)
    ft_canonical_torch = torch.tensor(np.stack([ft_canonical.real, ft_canonical.imag],
                                               axis=0),
                                      dtype=torch.float32,
                                      device=device)

    # complex-valued,
    # shape (n_valid_phase_shifts, n_rfft_frequencies)
    phaseshift_all_allowed_matrices = generate_fourier_phase_shift_matrices(valid_phase_shifts,
                                                                            n_true_frequencies)

    # component-wise complex representation, shape (2, n_valid_phase_shifts, n_rfft_frequencies)
    phaseshift_all_allowed_torch = torch.tensor(np.stack([phaseshift_all_allowed_matrices.real,
                                                          phaseshift_all_allowed_matrices.imag],
                                                         axis=0),
                                                dtype=torch.float32,
                                                device=device)

    # complex-valued,
    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(previous_shifts_integer,
                                                                 n_true_frequencies)

    # component-wise complex representation, shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shift_torch = torch.tensor(np.stack([phase_shift_matrices.real, phase_shift_matrices.imag],
                                              axis=0),
                                     dtype=torch.float32,
                                     device=device)

    # complex-valued, multiplication (x + ai) (y + bi) = (xy - ab) + (xb + ay)i
    # shape (2, n_observations, n_canonical_waveforms, n_rfft_frequencies)
    phase_shifted_canonical_ft_torch = torch.stack([
        phase_shift_torch[0, :, :, :] * ft_canonical_torch[0, None, :, :] - \
        phase_shift_torch[1, :, :, :] * ft_canonical_torch[1, None, :, :],

        phase_shift_torch[1, :, :, :] * ft_canonical_torch[0, None, :, :] + \
        phase_shift_torch[0, :, :, :] * ft_canonical_torch[1, None, :, :]
    ], dim=0)

    # real_amplitude_matrix_torch is (n_observations, n_canonical_waveforms)
    # real * complex multiplication
    # shape (2, n_observations, n_canonical_waveforms, n_frequencies)
    scaled_shifted_canonical_ft_torch = real_amplitude_matrix_torch[None, :, :, None] * phase_shifted_canonical_ft_torch

    # shape (2, n_observations, n_frequencies)
    canonical_reconstruction_ft = torch.sum(scaled_shifted_canonical_ft_torch, dim=2)

    # shape (2, n_observations, n_frequencies)
    deconvolved_ft_torch = observed_ft_torch - canonical_reconstruction_ft

    # shape (n_observations, n_canonical_waveforms)
    output_timeshifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)

    for i in range(n_canonical_waveforms):
        # shape (2, n_observations, n_frequencies)
        all_but_one_resid_ft = deconvolved_ft_torch + scaled_shifted_canonical_ft_torch[:, :, i, :]

        # real_amplitude_matrix_torch is (n_observations, n_canonical_waveforms)
        # ft_canonical_torch is (2, n_canonical_waveforms, n_rfft_frequencies)
        # shape (2, n_observations, n_frequencies)
        possible_ft_scaled = real_amplitude_matrix_torch[None, :, i, None] * ft_canonical_torch[:, None, i, :]

        # shape (2, n_observations, n_allowed_shifts, n_frequencies)
        possible_shifts_ft_scaled = torch.stack([
            possible_ft_scaled[0, :, None, :] * phaseshift_all_allowed_torch[0, None, :, :] - \
            possible_ft_scaled[1, :, None, :] * phaseshift_all_allowed_torch[1, None, :, :],

            possible_ft_scaled[0, :, None, :] * phaseshift_all_allowed_torch[1, None, :, :] + \
            possible_ft_scaled[1, :, None, :] * phaseshift_all_allowed_torch[0, None, :, :]
        ], dim=0)

        # shape (2, n_observations, n_allowed_shifts, n_frequencies)
        all_possible_subtracted = all_but_one_resid_ft[:, :, None, :] - possible_shifts_ft_scaled

        # shape (n_observations, n_allowed_shifts)
        magnitudes = torch.sum(all_possible_subtracted * all_possible_subtracted, dim=(0, 3))

        _, min_shift_indices = torch.min(magnitudes, dim=1)
        best_shifts_idx = min_shift_indices.cpu().numpy()

        best_shifts = valid_phase_shifts[best_shifts_idx]

        output_timeshifts[:, i] = best_shifts

    return output_timeshifts


def fit_shifts_all_but_one_template_match(observed_ft: np.ndarray,
                                          real_amplitude_matrix_np: np.ndarray,
                                          ft_canonical: np.ndarray,
                                          previous_shifts_integer: np.ndarray,
                                          valid_phase_shifts: np.ndarray,
                                          n_true_frequencies: int) -> np.ndarray:
    '''
    Crude approximate all-but-one template match.

    Description of algorithm:

        For the i^{th} canonical waveform:
            Deconvolve all j != i canonical waveforms from the observed data
                using fixed amplitude and shift from previous iterations
            Find the optimal shift for the i^{th} waveform with a zero-padded correlation

    :param observation_matrix_np: observed waveforms in time domain,
        shape (n_observations, n_frequencies)
    :param real_amplitude_matrix_np: real-valued amplitudes for each observation, each shifted canonical waveform,
        shape (n_observations, n_canonical_waveforms)
    :param ft_canonical: unshifted canonical waveforms,
        shape (n_canonical_waveforms, n_frequencies)
    :param previous_shifts_integer: previous canonical waveform shifts, shape (n_observations, n_canonical_waveforms)
    :param valid_phase_shifts: allowed phase shifts that we can test, shape (n_valid_phase_shifts, )
    :param n_true_frequencies: int, number of regular FFT frequencies (not the number of rFFT frequencies)
    :return: timeshifts, shape (n_observations, n_canonical_waveforms)
    '''
    n_observations, _ = observed_ft.shape
    _, n_canonical_waveforms = real_amplitude_matrix_np.shape

    # shape (n_valid_phase_shifts, n_frequencies)
    phaseshift_all_allowed_matrices = generate_fourier_phase_shift_matrices(valid_phase_shifts,
                                                                            n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shift_matrices = generate_fourier_phase_shift_matrices(previous_shifts_integer,
                                                                 n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    phase_shifted_canonical_ft = phase_shift_matrices * ft_canonical[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    scaled_shifted_canonical_ft = real_amplitude_matrix_np[:, :, None] * phase_shifted_canonical_ft

    # shape (n_canonical_waveforms, n_frequencies)
    deconvolved_ft = observed_ft - np.sum(scaled_shifted_canonical_ft, axis=1)

    # shape (n_observations, n_canonical_waveforms)
    output_timeshifts = np.zeros((n_observations, n_canonical_waveforms), dtype=np.int32)
    for i in range(n_canonical_waveforms):
        # shape (n_canonical_waveforms, n_frequencies)
        all_but_one_resid_ft = deconvolved_ft + scaled_shifted_canonical_ft[:, i, :]

        # shape (n_observations, n_frequencies)
        possible_ft_scaled = real_amplitude_matrix_np[:, i, None] * ft_canonical[None, i, :]

        # shape (n_observations, n_allowed_shifts, n_frequencies)
        possible_shifts_ft_scaled = possible_ft_scaled[:, None, :] * phaseshift_all_allowed_matrices[None, :, :]

        # shape (n_observations, n_allowed_shifts, n_frequencies)
        all_possible_subtracted = all_but_one_resid_ft[:, None, :] - possible_shifts_ft_scaled

        # shape (n_observations, n_allowed_shifts)
        magnitudes = np.linalg.norm(all_possible_subtracted, axis=2)

        # shape (n_observations, )
        best_shifts_idx = np.argmin(magnitudes, axis=1)
        best_shifts = valid_phase_shifts[best_shifts_idx]

        output_timeshifts[:, i] = best_shifts

    return output_timeshifts


def simple_deconv_time_shifts(waveform_data_matrix: np.ndarray,
                              normalized_canonical_waveforms: np.ndarray,
                              valid_sample_shifts: np.ndarray) -> np.ndarray:
    '''
    Estimate time shifts by calculating cross-correlations in Fourier domain

    :param waveform_data_matrix: np.ndarray, shape (n_observations, n_timepoints)
    :param normalized_canonical_waveforms: np.ndarray, shape (n_canonical_waveforms, n_timepoints)
        already normalized such that the L2 norm for each row is 1
    :param valid_sample_shifts: np.ndarray, all valid shifts
    :return: delays, shape (n_observations, n_canonical_waveforms)
    '''

    # shape (n_observations, n_rfft_frequencies)
    observed_ft = np.fft.rfft(waveform_data_matrix, axis=1)

    n_canonical_waveforms, n_timepoints = normalized_canonical_waveforms.shape

    # because we are calculating a cross-correlation, we first reverse the
    # the time domain canonical waveforms
    canonical_td_reversed = normalized_canonical_waveforms[:, ::-1]
    canonical_td_reversed = np.roll(canonical_td_reversed, 1, axis=1)

    # shape (n_canonical_waveforms, n_rfft_frequencies)
    canonical_reverse_ft = np.fft.rfft(canonical_td_reversed, axis=1)

    # shape (n_observations, n_canonical_waveforms, n_rfft_frequencies)
    cross_correlation_ft = observed_ft[:, None, :] * canonical_reverse_ft[None, :, :]

    # shape (n_observations, n_canonical_waveforms, n_timepoints)
    cross_correlation_td = np.fft.irfft(cross_correlation_ft, axis=2)

    # shape (n_observations, n_canonical_waveforms, n_valid_shifts)
    valid_timeshift_cross_correlations = np.take(cross_correlation_td, valid_sample_shifts, axis=2)

    # shape (n_observations, n_canonical_waveforms)
    best_timeshift_indices = np.argmax(valid_timeshift_cross_correlations, axis=2)
    return np.take(valid_sample_shifts, best_timeshift_indices, axis=0)