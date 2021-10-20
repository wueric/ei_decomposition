import numpy as np
from typing import Optional

from .util_fns import generate_fourier_phase_shift_matrices


def calculate_spatial_continuity_penalty(fitted_amplitudes_by_cell: np.ndarray,
                                         observed_norms_by_cell: np.ndarray,
                                         mean_matrices_by_cell: np.ndarray,
                                         lambda_spatial: float,
                                         use_scaled_spatial_penalty: bool = False) -> float:
    '''
    Calculates the value of the spatial continuity penalty

    :param fitted_amplitudes_by_cell: real-value scaled amplitudes of each canonical waveform
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    :param observed_norms_by_cell: scaling factor for the data, i.e. if you multiply this with the scaled version
        of the data, you get back the original data
        shape (n_cells, max_n_electrodes)
    :param mean_matrices_by_cell: mean operator matrix (right multiply) for calculating the spatial mean for
        each cell.
        shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param lambda_spatial:
    :return:
    '''

    n_cells, max_n_electrodes, n_basis_waveforms = fitted_amplitudes_by_cell.shape

    # shape (n_cells, max_n_electrodes, n_basis_waveforms)
    rescaled_fitted_amplitudes_by_cell = (fitted_amplitudes_by_cell * observed_norms_by_cell[:, :, None])

    # shape (n_cells, n_basis_waveforms, max_n_electrodes)
    rescaled_fitted_amplitudes_by_cell_t = rescaled_fitted_amplitudes_by_cell.transpose(0, 2, 1)

    # shape (n_cells, n_basis_waveforms, max_n_electrodes)
    am = rescaled_fitted_amplitudes_by_cell_t @ mean_matrices_by_cell

    # this is AM - A
    # shape (n_cells, n_basis_waveforms, max_n_electrodes)
    diff = am - rescaled_fitted_amplitudes_by_cell_t

    if not use_scaled_spatial_penalty:
        scaled_diff = diff / observed_norms_by_cell[:, None, :]
        return 0.5 * np.sum(scaled_diff * scaled_diff) * lambda_spatial
    else:
        return 0.5 * np.sum(diff * diff) * lambda_spatial


def calculate_l1_penalty_flat(fit_real_amplitudes_scaled: np.ndarray,
                              observed_norms: np.ndarray,
                              lambda_l1: float,
                              use_scaled_l1_penalty: bool = False) -> float:
    '''
    Calculates the value of the L1 penalty if the input is flat

    :param fit_real_amplitudes_scaled: real-valued scaled amplitudes of each canonical waveform
        shape (n_observations, n_canonical_waveforms)
    :param observed_norms: original scaling factor for the data, i.e. if you multiply this with the scaled version of
        the data, you get back the original data
        shape (n_observations, )
    :param lambda_l1: weight for the L1 penalty
    :return: the L1 loss
    '''

    # shape (n_observations, )
    l1_by_observation = np.sum(fit_real_amplitudes_scaled, axis=1)
    if not use_scaled_l1_penalty:
        return np.dot(l1_by_observation, observed_norms) * lambda_l1
    else:
        return np.sum(l1_by_observation) * lambda_l1


def calculate_l1_penalty_by_cell(fit_real_amplitudes_by_cell_scaled: np.ndarray,
                                 norm_scale_factor_by_cell: np.ndarray,
                                 n_valid_electrodes_by_cell: np.ndarray,
                                 lambda_l1: float,
                                 use_scaled_l1_penalty: bool = False) -> float:
    '''
    Calculate the value of the L1 penalty, if the input is structured by cell

    :param fit_real_amplitudes_by_cell_scaled: real-valued scaled amplitudes of each basis waveform, organized by cell
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    :param norm_scale_factor_by_cell: original scaling factor for the data, i.e. if you multiply this with the scaled
        version of the data, you get back the original data
        shape (n_cells, max_n_electrodes)
    :param n_valid_electrodes_by_cell: integer-valued, number of valid electrodes for each cell
        shape (n_cells, )
    :param lambda_l1: weight for the L1 penalty
    :param use_scaled_l1_penalty:
    :return:
    '''

    n_cells, max_n_electrodes, n_basis_waveforms = fit_real_amplitudes_by_cell_scaled.shape

    l1_acc = 0.0

    for cell_idx in range(n_cells):
        n_channels_for_cell = n_valid_electrodes_by_cell[cell_idx]

        # shape (n_channels_for_cell, )
        l1_for_cell = np.sum(fit_real_amplitudes_by_cell_scaled[cell_idx, :n_channels_for_cell, :], axis=1)

        if not use_scaled_l1_penalty:
            norm_scale_relevant = norm_scale_factor_by_cell[cell_idx, :n_channels_for_cell]
            l1_acc += np.dot(norm_scale_relevant, l1_for_cell) * lambda_l1
        else:
            l1_acc += np.sum(l1_for_cell) * lambda_l1

    return l1_acc


def batch_evaluate_mse_flat(batch_observed_scaled_ft: np.ndarray,
                            batch_fit_real_amplitudes_scaled: np.ndarray,
                            batch_canonical_waveform_ft: np.ndarray,
                            batch_time_shifts: np.ndarray,
                            batch_valid_mask: np.ndarray,
                            n_true_frequencies: int,
                            use_scaled_mse: bool = False,
                            batch_observed_norms: Optional[np.ndarray] = None,
                            take_mean_over_electrodes: bool = False) -> float:
    '''
    Calculates the unscaled MSE loss

    (i.e. how well does the decomposition fit the original raw data, with no extra scaling factors involved?)

    :param batch_observed_scaled_ft: scaled Fourier transform of observed data, complex-valued,
        shape (batch, n_observations, n_rfft_frequencies)
    :param batch_fit_real_amplitudes_scaled: real-valued scaled amplitudes of each canonical waveform
        shape (batch, n_observations, n_canonical_waveforms)
    :param batch_canonical_waveform_ft: Fourier transform of canonical waveforms, complex-valued
        shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    :param batch_time_shifts: Time shifts required for each basis waveform to fit each observation. Integer valued
        shape (batch, n_observations, n_canonical_waveforms)
    :param batch_observed_norms: original scaling factor for the data, i.e. if you multiply this with observed_scaled_ft,
        you should get the original unscaled Fourier transform
        shape (batch, n_observations)
    :param batch_valid_mask: shape (batch, n_observations), 0-1 integer valued, 1 if the corresponding observed waveform
        is a legit signal and should be included in the loss calculation; 0 if it is padding and should not
        be included in the loss calculation
    :param n_true_frequencies: int, the number of normal FFT frequencies (not the number of rFFT frequencies)

    :return: MSE error (not including regularization terms), shape (batch, )

        We either sum or take the mean along the observation dimension
    '''

    if not use_scaled_mse and batch_observed_norms is None:
        raise ValueError("If using scaled MSE loss, must also specify norm scaling factors")

    batch_valid_mask_float = batch_valid_mask.astype(np.float32)

    batch, n_observations, _ = batch_observed_scaled_ft.shape

    batched_time_shift_matrices = generate_fourier_phase_shift_matrices(batch_time_shifts,
                                                                        n_true_frequencies)

    # shape (batch, n_observations, n_basis_waveforms, n_frequencies)
    shifted_no_scale_ft = batch_canonical_waveform_ft[:, None, :, :] * batched_time_shift_matrices

    # shape (batch, n_observations, 1, n_basis_waveforms) @ (batch, n_observations, n_basis_waveforms, n_frequencies)
    # -> (batch, n_observations, 1, n_frequencies) -> (batch, n_observations, n_frequencies)
    model_ft = np.squeeze(batch_fit_real_amplitudes_scaled[:, :, None, :] @ shifted_no_scale_ft, axis=2)

    # we have to apply the mask here to hide "observations" that correspond to padding
    diff = (batch_observed_scaled_ft - model_ft) * batch_valid_mask_float[:, :, None]

    # shape (batch, )
    n_valid_per_batch = np.sum(batch_valid_mask_float, axis=1)

    # shape (batch, n_observations)
    errors = np.linalg.norm(diff, axis=2)

    if not use_scaled_mse:

        # shape (batch, n_observations) * (batch, n_observations) ->
        # shape (batch, )
        scaled_errors = np.sum(errors * batch_observed_norms, axis=1)

        if take_mean_over_electrodes:
            scaled_errors = scaled_errors / n_valid_per_batch
        return scaled_errors

    else:
        scaled_errors = np.sum(errors, axis=1)
        if take_mean_over_electrodes:
            scaled_errors = scaled_errors / n_valid_per_batch
        return scaled_errors


def evaluate_mse_flat(observed_scaled_ft: np.ndarray,
                      fit_real_amplitudes_scaled: np.ndarray,
                      canonical_waveform_ft: np.ndarray,
                      time_shifts: np.ndarray,
                      n_true_frequencies: int,
                      use_scaled_mse: bool = False,
                      observed_norms: Optional[np.ndarray] = None,
                      take_mean_over_electrodes: bool = False) -> float:
    '''
    Calculates the unscaled MSE loss

    (i.e. how well does the decomposition fit the original raw data, with no extra scaling factors involved?)

    :param observed_scaled_ft: scaled Fourier transform of observed data, complex-valued,
        shape (n_observations, n_rfft_frequencies)
    :param fit_real_amplitudes_scaled: real-valued scaled amplitudes of each canonical waveform
        shape (n_observations, n_canonical_waveforms)
    :param observed_norms: original scaling factor for the data, i.e. if you multiply this with observed_scaled_ft,
        you should get the original unscaled Fourier transform
        shape (n_observations, )
    :param canonical_waveform_ft: Fourier transform of canonical waveforms, complex-valued
        shape (n_canonical_waveforms, n_rfft_frequencies)
    :param time_shifts: Time shifts required for each basis waveform to fit each observation. Integer valued
        shape (n_observations, n_canonical_waveforms)
    :param n_true_frequencies: int, the number of normal FFT frequencies (not the number of rFFT frequencies)

    :return: MSE error (not including loss terms)
    '''

    n_observations, _ = observed_scaled_ft.shape

    time_shift_matrices = generate_fourier_phase_shift_matrices(time_shifts,
                                                                n_true_frequencies)

    # shape (n_observations, n_canonical_waveforms, n_frequencies)
    shifted_no_scale_ft = canonical_waveform_ft[None, :, :] * time_shift_matrices

    # shape (n_observations n_rfft_frequencies)
    model_ft = np.squeeze(fit_real_amplitudes_scaled[:, None, :] @ shifted_no_scale_ft, axis=1)
    diff = observed_scaled_ft - model_ft

    # shape (n_observations, )
    errors = np.linalg.norm(diff, axis=1)

    if not use_scaled_mse:
        scaled_errors = errors.dot(observed_norms)
        if take_mean_over_electrodes:
            return scaled_errors / n_observations
        return scaled_errors
    else:
        if take_mean_over_electrodes:
            return np.mean(errors)
        return np.sum(errors)


def evaluate_mse_by_cell(observed_ft_by_cell_scaled: np.ndarray,
                         fit_real_amplitudes_by_cell_scaled: np.ndarray,
                         n_valid_electrodes_by_cell: np.ndarray,
                         canonical_waveform_ft: np.ndarray,
                         time_shifts: np.ndarray,
                         n_true_frequencies: int,
                         norm_scale_factor_by_cell: Optional[np.ndarray] = None,
                         use_scaled_mse: bool = False,
                         take_mean_over_valid_electrodes: bool = False):
    '''
    Calculates the MSE loss for data arranged in the by-cell format

    :param observed_ft_by_cell_scaled: Scaled Fourier transform of the observed EI data waveforms,
        shape (n_cells, max_n_electrodes, n_rfft_frequencies)
    :param fit_real_amplitudes_by_cell_scaled: real-valued scaled amplitudes for each basis waveform,
        shape (n_cells, max_n_electrodes, n_basis_waveforms
    :param n_valid_electrodes_by_cell: integer-valued np.ndarray, number of valid electrodes for each cell,
        shape (n_cells, )
    :param canonical_waveform_ft: Fourier transform of the basis waveforms,
        shape (n_basis_waveforms, n_rfft_frequencies)
    :param time_shifts: Time shifts, integer valued,
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    :param n_true_frequencies: int, number of true FFT frequencies (not rFFT frequencies)
    :param norm_scale_factor_by_cell: np.ndarray, original scaling factor for the data, i.e. if you multiply this with
        observed_ft_by_cell_scaled, you should get the original unscaled Fourier transform
        shape (n_cells, max_n_electrodes)
    :param use_scaled_mse:
    :return: MSE loss
    '''

    n_cells, max_n_electrodes, n_rfft_frequencies = observed_ft_by_cell_scaled.shape

    mse_acc = 0.0
    for cell_idx in range(n_cells):

        n_valid_electrodes = n_valid_electrodes_by_cell[cell_idx]

        # shape (n_valid_electrodes, n_basis_waveforms)
        relevant_time_shifts = time_shifts[cell_idx, :n_valid_electrodes, :]
        relevant_amplitudes = fit_real_amplitudes_by_cell_scaled[cell_idx, :n_valid_electrodes, :]

        # shape (n_valid_electrodes, n_basis_waveforms, n_rfft_frequencies)
        time_shift_matrices = generate_fourier_phase_shift_matrices(relevant_time_shifts, n_true_frequencies)

        # shape (n_valid_electrodes, n_basis_waveforms, n_rfft_frequencies)
        shifted_basis_ft = time_shift_matrices * canonical_waveform_ft[None, :, :]

        # shape (n_valid_electrodes, n_rfft_frequencies)
        model_ft = np.squeeze(relevant_amplitudes[:, None, :] @ shifted_basis_ft, axis=1)

        # shape (n_valid_electrodes, n_rfft_frequencies)
        diff = observed_ft_by_cell_scaled[cell_idx, :n_valid_electrodes, :] - model_ft

        # shape (n_valid_electrodes, )
        errors = np.linalg.norm(diff, axis=1)

        if not use_scaled_mse:
            mse_acc += errors.dot(norm_scale_factor_by_cell[cell_idx, :n_valid_electrodes])
        else:
            mse_acc += np.sum(errors)

    if take_mean_over_valid_electrodes:
        return mse_acc / np.sum(n_valid_electrodes_by_cell)
    return mse_acc


def flat_pack_evaluate_loss(observed_ft_flat_scaled: np.ndarray,
                            fit_real_amplitudes_scaled: np.ndarray,
                            norm_scale_factor: np.ndarray,
                            canonical_waveform_ft: np.ndarray,
                            time_shifts_flat: np.ndarray,
                            n_true_frequencies: int,
                            lambda_l1: float = 0.0,
                            use_scaled_reg_penalty: bool = False,
                            use_scaled_mse: bool = False):
    '''
    Calculates the overall loss (including possible L1 regularization terms) for the
        optimization problem

    We cannot calculate the spatial continuity penalty using this method, since we don't have any spatial
        structure in the input data here

    Use by_cell_evaluate_loss if the spatial continuity penalty is important

    :param observed_ft_flat_scaled: Fourier transform of the observed data, scaled by some normalization factor
        shape (n_observations, n_rfft_frequencies)
    :param fit_real_amplitudes_scaled: real-valued scaled amplitudes of each canonical waveform
        shape (n_observations, n_canonical_waveforms)
    :param norm_scale_factor: np.ndarray, original scaling factor for the data, i.e. if you multiply this with
        observed_ft_flat_scaled, you should get the original unscaled Fourier transform
        shape (n_observations, ).
    :param canonical_waveform_ft: Fourier transform of canonical waveforms, complex-valued
        shape (n_canonical_waveforms, n_rfft_frequencies)
    :param time_shifts_flat: Time shifts required for each basis waveform to fit each observation. Integer valued
        shape (n_observations, n_canonical_waveforms)
    :param n_true_frequencies: int, the number of normal FFT frequencies (not the number of rFFT frequencies)
    :param lambda_l1: float, weight for the L1 regularization term. 0.0 if not used
    :return: total loss, with the optional L1 penalty
    '''

    mse_loss = evaluate_mse_flat(observed_ft_flat_scaled,
                                 fit_real_amplitudes_scaled,
                                 canonical_waveform_ft,
                                 time_shifts_flat,
                                 n_true_frequencies,
                                 use_scaled_mse=use_scaled_mse,
                                 observed_norms=None if use_scaled_mse else norm_scale_factor)

    l1_loss = 0.0
    if lambda_l1 != 0.0:
        l1_loss = calculate_l1_penalty_flat(fit_real_amplitudes_scaled,
                                            norm_scale_factor,
                                            lambda_l1,
                                            use_scaled_l1_penalty=use_scaled_reg_penalty)

    return mse_loss + l1_loss


def by_cell_evaluate_loss(observed_ft_by_cell_scaled: np.ndarray,
                          fit_real_amplitudes_by_cell_scaled: np.ndarray,
                          n_valid_electrodes_by_cell: np.ndarray,
                          norm_scale_factor_by_cell: np.ndarray,
                          canonical_waveform_ft: np.ndarray,
                          time_shifts_by_cell: np.ndarray,
                          n_true_frequencies: int,
                          lambda_l1: float = 0.0,
                          lambda_spatial: float = 0.0,
                          neighborhood_mean_matrices: Optional[np.ndarray] = None,
                          use_scaled_reg_penalty: bool = False,
                          use_scaled_mse: bool = False) -> float:
    '''
    Calculates the overall loss (including possible L1 regularization term as well as possible spatial
        continuity term) for the optimization problem

    :param observed_ft_by_cell_scaled: Scaled Fourier transform of the observed EI data waveforms,
        shape (n_cells, max_n_electrodes, n_rfft_frequencies)
    :param fit_real_amplitudes_by_cell_scaled: real-valued scaled amplitudes for each basis waveform,
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    :param n_valid_electrodes_by_cell: integer-valued np.ndarray, number of valid electrodes for each cell,
        shape (n_cells, )
    :param norm_scale_factor_by_cell: np.ndarray, original scaling factor for the data, i.e. if you multiply this with
        observed_ft_by_cell_scaled, you should get the original unscaled Fourier transform
        shape (n_cells, max_n_electrodes).
    :param canonical_waveform_ft: Fourier transform of the basis waveforms,
        shape (n_basis_waveforms, n_rfft_frequencies)
    :param time_shifts_by_cell: Time shifts, integer valued,
        shape (n_cells, max_n_electrodes, n_basis_waveforms)
    :param n_true_frequencies: int, number of true FFT frequencies (not rFFT frequencies)
    :param lambda_l1: scalar multiple for the L1 penalty, 0.0 if not used
    :param lambda_spatial: scalar multiple for the spatial continuity penalty, 0.0 if not used
    :param neighborhood_mean_matrices: Neighborhood mean matrices for each cell, must be specified if lambda_spatial is
        used. shape (n_cells, max_n_electrodes, max_n_electrodes)
    :param use_scaled_reg_penalty:
    :param use_scaled_mse:
    :return: loss, with optional penalties if specified
    '''

    if lambda_spatial != 0.0 and neighborhood_mean_matrices is None:
        assert False, "Cannot calculate spatial continuity penalty if neighborhood_mean_matrices not specified"

    mse_loss = evaluate_mse_by_cell(observed_ft_by_cell_scaled,
                                    fit_real_amplitudes_by_cell_scaled,
                                    n_valid_electrodes_by_cell,
                                    canonical_waveform_ft,
                                    time_shifts_by_cell,
                                    n_true_frequencies,
                                    norm_scale_factor_by_cell=norm_scale_factor_by_cell,
                                    use_scaled_mse=use_scaled_mse)

    l1_loss = 0.0
    if lambda_l1 != 0.0:
        l1_loss = calculate_l1_penalty_by_cell(fit_real_amplitudes_by_cell_scaled,
                                               norm_scale_factor_by_cell,
                                               n_valid_electrodes_by_cell,
                                               lambda_l1,
                                               use_scaled_l1_penalty=use_scaled_reg_penalty)

    spat_cont_loss = 0.0
    if lambda_spatial != 0.0:
        spat_cont_loss = calculate_spatial_continuity_penalty(fit_real_amplitudes_by_cell_scaled,
                                                              norm_scale_factor_by_cell,
                                                              neighborhood_mean_matrices,
                                                              lambda_spatial,
                                                              use_scaled_spatial_penalty=use_scaled_reg_penalty)

    return spat_cont_loss + l1_loss + mse_loss
