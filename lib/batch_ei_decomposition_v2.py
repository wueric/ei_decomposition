import torch
import torch.nn as nn

import numpy as np

from lib.optim.proxgrad_optim import BatchedMultiProxProblem

from typing import Union, Optional, Tuple


class SharedShiftsProxGradSolver(BatchedMultiProxProblem):

    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    1. norms, shape (batch, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0
    NORMS_IDX_ARGS = 1

    def __init__(self,
                 batched_shared_at_a_matrix: np.ndarray,
                 at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l12_lambda: float,
                 l12_group_sel_matrix: np.ndarray,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = -1e-1,
                 init_high: float = 1e-1):
        '''
        Conventions for the variables in this subclass

        (1) dim 0 batch dimension - different cells
        (2) dim 1 problem dimension - we combine the electrodes and phase shifts
            (this requires some reshapes to compute the loss function)

        :param batched_shared_at_a_matrix: np.ndarray, A^T A matrix for each problem
            shape (batch, n_valid_phase_shifts, n_basis_waveforms, n_basis_waveforms)
        :param at_b_vector: np.ndarray, A^T b vector for each problem
            shape (batch, n_electrodes, n_valid_phaseshifts, n_basis_waveforms)

            n_electrodes aka n_observations in the naming convention
        :param valid_problem_mask: np.ndarray, shape (batch, n_electrodes)
        :param l12_lambda:
        :param l12_group_sel_matrix: np.ndarray group assignments for the problem
            every problem is assumed to have the same groups

            value is 1 for membership in group, 0 for no membership in group

            shape (n_groups, n_basis_waveforms)

        :param amplitudes_matrix_init: np.ndarray, initial guesses for the amplitudes if available
            shape (batch, n_electrodes, n_phase_shifts, n_basis)
        :param verbose:
        '''

        temp_batch, n_electrodes, n_phase_shifts, n_basis = at_b_vector.shape
        valid_problem_total = np.zeros((temp_batch, n_electrodes, n_phase_shifts), dtype=bool)
        valid_problem_total[valid_problem_mask, :] = True

        super().__init__(temp_batch,
                         n_electrodes * n_phase_shifts,
                         valid_problems=valid_problem_total.reshape((temp_batch, -1)),
                         verbose=verbose)

        self.n_electrodes, self.n_phase_shifts, self.n_basis = n_electrodes, n_phase_shifts, n_basis
        self.n_groups = l12_group_sel_matrix.shape[0]

        self.l12_lambda = l12_lambda

        # shape (batch, n_valid_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_shared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_valid_phase_shifts, n_basis)
        self.register_buffer('at_b_vector', torch.tensor(at_b_vector, dtype=torch.float32))

        # shape (n_groups, n_basis)
        self.register_buffer('group_sel', torch.tensor(l12_group_sel_matrix), dtype=torch.float32)

        # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
        if amplitudes_matrix_init is None:
            self.amplitudes = nn.Parameter(
                torch.empty((temp_batch, self.n_electrodes * self.n_phase_shifts, self.n_basis)))
            nn.init.uniform_(self.amplitudes, a=init_low, b=init_high)
        else:
            self.amplitudes = nn.Parameter(
                torch.tensor(
                    amplitudes_matrix_init.reshape(temp_batch, self.n_electrodes * self.n_phase_shifts, self.n_basis),
                    dtype=torch.float32),
                requires_grad=True
            )

        # auxiliary variable for the constrained optimization
        # shape (batch, n_electrodes * n_phase_shifts, n_groups)
        self.norms = nn.Parameter(
            torch.empty(self.batch_size, self.n_electrodes * self.n_phase_shifts, self.n_groups)
        )
        nn.init.uniform_(self.norms, a=init_low, b=init_high)

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_phase_shifts), one value for each problem,
            doesn't matter if the problem is valid or not
        '''

        # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
        amplitudes = args[self.AMPLITUDES_IDX_ARGS]

        # shape (batch_size, n_electrodes, n_phase_shifts, n_basis)
        amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts, self.n_basis)

        # (batch, 1, n_valid_phase_shifts, n_basis, n_basis) @
        #       (batch, n_electrodes, n_phase_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_phase_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_phase_shifts, n_basis)
        at_a_x = (self.at_a_matrix[:, None, :, :, :] @ amplitudes_unflat[:, :, :, :, None]).squeeze(-1)

        # shape (batch, n_electrodes, n_phase_shifts)
        xt_at_a_x = torch.sum(amplitudes_unflat * at_a_x, dim=3)

        # shape (batch, n_electrodes, n_phase_shifts)
        xt_at_b = torch.sum(self.at_b_vector * amplitudes_unflat, dim=3)

        # shape (batch, n_electrodes, n_phase_shifts)
        # -> (batch, n_electrodes * n_phase_shifts)
        mse_loss_component = (xt_at_a_x + xt_at_b).reshape(self.batch_size, -1)

        # shape (batch, n_electrodes * n_phase_shifts, n_groups)
        norms = args[self.NORMS_IDX_ARGS]

        total_loss = mse_loss_component + self.l12_lambda * torch.sum(norms, dim=2)

        return total_loss

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]

            # shape (batch, n_electrodes * n_phase_shifts, n_groups)
            norms = args[self.NORMS_IDX_ARGS]

            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            z_amplitudes_clipped = torch.clamp_min_(amplitudes, 0.0)

            # distribute the basis weights into their respective groups
            # as long as self.group_sel does not assign weights to multiple groups
            # we should be able to recover the original basis weights by
            # performing the operation torch.sum(zi_groupsel, dim=2)

            # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
            #   (1, 1, n_groups, n_basis)
            # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            zi_groupsel = z_amplitudes_clipped[:, :, None, :] * self.group_sel[None, None, :, :]

            # shape (batch_size, n_electrodes * n_phase_shifts, n_groups)
            zi_norm = torch.norm(zi_groupsel, dim=3, p=2)

            # shape (batch, n_electrodes * n_phase_shifts, n_groups), boolean-valued
            exceeds_abs = zi_norm > torch.abs(norms)

            # shape (batch, n_electrodes * n_phase_shifts, n_groups), boolean-valued
            less_than_neg = zi_norm < -norms

            # now apply the clips
            # note that we need to project each group differently
            # FIXME
            # (batch_size, n_electrodes * n_phase_shifts, n_groups, 1) *
            #   (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            rescaled_zi_norm_mean = ((zi_norm + norms) / (2.0 * zi_norm))[:, :, :, None] * zi_groupsel

            zi_groupsel[exceeds_abs, :] = rescaled_zi_norm_mean[exceeds_abs, :]
            zi_groupsel[less_than_neg, :] = 0.0

            # shape (batch, n_electrodes * n_phase_shifts, n_basis)
            zi_amplitudes_projected = torch.sum(zi_groupsel, dim=2)

            # shape (batch, n_electrodes * n_phase_shifts, n_groups)
            rescaled_ti_norm_mean = (zi_norm + norms) / 2.0

            # shape (batch, n_electrodes * n_phase_shifts, n_groups)
            projected_norms = norms.detach().clone()

            projected_norms[exceeds_abs] = rescaled_ti_norm_mean[exceeds_abs]
            projected_norms[less_than_neg] = 0.0

            return zi_amplitudes_projected, projected_norms
