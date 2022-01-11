import torch
import torch.nn as nn

import numpy as np

from lib.optim.proxgrad_optim import BatchedMultiProxProblem
from lib.batch_joint_amplitude_time_opt import batched_build_at_a_matrix, batched_build_at_b_vector, \
    batched_build_unshared_at_a_matrix, batched_build_unshared_at_b_vector

from typing import Union, Optional, Tuple, Protocol
from enum import Enum

import tqdm


class RegularizationType(Enum):
    L1_SPARSE_REG = 1
    L12_GROUP_SPARSE_REG = 2


class SharedShiftSolver(Protocol):
    pass


class UnsharedShiftSolver(Protocol):
    pass


class BatchedShiftSolver(Protocol):
    def return_amplitudes(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class SharedShiftsNonNegL1ProxGradSolver(BatchedMultiProxProblem, BatchedShiftSolver, SharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_shared_at_a_matrix: np.ndarray,
                 at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l1_lambda: float,
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
        :param l1_lambda:
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

        self.l1_lambda = l1_lambda

        # shape (batch, n_valid_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_shared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_valid_phase_shifts, n_basis)
        self.register_buffer('at_b_vector', torch.tensor(at_b_vector, dtype=torch.float32))

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

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:
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

        return mse_loss_component

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_mse_loss(*args, **kwargs)

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]
            return (torch.clamp_min_(amplitudes - self.l1_lambda, 0.0),)

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            loss_unshape = self.compute_mse_loss(*self.parameters(recurse=False), **kwargs)

            mse_component = loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            # also need to add the regularization terms
            regularization_term = torch.sum(torch.abs(self.amplitudes), dim=2).reshape(self.batch_size,
                                                                                       self.n_electrodes, self.n_shifts)

            return mse_component + self.l1_lambda * regularization_term

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_clone = self.amplitudes.detach().clone()
        return amplitudes_clone.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts, self.n_basis)


class SharedShiftsGroupSparseProxGradSolver(BatchedMultiProxProblem, BatchedShiftSolver, SharedShiftSolver):
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
            torch.empty(self.batch_size, self.n_electrodes * self.n_phase_shifts, self.n_groups),
            requires_grad=True
        )
        nn.init.uniform_(self.norms, a=init_low, b=init_high)

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:

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

        return mse_loss_component

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_phase_shifts), one value for each problem,
            doesn't matter if the problem is valid or not
        '''

        # -> (batch, n_electrodes * n_phase_shifts)
        mse_loss_component = self.compute_mse_loss(*args, **kwargs)

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

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            loss_unshape = self._smooth_loss(*self.parameters(recurse=False), **kwargs)

            mse_loss = loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
            #   (1, 1, n_groups, n_basis)
            # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            grouped_components = self.amplitudes[:, :, None, :] * self.group_sel[None, None, :, :]

            # shape (batch_size, n_electrodes * phase_shifts)
            group_sparse_norm = torch.sum(torch.norm(grouped_components, p=2, dim=3), 2)

            group_sparse_penalty = group_sparse_norm.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            return mse_loss + self.l12_lambda * group_sparse_penalty

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_clone = self.amplitudes.detach().clone()
        return amplitudes_clone.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts, self.n_basis)


class UnsharedShiftsNonNegL1ProxGradSolver(BatchedMultiProxProblem, BatchedShiftSolver, UnsharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_unshared_at_a_matrix: np.ndarray,
                 batched_unshared_at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l1_lambda: float,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = -1e-1,
                 init_high: float = 1e-1):
        '''

                Conventions for the variables in this subclass

        (1) dim 0 batch dimension - different cells
        (2) dim 1 problem dimension - we combine the electrodes and phase shifts
            (this requires some reshapes to compute the loss function)


        :param batched_unshared_at_a_matrix: A^T A matrix for each problem, where each
                electrode may have different shifts for the basis vectors

            shape (batch, n_electrodes, n_phase_shifts, n_basis, n_basis)

        :param batched_unshared_at_b_vector: A^T b vector for each problem, where each
                electrode may have different shifts for the basis vectors

            shape (batch, n_electrodes, n_phase_shifts, n_basis)

        :param valid_problem_mask: shape (batch, n_electrodes)
            1-valued if the corresponding problem is valid, 0 if the corresponding problem
                is not valid and should be ignored
        :param l12_lambda:
        :param l12_group_sel_matrix: np.ndarray group assignments for the problem
            every problem is assumed to have the same groups

            value is 1 for membership in group, 0 for no membership in group

            shape (n_groups, n_basis_waveforms)
        :param amplitudes_matrix_init: np.ndarray, initial guesses for the amplitudes if available
            shape (batch, n_electrodes, n_phase_shifts, n_basis)
        :param verbose:
        :param init_low:
        :param init_high:
        '''

        batch, n_electrodes, n_shifts, n_basis = batched_unshared_at_b_vector.shape
        valid_problem_total = np.zeros((batch, n_electrodes, n_shifts), dtype=bool)
        valid_problem_total[valid_problem_mask, :] = True

        super().__init__(batch, n_electrodes * n_shifts,
                         valid_problems=valid_problem_total.reshape(batch, -1),
                         verbose=verbose)

        self.n_electrodes, self.n_shifts, self.n_basis = n_electrodes, n_shifts, n_basis

        self.l1_lambda = l1_lambda

        # shape (batch, n_electrodes, n_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_unshared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_phase_shifts, n_basis)
        self.register_buffer('at_b_matrix', torch.tensor(batched_unshared_at_b_vector, dtype=torch.float32))

        if amplitudes_matrix_init is None:
            self.amplitudes = nn.Parameter(
                torch.empty((batch, self.n_electrodes * self.n_shifts, self.n_basis)),
                requires_grad=True
            )
            nn.init.uniform_(self.amplitudes, a=init_low, b=init_high)

        else:
            self.amplitudes = nn.Parameter(
                torch.tensor(
                    amplitudes_matrix_init.reshape(batch, self.n_electrodes * self.n_shifts, n_basis),
                    dtype=torch.float32),
                requires_grad=True
            )

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_shifts)
        '''

        # shape (batch_size, n_electrodes * n_shifts, n_basis)
        amplitudes = args[self.AMPLITUDES_IDX_ARGS]

        # shape (batch, n_electrodes, n_shifts, n_basis)
        amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)

        # (batch, n_electrodes, n_shifts, n_basis, n_basis) @
        #   (batch, n_electrodes, n_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_shifts, n_basis)
        at_a_x_unshared = (self.at_a_matrix @ amplitudes_unflat[:, :, :, :, None]).squeeze(4)

        # shape (batch, n_electrodes, n_shifts)
        xt_at_a_x = torch.sum(amplitudes_unflat * at_a_x_unshared, dim=3)

        # shape (batch, n_electrodes, n_shifts)
        xt_at_b = torch.sum(amplitudes_unflat * self.at_b_matrix, dim=3)

        # shape (batch, n_electrodes, n_shifts)
        mse_loss_contrib = xt_at_b + xt_at_a_x

        # shape (batch, n_electrodes * n_shifts)
        return mse_loss_contrib.reshape(self.batch_size, -1)

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_shifts)
        '''
        return self.compute_mse_loss(*args, **kwargs)

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]
            return (torch.clamp_min_(amplitudes - self.l1_lambda, 0.0),)

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:

        with torch.no_grad():
            mse_loss_unflat = self.compute_mse_loss(*self.parameters(recurse=False), **kwargs)
            mse_loss = mse_loss_unflat.reshape(self.batch_size, self.n_electrodes, self.n_shifts)

            # also need to add the regularization terms
            regularization_term = torch.sum(torch.abs(self.amplitudes), dim=2).reshape(self.batch_size,
                                                                                       self.n_electrodes, self.n_shifts)

            return mse_loss + self.l1_lambda * regularization_term

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_copy = self.amplitudes.detach().clone()
        return amplitudes_copy.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)


class UnsharedShiftsGroupSparseProxGradSolver(BatchedMultiProxProblem, BatchedShiftSolver, UnsharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    1. norms, shape (batch, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0
    NORMS_IDX_ARGS = 1

    def __init__(self,
                 batched_unshared_at_a_matrix: np.ndarray,
                 batched_unshared_at_b_vector: np.ndarray,
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


        :param batched_unshared_at_a_matrix: A^T A matrix for each problem, where each
                electrode may have different shifts for the basis vectors

            shape (batch, n_electrodes, n_phase_shifts, n_basis, n_basis)

        :param batched_unshared_at_b_vector: A^T b vector for each problem, where each
                electrode may have different shifts for the basis vectors

            shape (batch, n_electrodes, n_phase_shifts, n_basis)

        :param valid_problem_mask: shape (batch, n_electrodes)
            1-valued if the corresponding problem is valid, 0 if the corresponding problem
                is not valid and should be ignored
        :param l12_lambda:
        :param l12_group_sel_matrix: np.ndarray group assignments for the problem
            every problem is assumed to have the same groups

            value is 1 for membership in group, 0 for no membership in group

            shape (n_groups, n_basis_waveforms)
        :param amplitudes_matrix_init: np.ndarray, initial guesses for the amplitudes if available
            shape (batch, n_electrodes, n_phase_shifts, n_basis)
        :param verbose:
        :param init_low:
        :param init_high:
        '''

        batch, n_electrodes, n_shifts, n_basis = batched_unshared_at_b_vector.shape
        valid_problem_total = np.zeros((batch, n_electrodes, n_shifts), dtype=bool)
        valid_problem_total[valid_problem_mask, :] = True

        super().__init__(batch, n_electrodes * n_shifts,
                         valid_problems=valid_problem_total.reshape(batch, -1),
                         verbose=verbose)

        self.n_electrodes, self.n_shifts, self.n_basis = n_electrodes, n_shifts, n_basis
        self.n_groups = l12_group_sel_matrix.shape[0]

        self.l12_lambda = l12_lambda

        # shape (batch, n_electrodes, n_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_unshared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_phase_shifts, n_basis)
        self.register_buffer('at_b_matrix', torch.tensor(batched_unshared_at_b_vector, dtype=torch.float32))

        # shape (n_groups, n_basis)
        self.register_buffer('group_sel', torch.tensor(l12_group_sel_matrix, dtype=torch.float32))

        if amplitudes_matrix_init is None:
            self.amplitudes = nn.Parameter(
                torch.empty((batch, self.n_electrodes * self.n_shifts, self.n_basis)),
                requires_grad=True
            )
            nn.init.uniform_(self.amplitudes, a=init_low, b=init_high)

        else:
            self.amplitudes = nn.Parameter(
                torch.tensor(
                    amplitudes_matrix_init.reshape(batch, self.n_electrodes * self.n_shifts, n_basis),
                    dtype=torch.float32),
                requires_grad=True
            )

        # aux variables for the constrained optimization
        self.norms = nn.Parameter(
            torch.empty(self.batch_size, self.n_electrodes * self.n_shifts, self.n_groups),
            requires_grad=True
        )
        nn.init.uniform_(self.norms, a=init_low, b=init_high)

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_shifts)
        '''

        # shape (batch_size, n_electrodes * n_shifts, n_basis)
        amplitudes = args[self.AMPLITUDES_IDX_ARGS]

        # shape (batch, n_electrodes * n_shifts, n_groups)
        norms = args[self.NORMS_IDX_ARGS]

        # shape (batch, n_electrodes, n_shifts, n_basis)
        amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)

        # (batch, n_electrodes, n_shifts, n_basis, n_basis) @
        #   (batch, n_electrodes, n_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_shifts, n_basis, 1)
        # -> (batch, n_electrodes, n_shifts, n_basis)
        at_a_x_unshared = (self.at_a_matrix @ amplitudes_unflat[:, :, :, :, None]).squeeze(4)

        # shape (batch, n_electrodes, n_shifts)
        xt_at_a_x = torch.sum(amplitudes_unflat * at_a_x_unshared, dim=3)

        # shape (batch, n_electrodes, n_shifts)
        xt_at_b = torch.sum(amplitudes_unflat * self.at_b_matrix, dim=3)

        # shape (batch, n_electrodes, n_shifts)
        mse_loss_contrib = xt_at_b + xt_at_a_x

        # shape (batch, n_electrodes * n_shifts)
        return mse_loss_contrib.reshape(self.batch_size, -1)

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, n_electrodes * n_shifts, n_groups)
        norms = args[self.NORMS_IDX_ARGS]

        # shape (batch, n_electrodes, n_shifts)
        mse_loss_contrib = self.compute_mse_loss(*args, **kwargs)

        # shape (batch, n_electrodes, n_shifts)
        total_loss = mse_loss_contrib + self.l12_lambda * torch.sum(norms, dim=2)

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

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:

        mse_loss_unshape = self.compute_mse_loss(self.parameters(recurse=False), **kwargs)

        mse_loss = mse_loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_shifts)

        # now we need to compute the group norms
        # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
        #   (1, 1, n_groups, n_basis)
        # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
        grouped_components = self.amplitudes[:, :, None, :] * self.group_sel[None, None, :, :]

        # shape (batch_size, n_electrodes * phase_shifts)
        group_sparse_norm = torch.sum(torch.norm(grouped_components, p=2, dim=3), 2)

        group_sparse_penalty = group_sparse_norm.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

        return mse_loss + self.l12_lambda * group_sparse_penalty

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_copy = self.amplitudes.detach().clone()
        return amplitudes_copy.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)


def make_shared_shift_solver(regularization_type: RegularizationType,
                             batch_shared_at_a_matrix: np.ndarray,
                             batch_at_b_vector: np.ndarray,
                             valid_problem_mask: np.ndarray,
                             reg_lambda: float,
                             group_sel_matrix: Optional[np.ndarray] = None,
                             amplitudes_matrix_init: Optional[np.ndarray] = None,
                             verbose: bool = True,
                             init_low: float = -1e-2,
                             init_high: float = 1e-2) \
        -> Union[SharedShiftSolver, BatchedMultiProxProblem]:
    if regularization_type == RegularizationType.L1_SPARSE_REG:

        return SharedShiftsNonNegL1ProxGradSolver(
            batch_shared_at_a_matrix,
            batch_at_b_vector,
            valid_problem_mask,
            reg_lambda,
            amplitudes_matrix_init=amplitudes_matrix_init,
            verbose=verbose,
            init_low=init_low,
            init_high=init_high
        )

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG:
        if group_sel_matrix is None:
            raise ValueError(f"group_sel_matrix cannot be none if specifying L12 group sparsity")

        return SharedShiftsGroupSparseProxGradSolver(
            batch_shared_at_a_matrix,
            batch_at_b_vector,
            valid_problem_mask,
            reg_lambda,
            group_sel_matrix,
            amplitudes_matrix_init=amplitudes_matrix_init,
            verbose=verbose,
            init_low=init_low,
            init_high=init_high
        )

    raise ValueError(f"Invalid regularization type {regularization_type}")


def batched_fast_time_shifts_and_amplitudes_shared_shifts2(
        observed_ft: np.ndarray,
        ft_basis: np.ndarray,
        valid_phase_shifts: np.ndarray,
        regularization_lambda: float,
        regularization_type: RegularizationType,
        n_true_frequencies: int,
        max_iter: int,
        device: torch.device,
        amplitude_matrix_real_np: Optional[np.ndarray] = None,
        converge_epsilon: float = 1e-2,
        group_sel_matrix: Optional[np.ndarray] = None,
        valid_problems: Optional[np.ndarray] = None,
        solver_verbose: bool = True,
        init_low: float = 1e-2,
        init_high: float = 1e-2) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Iterative coarse-to-fine search helper function, for the case where each individual cell
        has its own set of basis waveforms

    For fixed phase shifts, as defined by valid_phase_shifts, this functio solves the nonnegative
        least squares minimization problem with optional L1 or L12-group-sparse regularization, and
        returns all solutions and objective function values.

        Notation for the below function

        objective fn is 1/2 |Ax-b|^2 = 1/2 (Ax-b)^T (Ax-b) = 1/2 x^T A^T A x - x^T A^T b - 1/2 b^T b
        gradient is A^T A x - A^T b

    Implementation notes:

    (A^T A)_{i,j}^{(z)} corresponds to the cross-correlation of the i^{th} canonical waveform, delayed by
        valid_phase_shifts[i,z] number of samples, with the j^{th} canonical waveform, delayed by
        valid_phase_shifts[j,z] number of samples


    :param observed_ft: data waveforms in Fourier domain, complex-valued,
        shape (batch, n_electrodes, n_rfft_frequencies)
    :param ft_basis: Fourier transform of the basis waveforms, unshifted,complex-valued,
        shape (batch, n_basis_waveforms, n_rfft_frequencies)

        Note that each cell (batch dimension) has its own basis waveforms
    :param valid_phase_shifts: integer array, shape (batch, n_basis_waveforms, n_phase_shifts)
    :param n_true_frequencies: int
    :param max_iter: int, maximum number of iterations
    :param device:
    :param amplitude_matrix_real_np: shape (batch, n_electrodes, n_phase_shifts, n_basis_waveforms)
        Initial guess for the amplitudes, optional
    :param converge_epsilon:
    :param valid_problems: Optional boolean np.ndarray, shape (batch, n_electrodes)

        If not specified, we assume that all of the problems are valid. If specified,
            the solver terminates upon convergence of the valid problems only.
    :return:

        Amplitudes and objective function value for all of the problems

        (0) amplitudes, shape (batch, n_electrodes, n_phase_shifts, n_basis_vectors)
        (1) objective_valeus, shape (batch, n_electrodes, n_phase_shifts)
    '''

    batch, n_observations, n_rfft_frequencies = observed_ft.shape
    _, n_basis, n_phase_shifts = valid_phase_shifts.shape

    #### Step 1: build A^T A from circular cross correlation #####################################

    # shape (batch, n_valid_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    at_a_matrix = batched_build_at_a_matrix(ft_basis, valid_phase_shifts, n_true_frequencies)

    ##### Step 2: build A^T b from circular cross correlation with data matrix ##################
    # this one depends on absolute timing so it is much easier to pack
    # shape (batch, n_observations, n_valid_phase_shifts, n_canonical_waveforms)
    at_b_vector = batched_build_at_b_vector(observed_ft, ft_basis, valid_phase_shifts, n_true_frequencies)

    ##### Step 3: do the projected gradient descent  #############################################

    solver = make_shared_shift_solver(regularization_type,
                                      at_a_matrix,
                                      at_b_vector,
                                      valid_problems,
                                      regularization_lambda,
                                      group_sel_matrix=group_sel_matrix,
                                      amplitudes_matrix_init=amplitude_matrix_real_np,
                                      verbose=solver_verbose,
                                      init_low=init_low,
                                      init_high=init_high).to(
        device)  # type: Union[BatchedMultiProxProblem, SharedShiftSolver, BatchedShiftSolver]

    _ = solver.fista_prox_solve(1.0, max_iter, converge_epsilon,
                                0.5)  # FIXME get the solver parameters from somewhere

    # shape (batch, n_electrodes, n_phase_shifts, n_basis)
    amplitudes_solved = solver.return_amplitudes()

    # shape (batch, n_electrodes, n_phase_shifts)
    objective_fn_vals = solver.compute_loss_for_argmin()

    return amplitudes_solved.detach().cpu().numpy(), objective_fn_vals.detach().cpu().numpy()


def make_unshared_shifts_solver(
        regularization_type: RegularizationType,
        batch_unshared_at_a_matrix: np.ndarray,
        batch_at_b_vector: np.ndarray,
        valid_problem_mask: np.ndarray,
        reg_lambda: float,
        group_sel_matrix: Optional[np.ndarray] = None,
        amplitudes_matrix_init: Optional[np.ndarray] = None,
        verbose: bool = True,
        init_low: float = -1e-2,
        init_high: float = 1e-2) \
        -> Union[UnsharedShiftSolver, BatchedMultiProxProblem]:
    if regularization_type == RegularizationType.L1_SPARSE_REG:
        return UnsharedShiftsNonNegL1ProxGradSolver(
            batch_unshared_at_a_matrix,
            batch_at_b_vector,
            valid_problem_mask,
            reg_lambda,
            amplitudes_matrix_init=amplitudes_matrix_init,
            verbose=verbose,
            init_low=init_low,
            init_high=init_high
        )

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG:
        if group_sel_matrix is None:
            raise ValueError(f"group_sel_matrix cannot be None")

        return UnsharedShiftsGroupSparseProxGradSolver(
            batch_unshared_at_a_matrix,
            batch_at_b_vector,
            valid_problem_mask,
            reg_lambda,
            group_sel_matrix,
            amplitudes_matrix_init=amplitudes_matrix_init,
            verbose=verbose,
            init_low=init_low,
            init_high=init_high
        )

    else:
        raise ValueError(f"Invalid regularization type {regularization_type}")


def batched_fast_time_shifts_and_amplitudes_unshared_shifts2(
        observed_ft: np.ndarray,
        ft_basis: np.ndarray,
        unshared_phase_shifts: np.ndarray,
        regularization_lambda: float,
        regularization_type: RegularizationType,
        n_true_frequencies: int,
        max_iter: int,
        device: torch.device,
        converge_epsilon: float = 1e-2,  # FIXME
        amplitude_matrix_real_np: Optional[np.ndarray] = None,
        group_sel_matrix: Optional[np.ndarray] = None,
        valid_problems: Optional[np.ndarray] = None,
        solver_verbose: bool = True,
        init_low: float = 1e-2,
        init_high: float = 1e-2) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    Parallel implementation of the second step fine search, where for a collection of different optimization problems,
        we consider a separate set of shifts for each problem (i.e. for problems A and B, the basis vectors may be
        shifted different amounts for problem A and for problem B)


    :param observed_ft: observed waveforms in Fourier domain, complex valued,
            shape (batch, n_observations, n_rfft_frequencies)
    :param ft_basis: basis waveforms in Fourier domain, unshifted, complex valued,
            shape (batch, n_canonical_waveforms, n_rfft_frequencies)
    :param unshared_phase_shifts: integer-valued
            shape (batch, n_observations, n_canonical_waveforms, n_phase_shifts)
    :param regularization_lambda:
    :param regularization_type:
    :param n_true_frequencies: int, number of full FFT frequencies (not rFFT frequencies)
    :param max_iter:
    :param device:
    :param converge_epsilon:
    :param amplitude_matrix_real_np:
    :param group_sel_matrix:
    :param valid_problems:
    :param solver_verbose:
    :param init_low:
    :param init_high:
    :return:
    '''

    batch, n_observations, n_rfft_frequencies = observed_ft.shape

    ##### Generate the appropriate A^T A matrices and A^T b vectors ###########################################

    # shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms, n_canonical_waveforms)
    unshared_at_a_matrix_np = batched_build_unshared_at_a_matrix(ft_basis, unshared_phase_shifts,
                                                                 n_true_frequencies)

    # shape (batch, n_observations, n_phase_shifts, n_canonical_waveforms)
    unshared_at_b_vector_np = batched_build_unshared_at_b_vector(observed_ft, ft_basis,
                                                                 unshared_phase_shifts, n_true_frequencies)

    ##### Step 3: do the projected gradient descent  #############################################

    solver = make_unshared_shifts_solver(regularization_type,
                                         unshared_at_a_matrix_np,
                                         unshared_at_b_vector_np,
                                         valid_problems,
                                         regularization_lambda,
                                         group_sel_matrix=group_sel_matrix,
                                         amplitudes_matrix_init=amplitude_matrix_real_np,
                                         verbose=solver_verbose,
                                         init_low=init_low,
                                         init_high=init_high).to(
        device)  # type: Union[BatchedMultiProxProblem, UnsharedShiftSolver, BatchedShiftSolver]

    _ = solver.fista_prox_solve(1.0, max_iter, converge_epsilon,
                                0.5)  # FIXME get the solver parameters from somewhere

    # shape (batch, n_electrodes, n_phase_shifts, n_basis)
    amplitudes_solved = solver.return_amplitudes()

    # shape (batch, n_electrodes, n_phase_shifts)
    objective_fn_vals = solver.compute_loss_for_argmin()

    return amplitudes_solved.detach().cpu().numpy(), objective_fn_vals.detach().cpu().numpy()


def batched_coarse_to_fine_time_shifts_and_amplitudes2(
        observed_ft: np.ndarray,
        ft_basis: np.ndarray,
        n_true_frequencies,
        regularization_type: RegularizationType,
        regularization_lambda: float,
        valid_phase_shift_range: Tuple[int, int],
        first_pass_step_size: int,
        second_pass_best_n: int,
        second_pass_width: int,
        device: torch.device,
        group_sel_matrix: Optional[np.ndarray] = None,
        converge_epsilon: float = 1e-2,
        valid_problems: Optional[np.ndarray] = None,
        amplitude_initialize_range: Tuple[float, float] = (0.0, 10.0),
        max_batch_size: int = 1024,
        cell_batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Coarse-to-fine joint fitting of amplitudes and time shifts. Algorithm has two distinct phases:

    Phase (1): Perform a coarse grid search over all possible time shifts for each waveform, solving the nonnegative
        least squares problem for each point on the grid, and taking the second_pass_best_n best grid points
    Phase (2): For each of the second_pass_best_n best grid points, perform a fine search with width second_pass_width
        centered on those best grid points

    Returns the best amplitude, shift

    :param observed_ft: observed waveforms in Fourier domain, complex valued,
        shape (batch, n_electrodes, n_rfft_frequencies)
        Potentially scaled
    :param ft_basis: basis waveforms in Fourier domain, unshifted, complex valued,
            shape (batch, n_basis, n_rfft_frequencies)
    :param n_true_frequencies: int, number of actual (not rFFT) FFT frequencies
    :param regularization_type:
    :param regularization_lambda:
    :param valid_phase_shift_range: range of sample shifts to consider
    :param first_pass_step_size: int, step size for the first pass grid search
    :param second_pass_best_n: int, algorithm expands upon the second_pass_best_n best results from the first pass grid
        search to do the second pass grid search
    :param second_pass_width: int, one-sided width around the second_pass_best_n best results from the first pass grid
        search that the algorithm searches for the second pass
    :param device:
    :param converge_epsilon:
    :param converge_step_cutoff:
    :param valid_problems: shape (batch, n_electrodes) boolean valued, True if the corresponding problem
        is valid and needs to be solved, and False if the corresponding problem is invalid and doesn't need
        to be solved
    :param amplitude_initialize_range:
    :param max_batch_size: int, maximum batch size to solve at once
    :param cell_batch_size: int, maximum batch size to solve at once
    :return: amplitudes and shifts,
        each with shape (batch, n_electrodes, n_basis)
    '''

    batch, n_electrodes, _ = observed_ft.shape
    _, n_basis, n_rfft_frequencies = ft_basis.shape

    ######### Step 1: first pass, perform nonnegative least squares minimization on a coarse ###############
    ######## grid of phase shifts, then pick the N best ####################################################
    low_shift, high_shift = valid_phase_shift_range
    shift_steps = np.r_[low_shift:high_shift:first_pass_step_size]
    mg = np.stack(np.meshgrid(*[shift_steps for _ in range(n_basis)]), axis=0)

    # shape (n_basis, (high_shift - low_shift)^n_basis)
    valid_phase_shifts_matrix = mg.reshape((n_basis, -1))

    _, n_valid_phase_shifts = valid_phase_shifts_matrix.shape

    amplitude_results = np.zeros((batch, n_electrodes, n_valid_phase_shifts, n_basis),
                                 dtype=np.float32)
    objective_results = np.zeros((batch, n_electrodes, n_valid_phase_shifts), dtype=np.float32)

    pbar = tqdm.tqdm(total=int(np.ceil(n_valid_phase_shifts / max_batch_size)), leave=False, desc='First pass grid search')
    for low in range(0, n_valid_phase_shifts, max_batch_size):
        high = min(n_valid_phase_shifts, low + max_batch_size)

        amplitudes_random_init = np.random.uniform(amplitude_initialize_range[0],
                                                   amplitude_initialize_range[1],
                                                   size=(batch, n_electrodes, high - low, n_basis))

        amplitude_batch, objective_batch = batched_fast_time_shifts_and_amplitudes_shared_shifts2(
            observed_ft,
            ft_basis,
            valid_phase_shifts_matrix[None, :, low:high],
            regularization_lambda,
            regularization_type,
            n_true_frequencies,
            10000,
            device,
            group_sel_matrix=group_sel_matrix,
            converge_epsilon=converge_epsilon,
            valid_problems=valid_problems,
            amplitude_matrix_real_np=amplitudes_random_init,
            solver_verbose=False,
        )

        amplitude_results[:, :, low:high, :] = amplitude_batch
        objective_results[:, :, low:high] = objective_batch
        pbar.update(1)
    pbar.close()

    # pick the N best nodes to expand in detail
    # shape (batch, n_electrodes, second_pass_best_n)
    partition_idx = np.argpartition(objective_results, second_pass_best_n, axis=2)[:, :, :second_pass_best_n]

    # shape (batch * n_electrodes * second_pass_best_n)
    partition_idx_flat = partition_idx.reshape((-1))

    # shape (n_basis, batch * n_electrodes * second_pass_best)
    best_phases_flat = valid_phase_shifts_matrix[:, partition_idx_flat]

    # shape (batch, n_electrodes, n_basis, second_pass_best_n)
    best_phases = best_phases_flat.reshape(n_basis, batch, n_electrodes, second_pass_best_n).transpose(1, 2, 0, 3)

    #### Do the fine search ###################################################

    second_pass_fine_steps = np.r_[-second_pass_width:second_pass_width + 1]

    # define n_fine_phases = (1 + 2 * second_pass_width)^n_basis
    # shape (n_basis, n_fine_phases)
    second_pass_mg = np.stack(np.meshgrid(*[second_pass_fine_steps for _ in range(n_basis)]), axis=0)
    second_pass_mg_flat = second_pass_mg.reshape((second_pass_mg.shape[0], -1))

    # shape (batch, n_electrodes, n_basis, 1, second_pass_best_n)
    # + shape (1, 1, n_electrodes, n_fine_phases, 1)
    # -> (batch, n_electrodes, n_basis, n_fine_phases, second_pass_best_n)
    next_iter_phases = best_phases[:, :, :, None, :] + second_pass_mg_flat[None, None, :, :, None]

    # shape (batch, n_electrodes, n_basis, second_pass_best_n * n_fine_phases)
    next_iter_phases_flat = next_iter_phases.reshape((batch, n_electrodes, n_basis, -1))
    n_second_pass_shifts = next_iter_phases_flat.shape[3]

    amplitude_results = np.zeros((batch, n_electrodes, n_second_pass_shifts, n_basis),
                                 dtype=np.float32)
    objective_results = np.zeros((batch, n_electrodes, n_second_pass_shifts), dtype=np.float32)

    pbar = tqdm.tqdm(total=int(np.ceil(n_second_pass_shifts / max_batch_size)), leave=False, desc='Second pass fine search')
    for low in range(0, n_second_pass_shifts, max_batch_size):
        high = min(n_second_pass_shifts, low + max_batch_size)

        amplitudes_random_init = np.random.uniform(amplitude_initialize_range[0],
                                                   amplitude_initialize_range[1],
                                                   size=(batch, n_electrodes, high - low, n_basis))

        amplitude_batch, objective_batch = batched_fast_time_shifts_and_amplitudes_unshared_shifts2(
            observed_ft,
            ft_basis,
            next_iter_phases_flat[:, :, :, low:high],
            regularization_lambda,
            regularization_type,
            n_true_frequencies,
            10000,
            device,
            group_sel_matrix=group_sel_matrix,
            amplitude_matrix_real_np=amplitudes_random_init,
            converge_epsilon=converge_epsilon,
            valid_problems=valid_problems,
            solver_verbose=False
        )

        amplitude_results[:, :, low:high, :] = amplitude_batch
        objective_results[:, :, low:high] = objective_batch
        pbar.update(1)
    pbar.close()

    #### Now pick the best objective value for each observed waveform ########################
    best_objective = np.argmin(objective_results, axis=2)  # shape (batch, n_observations)

    best_amplitudes = np.take_along_axis(amplitude_results, best_objective[:, :, None, None], axis=2).squeeze(2)
    best_shifts = np.take_along_axis(next_iter_phases_flat, best_objective[:, :, None, None], axis=3).squeeze(3)

    return best_amplitudes, best_shifts
