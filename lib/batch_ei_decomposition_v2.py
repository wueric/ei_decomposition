import torch
import torch.nn as nn

import numpy as np

from sklearn.gaussian_process.kernels import RBF

from lib.frequency_domain_optimization import batch_fourier_complex_least_square_optimize3, \
    batch_fourier_complex_least_square_with_prior_optimize, construct_rfft_covariance_matrix, _pack_complex_to_real_imag
from lib.losseval import batch_evaluate_mse_flat
from lib.optim.optim_base import BatchedMultiProxProblem, \
    ManualGradBatchMultiProxProblem, ProxSolverParams, ProxFISTASolverParams, ProxFixedStepSizeSolverParams, \
    ProxGradSolverParams
from lib.optim.prox_optim import batch_multiproblem_parallel_prox_solve
from lib.batch_joint_amplitude_time_opt import batched_build_at_a_matrix, batched_build_at_b_vector, \
    batched_build_unshared_at_a_matrix, batched_build_unshared_at_b_vector

from typing import Union, Optional, Tuple, Dict, List, Any
from enum import Enum
from dataclasses import dataclass

import tqdm

from lib.util_fns import auto_unbatch_unpack_significant_electrodes, auto_prebatch_pack_significant_electrodes, \
    bspline_upsample_waveforms, shift_waveform_peaks_and_adjust_shifts


class RegularizationType(Enum):
    L1_SPARSE_REG = 1
    L12_GROUP_SPARSE_REG_CONSTRAINED = 2
    L12_GROUP_SPARSE_REG_SMOOTH = 3


class SharedShiftSolver:
    pass


class UnsharedShiftSolver:
    pass


class BatchedShiftSolver:
    def return_amplitudes(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_mse_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class SharedShiftsNonNegL1ProxGradSolver(ManualGradBatchMultiProxProblem,
                                         BatchedShiftSolver,
                                         SharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_shared_at_a_matrix: np.ndarray,
                 at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l1_lambda: Union[float, np.ndarray],
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = 0.0,
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

        self.lambda_is_tensor = isinstance(l1_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l1_lambda.shape != (temp_batch, n_electrodes, n_phase_shifts):
                raise ValueError(f"l1_lambda must have shape {(temp_batch, n_electrodes, n_phase_shifts)}")

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l1_lambda', torch.tensor(l1_lambda.reshape(temp_batch, -1), dtype=torch.float32))
        else:
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

    def compute_fixed_step_size(self) -> torch.Tensor:

        with torch.no_grad():
            eigenvalues, _ = torch.linalg.eigh(self.at_a_matrix)
            max_eigenvalues = eigenvalues[:, :, -1]

            step_size = 1.0 / (2 * max_eigenvalues)

            step_size_repeated = step_size.repeat(1, self.n_electrodes)
            return step_size_repeated

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
        mse_loss_component = (0.5 * xt_at_a_x - xt_at_b).reshape(self.batch_size, -1)
        return mse_loss_component

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param packed_variables: shape (batch, n_electrodes * n_phase_shifts, n_basis + n_groups)
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = packed_variables  # there's only 1 variable

            # shape (batch_size, n_electrodes, n_phase_shifts, n_basis)
            amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts,
                                                   self.n_basis)

            # (batch, 1, n_valid_phase_shifts, n_basis, n_basis) @
            #       (batch, n_electrodes, n_phase_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_phase_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_phase_shifts, n_basis)
            at_a_x = (self.at_a_matrix[:, None, :, :, :] @ amplitudes_unflat[:, :, :, :, None]).squeeze(-1)

            gradient_unflat = at_a_x - self.at_b_vector
            gradient_flat = gradient_unflat.reshape(self.batch_size, -1, self.n_basis)

            # shape (batch, n_electrodes, n_phase_shifts)
            xt_at_a_x = torch.sum(amplitudes_unflat * at_a_x, dim=3)

            # shape (batch, n_electrodes, n_phase_shifts)
            xt_at_b = torch.sum(self.at_b_vector * amplitudes_unflat, dim=3)

            # shape (batch, n_electrodes, n_phase_shifts)
            # -> (batch, n_electrodes * n_phase_shifts)
            mse_loss_component = (0.5 * xt_at_a_x - xt_at_b).reshape(self.batch_size, -1)

            return mse_loss_component, gradient_flat

    def _manual_gradients(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]

            # shape (batch_size, n_electrodes, n_phase_shifts, n_basis)
            amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts,
                                                   self.n_basis)

            # (batch, 1, n_valid_phase_shifts, n_basis, n_basis) @
            #       (batch, n_electrodes, n_phase_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_phase_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_phase_shifts, n_basis)
            at_a_x = (self.at_a_matrix[:, None, :, :, :] @ amplitudes_unflat[:, :, :, :, None]).squeeze(-1)

            gradient_unflat = at_a_x - self.at_b_vector

            gradient_flat = gradient_unflat.reshape(self.batch_size, -1, self.n_basis)

            return (gradient_flat,)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
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

            if self.lambda_is_tensor:
                return (torch.clamp(amplitudes - self.l1_lambda[None, :, :], min=0.0),)
            return (torch.clamp(amplitudes - self.l1_lambda, min=0.0),)

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            loss_unshape = self.compute_mse_loss(*self.parameters(recurse=False), **kwargs)

            mse_component = loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            # also need to add the regularization terms
            regularization_term = torch.sum(torch.abs(self.amplitudes), dim=2) * self.l1_lambda
            regularization_term = regularization_term.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            return mse_component + regularization_term

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_clone = self.amplitudes.detach().clone()
        return amplitudes_clone.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts, self.n_basis)


class SharedShiftsNonNegOrthantGroupSparseProxGradSolver(BatchedMultiProxProblem,
                                                         BatchedShiftSolver,
                                                         SharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    
    IMPORTANT: THERE IS A CONCEPTUAL PROBLEM FOR THE NAIVE IMPLEMENTATION OF THE SMOOTH L12 GROUP SPARSITY NORM!!!
    The gradient of the norm penalty involves a root of a norm in the denominator; in the case that the
    the components of the vector that goes into that norm are 0 (which is very possible since we want
    sparsity), then the gradient contains a divide-by-0 and we have problems.

    To deal with this problem, we add a small epsilon value to each norm inside of the square root.
    '''

    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_shared_at_a_matrix: np.ndarray,
                 at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l12_lambda: Union[float, np.ndarray],
                 l12_group_sel_matrix: np.ndarray,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = 0.0,
                 init_high: float = 1e-1,
                 epsilon_div_by_zero: float = 1e-6):
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

        self.lambda_is_tensor = isinstance(l12_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l12_lambda.shape != (temp_batch, n_electrodes, n_phase_shifts):
                raise ValueError(f"l12_lambda must have shape {(temp_batch, n_electrodes, n_phase_shifts)}")

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l12_lambda', torch.tensor(l12_lambda.reshape(temp_batch, -1), dtype=torch.float32))
        else:
            self.l12_lambda = l12_lambda

        self.epsilon_div_by_zero = epsilon_div_by_zero

        # shape (batch, n_valid_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_shared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_valid_phase_shifts, n_basis)
        self.register_buffer('at_b_vector', torch.tensor(at_b_vector, dtype=torch.float32))

        # shape (n_groups, n_basis)
        self.register_buffer('group_sel', torch.tensor(l12_group_sel_matrix, dtype=torch.float32))

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

        self.check_variables()

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
        mse_loss_component = (0.5 * xt_at_a_x - xt_at_b).reshape(self.batch_size, -1)

        return mse_loss_component

    def compute_group_sparse_term(self, *args, **kwargs) -> torch.Tensor:

        # (batch_size, n_electrodes * n_phase_shifts, n_basis)
        amplitudes = args[self.AMPLITUDES_IDX_ARGS]

        # we need a smarter way to compute the norm without blowing up the amount of memory
        # since that produces a huge tensor, then throws it away...
        # (batch_size, n_electrodes * n_phase_shifts, n_basis)
        square_amplitudes_perm = torch.square(amplitudes)

        # shape (batch_size, n_electrodes * n_phase_shifts, n_basis) @ (1, n_basis, n_groups)
        # -> (batch_size, n_electrodes * n_phase_shifts, n_groups)
        grouped_square_norms = square_amplitudes_perm @ (self.group_sel.T)[None, :, :]

        # (batch_size, n_electrodes * n_phase_shifts, n_groups)
        # -> (batch_size, n_electrodes * n_phase_shifts)
        group_sparse_norm = torch.sum(torch.sqrt(grouped_square_norms + self.epsilon_div_by_zero), dim=2)

        # shape (batch_size, n_electrodes * phase_shifts)
        return self.l12_lambda * group_sparse_norm

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_phase_shifts), one value for each problem,
            doesn't matter if the problem is valid or not
        '''

        # -> (batch, n_electrodes * n_phase_shifts)
        mse_loss_component = self.compute_mse_loss(*args, **kwargs)

        # -> (batch, n_electrodes * n_phase_shifts)
        group_sparse_component = self.compute_group_sparse_term(*args, **kwargs)

        # -> (batch, n_electrodes * n_phase_shifts)
        return mse_loss_component + group_sparse_component

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]
            return (torch.clamp_min_(amplitudes, 0.0),)

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            loss_unshape = self._eval_smooth_loss(*self.parameters(recurse=False), **kwargs)
            total_loss = loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)
            return total_loss

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
                 l12_lambda: Union[float, np.ndarray],
                 l12_group_sel_matrix: np.ndarray,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = 0,
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

        self.lambda_is_tensor = isinstance(l12_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l12_lambda.shape != (temp_batch, n_electrodes):
                raise ValueError(f"l12_lambda must have shape {(temp_batch, n_electrodes)}, got {l12_lambda.shape}")

            l12_lambda = np.tile(l12_lambda[:, :, None], (1, 1, n_phase_shifts))

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l12_lambda', torch.tensor(l12_lambda.reshape(temp_batch, -1), dtype=torch.float32))
        else:
            self.l12_lambda = l12_lambda

        # shape (batch, n_valid_phase_shifts, n_basis, n_basis)
        self.register_buffer('at_a_matrix', torch.tensor(batched_shared_at_a_matrix, dtype=torch.float32))

        # shape (batch, n_electrodes, n_valid_phase_shifts, n_basis)
        self.register_buffer('at_b_vector', torch.tensor(at_b_vector, dtype=torch.float32))

        # shape (n_groups, n_basis)
        self.register_buffer('group_sel', torch.tensor(l12_group_sel_matrix, dtype=torch.float32))

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

        self.check_variables()

    def compute_fixed_step_size(self) -> torch.Tensor:
        with torch.no_grad():
            eigenvalues, _ = torch.linalg.eigh(self.at_a_matrix)
            max_eigenvalues = eigenvalues[:, :, -1]

            step_size = 1.0 / (2 * max_eigenvalues)

            step_size_repeated = step_size.repeat(1, self.n_electrodes)
            return step_size_repeated

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
        mse_loss_component = (0.5 * xt_at_a_x - xt_at_b).reshape(self.batch_size, -1)

        return mse_loss_component

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
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
            zi_norm = torch.linalg.norm(zi_groupsel, dim=3, ord=2)

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
            loss_unshape = self._eval_smooth_loss(*self.parameters(recurse=False), **kwargs)

            mse_loss = loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
            #   (1, 1, n_groups, n_basis)
            # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            grouped_components = self.amplitudes[:, :, None, :] * self.group_sel[None, None, :, :]

            # shape (batch_size, n_electrodes * phase_shifts)
            group_sparse_norm = torch.sum(torch.linalg.norm(grouped_components, ord=2, dim=3), dim=2) * self.l12_lambda

            group_sparse_penalty = group_sparse_norm.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts)

            return mse_loss + group_sparse_penalty

    def return_amplitudes(self) -> torch.Tensor:
        amplitudes_clone = self.amplitudes.detach().clone()
        return amplitudes_clone.reshape(self.batch_size, self.n_electrodes, self.n_phase_shifts, self.n_basis)


class UnsharedShiftsNonNegL1ProxGradSolver(ManualGradBatchMultiProxProblem, BatchedShiftSolver, UnsharedShiftSolver):
    '''
    Variable order convention (in order of registration)

    0. amplitudes, shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
    '''

    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_unshared_at_a_matrix: np.ndarray,
                 batched_unshared_at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l1_lambda: Union[float, np.ndarray],
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
        :param l1_lambda: either a constant, or np.ndarray with shape
            (batch, n_electrodes, n_phase_shifts)
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

        self.lambda_is_tensor = isinstance(l1_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l1_lambda.shape != (batch, n_electrodes, n_shifts):
                raise ValueError(f"l1_lambda must have shape {(batch, n_electrodes, n_shifts)}")

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l1_lambda', torch.tensor(l1_lambda.reshape(batch, -1), dtype=torch.float32))
        else:
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
        mse_loss_contrib = 0.5 * xt_at_a_x - xt_at_b

        # shape (batch, n_electrodes * n_shifts)
        return mse_loss_contrib.reshape(self.batch_size, -1)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return: shape (batch, n_electrodes * n_shifts)
        '''
        return self.compute_mse_loss(*args, **kwargs)

    def _manual_gradients(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]

            # shape (batch, n_electrodes, n_shifts, n_basis)
            amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)

            # (batch, n_electrodes, n_shifts, n_basis, n_basis) @
            #   (batch, n_electrodes, n_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_shifts, n_basis)
            at_a_x_unshared = (self.at_a_matrix @ amplitudes_unflat[:, :, :, :, None]).squeeze(4)

            # -> (batch, n_electrodes, n_shifts, n_basis)
            gradient_unflat = at_a_x_unshared - self.at_b_matrix

            gradient_flat = gradient_unflat.reshape(self.batch_size, -1, self.n_basis)

            return (gradient_flat,)

    def _packed_loss_and_gradients(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]

            # shape (batch, n_electrodes, n_shifts, n_basis)
            amplitudes_unflat = amplitudes.reshape(self.batch_size, self.n_electrodes, self.n_shifts, self.n_basis)

            # (batch, n_electrodes, n_shifts, n_basis, n_basis) @
            #   (batch, n_electrodes, n_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_shifts, n_basis, 1)
            # -> (batch, n_electrodes, n_shifts, n_basis)
            at_a_x_unshared = (self.at_a_matrix @ amplitudes_unflat[:, :, :, :, None]).squeeze(4)

            # -> (batch, n_electrodes, n_shifts, n_basis)
            gradient_unflat = at_a_x_unshared - self.at_b_matrix

            gradient_flat = gradient_unflat.reshape(self.batch_size, -1, self.n_basis)

            # shape (batch, n_electrodes, n_shifts)
            xt_at_a_x = torch.sum(amplitudes_unflat * at_a_x_unshared, dim=3)

            # shape (batch, n_electrodes, n_shifts)
            xt_at_b = torch.sum(amplitudes_unflat * self.at_b_matrix, dim=3)

            # shape (batch, n_electrodes, n_shifts)
            mse_loss_contrib = 0.5 * xt_at_a_x - xt_at_b

            # shape (batch, n_electrodes * n_shifts)
            mse_loss_flat = mse_loss_contrib.reshape(self.batch_size, -1)

            return mse_loss_flat, gradient_flat

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            # shape (batch_size, n_electrodes * n_phase_shifts, n_basis)
            amplitudes = args[self.AMPLITUDES_IDX_ARGS]

            if self.lambda_is_tensor:
                return (torch.clamp_min_(amplitudes - self.l1_lambda[:, :, None], 0.0),)
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


class UnsharedShiftsNonNegOrthantGroupSparseProxGradSolver(BatchedMultiProxProblem,
                                                           BatchedShiftSolver,
                                                           UnsharedShiftSolver):
    '''
    IMPORTANT: THERE IS A CONCEPTUAL PROBLEM FOR THE NAIVE IMPLEMENTATION OF THE SMOOTH L12 GROUP SPARSITY NORM!!!
    The gradient of the norm penalty involves a root of a norm in the denominator; in the case that the
    the components of the vector that goes into that norm are 0 (which is very possible since we want
    sparsity), then the gradient contains a divide-by-0 and we have problems.

    We add a small epsilon inside the square root of the penalty to deal with this problem

    '''
    AMPLITUDES_IDX_ARGS = 0

    def __init__(self,
                 batched_unshared_at_a_matrix: np.ndarray,
                 batched_unshared_at_b_vector: np.ndarray,
                 valid_problem_mask: np.ndarray,
                 l12_lambda: Union[float, np.ndarray],
                 l12_group_sel_matrix: np.ndarray,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = -1e-1,
                 init_high: float = 1e-1,
                 epsilon_avoid_div_by_zero: float = 1e-6):

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
        :param l12_lambda: either a float constant, or np.ndarray with shape
                (batch, n_electrodes, n_phase_shifts)
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

        self.lambda_is_tensor = isinstance(l12_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l12_lambda.shape != (batch, n_electrodes, n_shifts):
                raise ValueError(f"l12_lambda must have shape {(batch, n_electrodes, n_shifts)}")

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l12_lambda', torch.tensor(l12_lambda.reshape(batch, -1), dtype=torch.float32))
        else:
            self.l12_lambda = l12_lambda

        self.epsilon_avoid_div_by_zero = epsilon_avoid_div_by_zero

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
        mse_loss_contrib = 0.5 * xt_at_a_x - xt_at_b

        # shape (batch, n_electrodes * n_shifts)
        return mse_loss_contrib.reshape(self.batch_size, -1)

    def compute_group_sparse_term(self, *args, **kwargs) -> torch.Tensor:

        # (batch_size, n_electrodes * n_phase_shifts, n_basis)
        amplitudes = args[self.AMPLITUDES_IDX_ARGS]

        # we need a smarter way to compute the norm without blowing up the amount of memory
        # since that produces a huge tensor, then throws it away...
        # (batch_size, n_electrodes * n_phase_shifts, n_basis)
        square_amplitudes_perm = torch.square(amplitudes)

        # shape (batch_size, n_electrodes * n_phase_shifts, n_basis) @ (1, n_basis, n_groups)
        # -> (batch_size, n_electrodes * n_phase_shifts, n_groups)
        grouped_square_norms = square_amplitudes_perm @ (self.group_sel.T)[None, :, :]

        # (batch_size, n_electrodes * n_phase_shifts, n_groups)
        # -> (batch_size, n_electrodes * n_phase_shifts)
        group_sparse_norm = torch.sum(torch.sqrt(grouped_square_norms + self.epsilon_avoid_div_by_zero), dim=2)

        # no special modifications needed to deal with the tensor case for l12_lambda
        # since the shapes match already
        # shape (batch_size, n_electrodes * phase_shifts)
        return self.l12_lambda * group_sparse_norm

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, n_electrodes * n_shifts)
        mse_loss_contrib = self.compute_mse_loss(*args, **kwargs)

        # shape (batch, n_electrodes * n_shifts)
        group_sparse_term = self.compute_group_sparse_term(*args, **kwargs)

        # shape (batch, n_electrodes * n_shifts)
        total_loss = mse_loss_contrib + group_sparse_term

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
            return (torch.clamp_min_(amplitudes, 0.0),)

    def compute_loss_for_argmin(self, **kwargs) -> torch.Tensor:

        mse_loss_unshape = self.compute_mse_loss(*self.parameters(recurse=False), **kwargs)
        group_sparse_unshape = self.compute_group_sparse_term(*self.parameters(recurse=False), **kwargs)

        total_loss_unshape = mse_loss_unshape + group_sparse_unshape

        total_loss = total_loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_shifts)

        return total_loss

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
                 l12_lambda: Union[float, np.ndarray],
                 l12_group_sel_matrix: np.ndarray,
                 amplitudes_matrix_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 init_low: float = 0.0,
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

        self.lambda_is_tensor = isinstance(l12_lambda, np.ndarray)
        if self.lambda_is_tensor:
            if l12_lambda.shape != (batch, n_electrodes):
                raise ValueError(f"l12_lambda must have shape {(batch, n_electrodes)}, got {l12_lambda.shape}")

            l12_lambda = np.tile(l12_lambda[:, :, None], (1, 1, n_shifts))

            # we will flatten this tensor to have
            # shape (batch, n_electrodes * n_shifts)
            self.register_buffer('l12_lambda', torch.tensor(l12_lambda.reshape(batch, -1), dtype=torch.float32))
        else:
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
        mse_loss_contrib = 0.5 * xt_at_a_x - xt_at_b

        # shape (batch, n_electrodes * n_shifts)
        return mse_loss_contrib.reshape(self.batch_size, -1)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        '''

        :param args:
        :param kwargs:
        :return:
        '''

        # shape (batch, n_electrodes * n_shifts, n_groups)
        norms = args[self.NORMS_IDX_ARGS]

        # shape (batch, n_electrodes * n_shifts)
        mse_loss_contrib = self.compute_mse_loss(*args, **kwargs)

        # shape (batch, n_electrodes * n_shifts)
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

            ####### EVERYTHING BELOW THIS LINE IS THE OLD IMPLEMENTATION THAT WORKS #####
            # distribute the basis weights into their respective groups
            # as long as self.group_sel does not assign weights to multiple groups
            # we should be able to recover the original basis weights by
            # performing the operation torch.sum(zi_groupsel, dim=2)

            # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
            #   (1, 1, n_groups, n_basis)
            # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
            zi_groupsel = z_amplitudes_clipped[:, :, None, :] * self.group_sel[None, None, :, :]

            # shape (batch_size, n_electrodes * n_phase_shifts, n_groups)
            zi_norm = torch.linalg.norm(zi_groupsel, dim=3, ord=2)

            # shape (batch, n_electrodes * n_phase_shifts, n_groups), boolean-valued
            exceeds_abs = zi_norm > torch.abs(norms)

            # shape (batch, n_electrodes * n_phase_shifts, n_groups), boolean-valued
            less_than_neg = zi_norm <= -norms

            # now apply the clips
            # note that we need to project each group differently
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

        mse_loss_unshape = self.compute_mse_loss(*self.parameters(recurse=False), **kwargs)

        mse_loss = mse_loss_unshape.reshape(self.batch_size, self.n_electrodes, self.n_shifts)

        # now we need to compute the group norms
        # (batch_size, n_electrodes * n_phase_shifts, 1, n_basis) *
        #   (1, 1, n_groups, n_basis)
        # -> (batch_size, n_electrodes * n_phase_shifts, n_groups, n_basis)
        grouped_components = self.amplitudes[:, :, None, :] * self.group_sel[None, None, :, :]

        # shape (batch_size, n_electrodes * phase_shifts)
        group_sparse_norm = torch.sum(torch.norm(grouped_components, p=2, dim=3), 2) * self.l12_lambda

        group_sparse_penalty = group_sparse_norm.reshape(self.batch_size, self.n_electrodes, self.n_shifts)

        return mse_loss + group_sparse_penalty

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

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG_CONSTRAINED:
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

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG_SMOOTH:
        if group_sel_matrix is None:
            raise ValueError(f"group_sel_matrix cannot be none if specifying L12 group sparsity")

        return SharedShiftsNonNegOrthantGroupSparseProxGradSolver(
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
        regularization_lambda: Union[float, np.ndarray],
        regularization_type: RegularizationType,
        n_true_frequencies: int,
        solver_params: ProxSolverParams,
        device: torch.device,
        amplitude_matrix_real_np: Optional[np.ndarray] = None,
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
        (1) objective_values, shape (batch, n_electrodes, n_phase_shifts)
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

    #### Step 3: set up projected gradient descent problem #######################################
    _ = batch_multiproblem_parallel_prox_solve(solver,
                                               solver_params,
                                               verbose=solver_verbose)

    # shape (batch, n_electrodes, n_phase_shifts, n_basis)
    amplitudes_solved = solver.return_amplitudes()

    # shape (batch, n_electrodes, n_phase_shifts)
    objective_fn_vals = solver.compute_loss_for_argmin()

    del solver

    return amplitudes_solved.detach().cpu().numpy(), objective_fn_vals.detach().cpu().numpy()


def make_unshared_shifts_solver(
        regularization_type: RegularizationType,
        batch_unshared_at_a_matrix: np.ndarray,
        batch_at_b_vector: np.ndarray,
        valid_problem_mask: np.ndarray,
        reg_lambda: Union[float, np.ndarray],
        group_sel_matrix: Optional[np.ndarray] = None,
        amplitudes_matrix_init: Optional[np.ndarray] = None,
        verbose: bool = True,
        init_low: float = 0.0,
        init_high: float = 1e-2) \
        -> Union[UnsharedShiftSolver, BatchedMultiProxProblem]:
    '''

    :param regularization_type:
    :param batch_unshared_at_a_matrix:
    :param batch_at_b_vector:
    :param valid_problem_mask:
    :param reg_lambda:
    :param group_sel_matrix:
    :param amplitudes_matrix_init:
    :param verbose:
    :param init_low:
    :param init_high:
    :return:
    '''
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

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG_CONSTRAINED:
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

    elif regularization_type == RegularizationType.L12_GROUP_SPARSE_REG_SMOOTH:
        if group_sel_matrix is None:
            raise ValueError(f"group_sel_matrix cannot be None")

        return UnsharedShiftsNonNegOrthantGroupSparseProxGradSolver(
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
        regularization_lambda: Union[float, np.ndarray],
        regularization_type: RegularizationType,
        n_true_frequencies: int,
        solver_params: ProxSolverParams,
        device: torch.device,
        amplitude_matrix_real_np: Optional[np.ndarray] = None,
        group_sel_matrix: Optional[np.ndarray] = None,
        valid_problems: Optional[np.ndarray] = None,
        solver_verbose: bool = True,
        init_low: float = 0.0,
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

    _ = batch_multiproblem_parallel_prox_solve(solver,
                                               solver_params,
                                               verbose=solver_verbose)

    # shape (batch, n_electrodes, n_phase_shifts, n_basis)
    amplitudes_solved = solver.return_amplitudes()

    # shape (batch, n_electrodes, n_phase_shifts)
    objective_fn_vals = solver.compute_loss_for_argmin()

    del solver

    return amplitudes_solved.detach().cpu().numpy(), objective_fn_vals.detach().cpu().numpy()


def batched_coarse_to_fine_time_shifts_and_amplitudes2(
        observed_ft: np.ndarray,
        ft_basis: np.ndarray,
        n_true_frequencies: int,
        regularization_type: RegularizationType,
        regularization_lambda: Union[float, np.ndarray],
        valid_phase_shift_range: Tuple[int, int],
        first_pass_step_size: int,
        second_pass_best_n: int,
        second_pass_width: int,
        solver_params: ProxSolverParams,
        device: torch.device,
        group_sel_matrix: Optional[np.ndarray] = None,
        valid_problems: Optional[np.ndarray] = None,
        amplitude_initialize_range: Tuple[float, float] = (0.0, 1.0),
        max_batch_size: int = 1024,
        verbose_solver: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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

    pbar = tqdm.tqdm(total=int(np.ceil(n_valid_phase_shifts / max_batch_size)), leave=False,
                     desc='First pass grid search')
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
            solver_params,
            device,
            group_sel_matrix=group_sel_matrix,
            valid_problems=valid_problems,
            amplitude_matrix_real_np=amplitudes_random_init,
            solver_verbose=verbose_solver,
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

    pbar = tqdm.tqdm(total=int(np.ceil(n_second_pass_shifts / max_batch_size)), leave=False,
                     desc='Second pass fine search')
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
            solver_params,
            device,
            group_sel_matrix=group_sel_matrix,
            amplitude_matrix_real_np=amplitudes_random_init,
            valid_problems=valid_problems,
            solver_verbose=verbose_solver
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


def make_group_sparse_mat_from_group_list(group_list: List[np.ndarray],
                                          basis_size: int) -> np.ndarray:
    '''
    Transforms group list into group selection matrix

    :param group_list:
    :param basis_size:
    :return:
    '''

    n_groups = len(group_list)
    group_sel_matrix = np.zeros((n_groups, basis_size), dtype=np.float32)
    for i, group_idx in enumerate(group_list):
        group_sel_matrix[i, group_idx] = 1.0
    return group_sel_matrix


@dataclass
class WaveformPriorParams:
    waveform_cov_mat_time_domain: np.ndarray
    wavefrom_mean_time_domain: np.ndarray
    waveform_regularization_lambda: np.ndarray


def batch_shifted_fourier_nmf_iterative_optimization4(raw_waveform_data_matrix: np.ndarray,
                                                      is_valid_matrix: np.ndarray,
                                                      initialized_basis_waveforms: np.ndarray,
                                                      sparsity_regularization_type: RegularizationType,
                                                      sparsity_regularization_lambda: float,
                                                      valid_shift_range: Tuple[int, int],
                                                      shift_grid_step: int,
                                                      fine_search_top_n: int,
                                                      fine_search_width: int,
                                                      amplitude_init_range: Tuple[float, float],
                                                      n_iter: int,
                                                      solver_params: ProxSolverParams,
                                                      device: torch.device,
                                                      max_batch_size=8192,
                                                      waveform_prior_params: Optional[WaveformPriorParams] = None,
                                                      realign_basis_time_per_iter: bool = False,
                                                      realign_basis_sample_num: int = 60,
                                                      use_scaled_mse_penalty: bool = False,
                                                      use_scaled_regularization_terms: bool = False,
                                                      group_sel_matrix: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    '''
    Batched version of the main iteration loop for the two-step alternating optimization process.
    We can optionally specify a Gaussian prior for the waveform shapes
    Optimization steps are

        (1) With fixed waveforms, solve for both the amplitudes and the timeshifts together by performing grid search
            over a bunch of nonnegative least squares problems, with optional L1 regularization
        (2) With fixed amplitudes and timeshifts, solve for the waveforms with complex-valued least squares

    There are two different ways to specify the L1 / L12 group sparsity weights relative to the MSE loss
        (1) use_scaled_regularization_terms = False : In this case, we do not normalize the data waveforms,
            and the regularization is with respect to the raw waveform
        (2) use_scaled_regularization_terms = True : In this case, we normalize the data waveforms such that
            every data waveform has L2 norm 1, and then the regularization is defined with respect to the
            normalized waveform

    There are two different ways to specify how to weight the MSE contribution of each channel to the loss when performing
        the basis waveform shape optimization:
        (1) use_scaled_mse_penalty = False: The contribution of each channel to the overall MSE is the raw MSE with
            respect to the original unscaled data waveform. This method increases the importance of the large amplitude
            channels, since the large amplitude channels will tend to have the largest MSE values
        (2) use_scaled_mse_penalty = True: The contribution of each channel to the overall MSE is normalized
            by the magnitude of the data waveform. This method weights each channel approximately equally no matter
            the amplitude.

    :param raw_waveform_data_matrix: np.ndarray, time domain data matrix, shape (batch, n_electrodes, n_timepoints)
        Not normalized. This function takes care of the normalization
    :param is_valid_matrix: np.ndarray, 0-1 integer valued. shape (batch, n_electrodes). Each entry is 1 if the
        corresponding data waveform in raw_waveform_data_matrix corresponds to real data, and is 0 if the corresponding
        data waveform is nonsense / padding
    :param initialized_basis_waveforms: np.ndarray, time domain samples for basis waveforms,
        shape (batch, n_canonical_waveforms, n_timepoints)
    :param valid_shift_range: (low, high), range of valid sample shifts to consider for each canonical waveform
    :param shift_grid_step: spacing of the grid for the grid search
    :param fine_search_top_n: top n points from the grid search to expand on during the fine search
    :param fine_search_width: one-sided width of the fine search, centered around the top n points
    :param amplitude_init_range: (low, high), range to initialize the amplitudes
    :param n_iter: number of iteration steps
    :param device:
    :param l1_regularization_lambda: weight for L1 regularization
    :param waveform_prior_params: optional parameters for the waveform shape Gaussian prior
        If specified, we drop the L2 renormalization step for the waveforms, since the Gaussian prior
        should constrain the norm of the waveforms to a value near 1.

        This function parameter summarizes the time-domain covariance matrix and mean for each of the
        basis waveforms
    :return:
    '''

    is_valid_bool = is_valid_matrix.astype(bool)

    batch, n_observations, n_samples = raw_waveform_data_matrix.shape
    n_frequencies_not_rfft = n_samples
    n_basis_waveforms = initialized_basis_waveforms.shape[1]

    # shape (batch, n_observations, 1)
    raw_data_magnitude = np.linalg.norm(raw_waveform_data_matrix, axis=2, keepdims=True)
    raw_data_magnitude[~is_valid_bool, :] = 1.0  # set null magnitudes to 1 to avoid dividing by zero
    # shape (batch, n_observations, n_samples)
    scaled_raw_data = raw_waveform_data_matrix / raw_data_magnitude

    # compute the Fourier transform of the observed data once, ahead of time
    # shape (batch, n_observations, n_frequencies)
    observations_fourier_transform = np.fft.rfft(scaled_raw_data, axis=2)

    # shape (batch, n_basis_waveforms, n_frequencies)
    iter_basis_waveform_ft = np.fft.rfft(initialized_basis_waveforms, axis=2)

    waveform_observation_loss_weight = None
    if not use_scaled_mse_penalty:
        # shape (batch, n_observations)
        waveform_observation_loss_weight = raw_data_magnitude.copy().squeeze(2)

    if not use_scaled_regularization_terms:
        sparsity_regularization_lambda = 1.0 / raw_data_magnitude.squeeze(2)

    use_prior_waveform_optim = waveform_prior_params is not None
    if use_prior_waveform_optim:
        # shape (n_basis, n_rfft_freqs)
        waveform_prior_mean_rfft = np.fft.rfft(waveform_prior_params.wavefrom_mean_time_domain,
                                               axis=1)

        waveform_prior_mean_rfft_stacked = _pack_complex_to_real_imag(waveform_prior_mean_rfft,
                                                                      n_samples,
                                                                      axis=1)

        # convert the time-domain waveform covariance matrices to Fourier domain
        # shape (n_basis, n_rfft_freqs, n_rfft_freqs)
        waveform_cov_mat_rfft_domain = construct_rfft_covariance_matrix(
            waveform_prior_params.waveform_cov_mat_time_domain)

        # shape (n_basis, n_rfft_freqs, n_rfft_freqs)
        waveform_inv_cov_matrices = np.linalg.inv(waveform_cov_mat_rfft_domain)


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
        iter_real_amplitudes, iter_delays = batched_coarse_to_fine_time_shifts_and_amplitudes2(
            observations_fourier_transform,
            iter_basis_waveform_ft,
            n_frequencies_not_rfft,
            sparsity_regularization_type,
            sparsity_regularization_lambda,
            valid_shift_range,
            shift_grid_step,
            fine_search_top_n,
            fine_search_width,
            solver_params,
            device,
            group_sel_matrix=group_sel_matrix,
            valid_problems=is_valid_matrix,
            amplitude_initialize_range=amplitude_init_range,
            max_batch_size=max_batch_size,
            verbose_solver=False)

        if use_prior_waveform_optim:
            # complex valued, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
            iter_basis_waveform_ft = batch_fourier_complex_least_square_with_prior_optimize(
                iter_real_amplitudes,
                iter_delays,
                observations_fourier_transform,
                is_valid_matrix,
                waveform_inv_cov_matrices,
                waveform_prior_mean_rfft_stacked,
                n_frequencies_not_rfft,
                waveform_prior_params.waveform_regularization_lambda,
                device,
                observation_loss_weight=waveform_observation_loss_weight,
            )

        else:
            # complex valued, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
            iter_basis_waveform_ft = batch_fourier_complex_least_square_optimize3(
                iter_real_amplitudes,
                iter_delays,
                observations_fourier_transform,
                is_valid_matrix,
                iter_basis_waveform_ft,
                n_frequencies_not_rfft,
                device,
                observation_loss_weight=waveform_observation_loss_weight,
            )

        # shape (batch, n_canonical_waveforms, n_samples), real-valued float
        iter_basis_waveform_td = np.real(np.fft.irfft(iter_basis_waveform_ft, n=n_samples, axis=2))

        if realign_basis_time_per_iter:
            # shape (batch, n_canonical_waveforms, n_samples), real-valued float
            # and shape (batch, n_canonical_waveforms), integer
            iter_basis_waveform_td, iter_delays = shift_waveform_peaks_and_adjust_shifts(iter_basis_waveform_td,
                                                                                         iter_delays,
                                                                                         realign_basis_sample_num)
            iter_basis_waveform_ft = np.fft.rfft(iter_basis_waveform_td)

        if not use_prior_waveform_optim:
            # real valued np.ndarray, shape (batch, n_canonical_waveforms, 1)
            raw_optimized_waveform_magnitude = np.linalg.norm(iter_basis_waveform_td, axis=2, keepdims=True)
            # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_rfft_frequencies)
            iter_basis_waveform_ft = iter_basis_waveform_ft / raw_optimized_waveform_magnitude
            # real valued np.ndarray, shape (batch, n_canonical_waveforms, n_samples)
            iter_basis_waveform_td = iter_basis_waveform_td / raw_optimized_waveform_magnitude

            # shape (batch, n_observations, n_basis_waveforms) * (batch, 1, n_basis_waveforms)
            # -> (batch, n_observations, n_basis_waveforms)
            iter_real_amplitudes = iter_real_amplitudes * raw_optimized_waveform_magnitude.transpose((0, 2, 1))

        orig_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_basis_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=True,
                                           batch_observed_norms=waveform_observation_loss_weight,
                                           take_mean_over_electrodes=True)

        true_MSE = batch_evaluate_mse_flat(observations_fourier_transform,
                                           iter_real_amplitudes,
                                           iter_basis_waveform_ft,
                                           iter_delays,
                                           is_valid_matrix,
                                           n_frequencies_not_rfft,
                                           use_scaled_mse=False,
                                           batch_observed_norms=raw_data_magnitude.squeeze(2))

        mse_component = batch_evaluate_mse_flat(observations_fourier_transform,
                                                iter_real_amplitudes,
                                                iter_basis_waveform_ft,
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
    return fit_amplitudes_rescaled, iter_basis_waveform_td, iter_delays, loss_dict


@dataclass
class WaveformPriorSummary:
    gaussian_width_samples: Union[float, np.ndarray]
    waveform_regularization_lambda: Union[float, np.ndarray]


def construct_waveform_prior_params(waveform_prior_summary: WaveformPriorSummary,
                                    waveform_prior_mean: np.ndarray) \
        -> WaveformPriorParams:

    n_basis, n_timepoints = waveform_prior_mean.shape
    
    t_time = np.r_[0:n_timepoints]
    t_diff = t_time[:, None] - t_time[None, :]

    # by default use the same covariance matrix for everybody
    gaussian_prior_width = waveform_prior_summary.gaussian_width_samples
    cov_matrix_per_basis = np.zeros((n_basis, n_timepoints, n_timepoints), 
                                     dtype=np.float32)
    if isinstance(gaussian_prior_width, np.ndarray):
        if gaussian_prior_width.shape[0] != n_basis:
            raise ValueError(
                f'waveform_prior_summary shape must match n_basis, got {waveform_prior_summary.shape[0]}, expected {n_basis}')

        for ii, gp_width in enumerate(gaussian_prior_width):
            initial_basis_covariance = RBF(length_scale=gp_width)
            time_cov_matrix = initial_basis_covariance(t_diff)
            cov_matrix_per_basis[ii, ...] = time_cov_matrix

    else:
        initial_basis_covariance = RBF(length_scale=gaussian_prior_width)
        time_cov_matrix = initial_basis_covariance(t_diff)
        cov_matrix_per_basis[...] = time_cov_matrix[None, :, :]

    specified_lambda = waveform_prior_summary.waveform_regularization_lambda
    if isinstance(specified_lambda, np.ndarray):
        if specified_lambda.shape[0] != n_basis:
            raise ValueError(
                f'specified lambda shape must match n_basis, got {specified_lambda.shape[0]}, expected {n_basis}')
    else:
        specified_lambda = specified_lambda * np.ones((n_basis, ), dtype=np.float32)

    return WaveformPriorParams(cov_matrix_per_basis,
                               waveform_prior_mean,
                               specified_lambda)


def batch_two_step_decompose_cells_by_fitted_compartments2(
        eis_by_cell_id: Dict[int, np.ndarray],
        initialized_basis_vectors: np.ndarray,
        regularization_type: RegularizationType,
        solver_params: ProxSolverParams,
        device: torch.device,
        snr_abs_threshold: float = 5.0,
        amplitude_random_init_range: Tuple[float, float] = (0.0, 10.0),
        shifts: Tuple[int, int] = (-100, 100),
        grid_search_step: int = 5,
        grid_search_top_n: int = 4,
        fine_search_width: int = 2,
        grid_search_batch_size: int = 8192,
        maxiter_decomp: int = 3,
        waveform_prior_summary: Optional[WaveformPriorSummary] = None,
        realign_basis_time_per_iter: bool = False,
        realign_basis_sample_num: int = 20,
        l1_regularize_lambda: Optional[float] = None,
        use_scaled_mse_penalty: bool = False,
        use_scaled_regularization_terms: bool = False,
        grouped_l1l2_groups: Optional[List[np.ndarray]] = None) \
        -> Dict[int, Dict[str, np.ndarray]]:
    '''
    REVISION NOTE:

    No upsampling here anymore; that is taken care of during the data export/preprocessing step

    :param eis_by_cell_id: Dict mapping cell id to raw EIs. Each EI must have shape (n_electrodes, n_timepoints)
    :param initialized_basis_vectors: Initial value for basis waveforms in time domain. It helps to specify reasonable
        basis waveforms, for accelerated convergence and better fits

        Even though each EI will be fitted with its own separate basis waveforms, we use the same initialization
            for each EI, since this initialization corresponds to a generic reasonable initial guess

        shape (n_basis_waveforms, n_timepoints)
    :param device: torch device
    :param l1_regularize_lambda: float, regularization constant for L1 or group-sparse L1
    :param snr_abs_threshold:
    :param supersample_factor:
    :param shifts:
    :param maxiter_decomp:
    :return:
    '''
    n_basis, n_timepoints = initialized_basis_vectors.shape

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

        batch, max_n_sig_electrodes, n_timepoints = batched_data_mat.shape

        # shape (batch, n_basis, n_timepoints)
        batched_basis_waveforms = np.tile(initialized_basis_vectors, (batch, 1, 1))

        # if we are using the waveform prior, construct the covariance matrix
        waveform_prior_params = None
        if waveform_prior_summary is not None:
            waveform_prior_params = construct_waveform_prior_params(waveform_prior_summary,
                                                                    initialized_basis_vectors)

        # amplitudes has shape (batch, n_observations, n_basis_waveforms))
        # waveforms has shape (batch, n_basis_waveforms, n_timepoints)
        amplitudes, waveforms, delays, mse = batch_shifted_fourier_nmf_iterative_optimization4(
            batched_data_mat,
            is_valid_mat,
            batched_basis_waveforms,
            regularization_type,
            l1_regularize_lambda,
            shifts,
            grid_search_step,
            grid_search_top_n,
            fine_search_width,
            amplitude_random_init_range,
            maxiter_decomp,
            solver_params,
            device,
            max_batch_size=grid_search_batch_size,
            use_scaled_mse_penalty=use_scaled_mse_penalty,
            use_scaled_regularization_terms=use_scaled_regularization_terms,
            waveform_prior_params=waveform_prior_params,
            realign_basis_time_per_iter=realign_basis_time_per_iter,
            realign_basis_sample_num=realign_basis_sample_num,
            group_sel_matrix=make_group_sparse_mat_from_group_list(grouped_l1l2_groups,
                                                                   initialized_basis_vectors.shape[0]),
        )

        wip_decomp_list.append((amplitudes, delays, waveforms))
        batch_pbar.update(1)

    batch_pbar.close()

    # now unpack the results
    result_dict = auto_unbatch_unpack_significant_electrodes(wip_decomp_list, autobatched_list)

    return result_dict
