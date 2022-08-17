from abc import ABC, ABCMeta

import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd

from typing import Tuple, Iterable, List, Union, Optional, Callable
import functools


def _multiproblem_flatten_variables(variable_list: Iterable[torch.Tensor],
                                    n_problems: int) -> torch.Tensor:
    '''
    Assumes that every variable has the same size for dim0, where dim0 corresponds to
        the number of convex problems

    This is so that we can flatten and then concatenate along the resulting dim1

    :param variable_list:
    :return:
    '''

    acc_list = [x.reshape(n_problems, -1) for x in variable_list]
    return torch.cat(acc_list, dim=1)


def _multiproblem_unflatten_variables(flat_variables: torch.Tensor,
                                      shape_sequence: Iterable[Tuple[int, ...]]) -> List[torch.Tensor]:
    return_sequence = []
    offset = 0
    for tup_seq in shape_sequence:
        sizeof = functools.reduce(lambda x, y: x * y, tup_seq)
        return_sequence.append(flat_variables[offset:offset + sizeof].reshape(tup_seq))
        offset += sizeof

    return return_sequence


class SolverParams(ABC):
    pass


class ProxSolverParams(SolverParams):
    pass


class ProxFixedStepSizeSolverParams(ProxSolverParams):
    def __init__(self,
                 initial_learning_rate: Union[float, torch.Tensor, None],
                 max_iter: int = 250,
                 converge_epsilon: float = 1e-6):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon


class ProxFISTASolverParams(ProxSolverParams):
    def __init__(self,
                 initial_learning_rate: float = 1.0,
                 max_iter: int = 250,
                 converge_epsilon: float = 1e-6,
                 backtracking_beta: float = 0.5):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon
        self.backtracking_beta = backtracking_beta


class ProxGradSolverParams(ProxSolverParams):

    def __init__(self,
                 initial_learning_rate: float = 1.0,
                 max_iter: int = 1000,
                 converge_epsilon: float = 1e-6,
                 backtracking_beta: float = 0.5):
        self.initial_learning_rate = initial_learning_rate
        self.max_iter = max_iter
        self.converge_epsilon = converge_epsilon
        self.backtracking_beta = backtracking_beta


class MultiProxProblem(nn.Module):

    def __init__(self, n_problems: int, verbose: bool = False):

        super().__init__()

        self.n_problems = n_problems
        self.verbose = verbose

    def check_variables(self):
        for x in self.parameters(recurse=False):
            assert x.shape[0] == self.n_problems, f"dim0 of each variable must be {self.n_problems}"

    def _batch_flatten_variables(self,
                                 variable_list: Iterable[torch.Tensor]) \
            -> torch.Tensor:
        return torch.cat([x.reshape(self.n_problems, -1)
                          for x in variable_list], dim=1)

    def _batch_unflatten_variables(self,
                                   flat_variables: torch.Tensor) \
            -> List[torch.Tensor]:

        return_seq, offset = [], 0
        shape_sequence = [x.shape for x in self.parameters(recurse=False)]
        for tup_seq in shape_sequence:
            sizeof = tup_seq[-1]
            return_seq.append(flat_variables[:, offset:offset + sizeof].reshape(tup_seq))
            offset += sizeof

        return return_seq

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _packed_prox_proj(self, packed_variables, **kwargs) -> torch.Tensor:

        with torch.no_grad():
            unpacked_variables = self._batch_unflatten_variables(packed_variables)
            prox_projected_unpacked = self._prox_proj(*unpacked_variables, **kwargs)
            return self._batch_flatten_variables(prox_projected_unpacked)

    def _smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _packed_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        '''

        :param packed_variables: packed variables, shape (n_problems, ?)
        :param kwargs:
        :return: shape (n_problems, )
        '''
        unpacked_variables = self._batch_unflatten_variables(packed_variables)
        return self._smooth_loss(*unpacked_variables, **kwargs)

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        (This works but might be very slow)

        This is a default implementation using autograd, if it is possible and easier to do
            by manually computing the gradient instead of using autograd, this method
            should be overridden.

        :param packed_variables:
        :param kwargs:
        :return: shape (n_problems, ) and (n_problems, ?)
        '''

        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        loss_tensor = self._packed_smooth_loss(packed_variables, **kwargs)
        gradients_output, = autograd.grad(torch.sum(loss_tensor),
                                          packed_variables)

        return loss_tensor, gradients_output

    def assign_optimization_vars(self, *cloned_parameters) -> None:
        for param, assign_val in zip(self.parameters(recurse=False),
                                     cloned_parameters):
            param.data[:] = assign_val.data[:]

    def fixed_step_size_prox_solve(self,
                                   step_size: torch.Tensor,
                                   max_iter: int,
                                   converge_epsilon: float,
                                   **kwargs) -> torch.Tensor:
        '''
        Since we are taking fixed steps here, we can be much much
            simpler with our implementation
        :param step_size: shape (n_problems, )
        :param max_iter: int, maximum number of iterations to
            run the optimization for before giving up
        :param converge_epsilon: if the norm of the gradient is smaller
            than this value for all of the problems, stop early
        :param kwargs:
        :return:
        '''

        # shape (n_problems, ?)
        vars_iter = self._batch_flatten_variables(self.parameters(recurse=False)).detach().clone()

        for iter in range(max_iter):

            # shape (n_problems, ) and (n_problems, ?)
            loss, gradient = self._packed_loss_and_gradients(vars_iter, **kwargs)

            with torch.no_grad():
                update_unproj = vars_iter - step_size[:, None] * gradient
                vars_iter = self._packed_prox_proj(update_unproj)

                # evaluate termination condition

                # shape (n_problems, )
                grad_norm = torch.norm(gradient, dim=1)
                if torch.all(grad_norm < converge_epsilon):
                    break

        with torch.no_grad():
            stepped_variables = self._batch_unflatten_variables(vars_iter)
            self.assign_optimization_vars(*stepped_variables)
            return self._smooth_loss(*stepped_variables, **kwargs)

    def prox_grad_solve(self,
                        init_learning_rate: float,
                        max_iter: int,
                        converge_epsilon: float,
                        backtracking_beta: float,
                        **kwargs) -> torch.Tensor:

        # shape (n_problems, ?)
        vars_iter = self._batch_flatten_variables(self.parameters(recurse=False)).detach().clone()

        # shape (n_problems, )
        step_size = torch.ones((self.n_problems, ), dtype=vars_iter.dtype,
                               device=vars_iter.device) * init_learning_rate

        for iter in range(max_iter):

            # shape (n_problems, ) and shape (n_problems, ?)
            current_loss, gradient_flattened = self._packed_loss_and_gradients(vars_iter, **kwargs)

            # Backtracking search, with projection inside the search
            vars_iter, step_size = self._projected_backtracking_search(
                current_loss,
                vars_iter,
                gradient_flattened,
                step_size,
                backtracking_beta,
                **kwargs
            )

            if self.verbose:
                current_mean = torch.mean(current_loss)
                print(f"iter={iter}, mean loss={current_mean.item()}\r", end="")

            has_converged = torch.norm(gradient_flattened, dim=1) < converge_epsilon
            if torch.all(has_converged):
                break

        with torch.no_grad():
            stepped_variables = self._batch_unflatten_variables(vars_iter)
            self.assign_optimization_vars(*stepped_variables)
            return self._smooth_loss(*stepped_variables, **kwargs)

    def _projected_backtracking_search(self,
                                       s_loss: torch.Tensor,
                                       s_vars_vectorized: torch.Tensor,
                                       gradients_vectorized: torch.Tensor,
                                       step_size: torch.Tensor,
                                       backtracking_beta: float,
                                       **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Backtracking search on a quadratic approximation to the smooth part
            of the loss function

        :param s_loss: shape (n_problems, )
        :param s_vars_vectorized: shape (n_problems, ?)
        :param gradients_vectorized: shape (n_problems, ?)
        :param step_size: shape (n_problems, )
        :param backtracking_beta:
        :param kwargs:
        :return:
        '''

        def eval_quadratic_approximation(next_vars: torch.Tensor,
                                         approx_center: torch.Tensor,
                                         step: torch.Tensor) -> torch.Tensor:
            '''

            :param next_vars: shape (n_problems, ?)
            :param approx_center: shape (n_problems, ?)
            :param step: shape (n_problems, )
            :return: quadratic approximation to the loss function,
                shape (n_problems, )
            '''

            with torch.no_grad():
                # shape (n_problems, )
                div_mul = 1.0 / (2.0 * step)

                # shape (n_problems, ?)
                diff = next_vars - approx_center

                # shape (n_problems, ?) -> (n_problems, )
                gradient_term = torch.sum(gradients_vectorized * diff, dim=1)

                # shape (n_problems, ?) -> (n_problems, )
                diff_term = torch.sum(diff * diff, dim=1)

                return s_loss + gradient_term + div_mul * diff_term

        with torch.no_grad():
            # shape (n_problems, ?)
            stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, None]

            # shape (n_problems, ?)
            projected_stepped_vars = self._packed_prox_proj(stepped_vars_vectorized)

            # shape (n_problems, )
            quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

            # shape (n_problems, )
            stepped_loss = self._packed_eval_smooth_loss(projected_stepped_vars, **kwargs)

            approx_smaller_loss = quad_approx_val < stepped_loss

            while torch.any(approx_smaller_loss):
                # shape (n_problems, )
                update_multiplier = approx_smaller_loss.float() * backtracking_beta \
                                    + (~approx_smaller_loss).float() * 1.0
                # shape (n_problems, )
                step_size = step_size * update_multiplier

                # shape (n_problems, ?)
                stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, None]

                # shape (n_problems, ?)
                projected_stepped_vars = self._packed_prox_proj(stepped_vars_vectorized)

                # shape (n_problems, )
                quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

                # shape (n_problems, )
                stepped_loss = self._packed_eval_smooth_loss(projected_stepped_vars, **kwargs)

                approx_smaller_loss = quad_approx_val < stepped_loss

            return projected_stepped_vars, step_size

    def fista_prox_solve(self,
                         initial_learning_rate: float,
                         max_iter: int,
                         converge_epsilon: float,
                         backtracking_beta: float,
                         **kwargs) -> torch.Tensor:
        '''
        Solves the smooth / unsmooth separable problem using the FISTA
            accelerated first order gradient method

        For now, we work with a single vector of combined flattened parameters
            when we implement the generic algorithm

        :param initial_learning_rate: float, initial learning rate
        :param max_iter: int, maximum number of FISTA iterations to run
        :param converge_epsilon:
        :param line_search_alpha:
        :param backtracking_beta:
        :param return_loss:
        :param kwargs:
        :return:
        '''

        # shape (n_problems, ?)
        vars_iter = self._batch_flatten_variables(self.parameters(recurse=False)).detach().clone()
        vars_iter_min1 = vars_iter.detach().clone()

        # shape (n_problems, )
        step_size = torch.ones((self.n_problems,), dtype=vars_iter.dtype,
                               device=vars_iter.device) * initial_learning_rate

        t_iter_min1, t_iter = 0.0, 1.0

        for iter in range(max_iter):
            alpha_iter = t_iter_min1 / t_iter
            t_iter_min1 = t_iter
            t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

            with torch.no_grad():
                s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)
            s_iter.requires_grad = True

            vars_iter_min1 = vars_iter

            # compute the forward loss and gradients for s_iter

            # shape (n_problems, ) and (n_problems, ?)
            current_loss, gradient_flattened = self._packed_loss_and_gradients(s_iter, **kwargs)

            # FISTA backtracking search, with projection inside the backtracking search
            vars_iter, step_size = self._projected_backtracking_search(
                current_loss,
                s_iter,
                gradient_flattened,
                step_size,
                backtracking_beta,
                **kwargs
            )

            if self.verbose:
                print(f"iter={iter}, mean loss={torch.mean(current_loss).item()}\r", end='')

            # quit early if all of the problems converged
            has_converged = torch.norm(vars_iter - vars_iter_min1, dim=1) < converge_epsilon
            if torch.all(has_converged):
                break

        with torch.no_grad():
            stepped_variables = self._batch_unflatten_variables(vars_iter)
            self.assign_optimization_vars(*stepped_variables)
            return self._smooth_loss(*stepped_variables, **kwargs)

    def solve(self,
              solver_params: ProxSolverParams,
              **kwargs) -> torch.Tensor:
        if isinstance(solver_params, ProxFixedStepSizeSolverParams):

            if solver_params.initial_learning_rate is None:
                try:
                    step_size = self.compute_fixed_step_size()
                    return self.fixed_step_size_prox_solve(
                        step_size,
                        solver_params.max_iter,
                        solver_params.converge_epsilon
                    )
                except NotImplementedError:
                    raise ValueError("initial_learning_rate if compute_fixed_step_size not implemented")

            return self.fixed_step_size_prox_solve(
                solver_params.initial_learning_rate,
                solver_params.max_iter,
                solver_params.converge_epsilon
            )

        elif isinstance(solver_params, ProxGradSolverParams):
            return self.prox_grad_solve(
                solver_params.initial_learning_rate,
                solver_params.max_iter,
                solver_params.converge_epsilon,
                solver_params.backtracking_beta,
                **kwargs
            )

        elif isinstance(solver_params, ProxFISTASolverParams):
            return self.fista_prox_solve(
                solver_params.initial_learning_rate,
                solver_params.max_iter,
                solver_params.converge_epsilon,
                solver_params.backtracking_beta,
                **kwargs
            )

        raise TypeError("solver_params must be instance of ProxSolverParams")


class _BatchedMultiSolverMixin(metaclass=ABCMeta):

    batch_size: int
    n_problems: int
    parameters: Callable[[Optional[bool]], Iterable[torch.Tensor]]

    def _batched_multiproblem_flatten_variables(self,
                                                variable_list: Iterable[torch.Tensor]) \
            -> torch.Tensor:
        '''

        :param variable_list:
        :return:
        '''
        return torch.cat([x.reshape(self.batch_size, self.n_problems, -1)
                          for x in variable_list], dim=2)

    def _batched_multiproblem_unflatten_variables(self,
                                                  flat_variables: torch.Tensor) \
            -> List[torch.Tensor]:

        return_sequence = []
        offset = 0
        shape_sequence = [x.shape for x in self.parameters(recurse=False)]
        for tup_seq in shape_sequence:
            sizeof = tup_seq[-1]
            return_sequence.append(flat_variables[:, :, offset:offset + sizeof].reshape(tup_seq))
            offset += sizeof

        return return_sequence



class BatchedMultiProxProblem(nn.Module, _BatchedMultiSolverMixin, metaclass=ABCMeta):

    def __init__(self,
                 batch_size: int,
                 n_problems: int,
                 valid_problems: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 verbose: bool = True):

        super().__init__()

        self.batch_size = batch_size
        self.n_problems = n_problems

        self.has_invalid_problems = (valid_problems is not None)
        if self.has_invalid_problems:
            if valid_problems.shape != (batch_size, n_problems):
                raise ValueError(f"valid_problems must have shape ({batch_size}, {n_problems}) if it is specified")

            if isinstance(valid_problems, np.ndarray):
                self.register_buffer('valid_problems', torch.tensor(valid_problems, dtype=torch.bool))
            else:
                self.register_buffer('valid_problems', valid_problems.detach().clone().bool())

        self.verbose = verbose

    def check_variables(self):
        for x in self.parameters(recurse=False):
            assert x.shape[0] == self.batch_size and x.shape[1] == self.n_problems, \
                f"dim0 of each variable must be {self.batch_size}, dim1 of each variable must be {self.n_problems}"

    def _prox_proj(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _packed_prox_proj(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:

        with torch.no_grad():
            unpacked_variables = self._batched_multiproblem_unflatten_variables(packed_variables)
            prox_projected_unpacked = self._prox_proj(*unpacked_variables, **kwargs)
            return self._batched_multiproblem_flatten_variables(prox_projected_unpacked)

    def _eval_smooth_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _packed_eval_smooth_loss(self, packed_variables: torch.Tensor, **kwargs) -> torch.Tensor:
        '''

        :param packed_variables: shape (batch, n_problems, ?)
        :param kwargs:
        :return: shape (batch, n_problems)
        '''

        unpacked_variables = self._batched_multiproblem_unflatten_variables(packed_variables)
        return self._eval_smooth_loss(*unpacked_variables, **kwargs)

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) \
            -> torch.Tensor:

        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        # shape (batch, n_problems)
        loss_tensor = self._packed_eval_smooth_loss(packed_variables, **kwargs)

        gradients_output, = autograd.grad(torch.sum(loss_tensor),
                                          packed_variables)

        return gradients_output

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        packed_variables.requires_grad_(True)
        if packed_variables.grad is not None:
            packed_variables.grad.zero_()

        # shape (batch, n_problems)
        loss_tensor = self._packed_eval_smooth_loss(packed_variables, **kwargs)

        gradients_output, = autograd.grad(torch.sum(loss_tensor),
                                          packed_variables)

        return loss_tensor, gradients_output

    def assign_optimization_vars(self, *cloned_parameters) -> None:
        for param, assign_val in zip(self.parameters(recurse=False),
                                     cloned_parameters):
            param.data[:] = assign_val.data[:]

    def compute_fixed_step_size(self) -> torch.Tensor:
        raise NotImplementedError


class ManualGradBatchMultiProxProblem(BatchedMultiProxProblem, ABC):

    def __init__(self,
                 batch_size: int,
                 n_problems: int,
                 valid_problems: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 verbose: bool = True):
        super().__init__(batch_size,
                         n_problems,
                         valid_problems=valid_problems,
                         verbose=verbose)

    def _manual_gradients(self, *args, **kwargs) \
            -> Tuple[torch.Tensor, ...]:
        '''
        Manually computes the gradients, bypassing the autograd

        This is really only useable for simple objective functions,
            or for applications where the user is optimizing for
            straight-line speed or GPU memory consumption

        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError

    def _packed_gradients_only(self, packed_variables: torch.Tensor, **kwargs) \
            -> torch.Tensor:
        '''
        Implementation does not require autograd

        :param packed_variables:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            unpacked_variables = self._batched_multiproblem_unflatten_variables(packed_variables)
            prox_projected_unpacked = self._manual_gradients(*unpacked_variables, **kwargs)
            return self._batched_multiproblem_flatten_variables(prox_projected_unpacked)

    def _packed_loss_and_gradients(self, packed_variables: torch.Tensor, **kwargs) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Default implementation; it may be more efficient to override this
            if there is substantial overlap between the gradient calculation
            and the loss calculation.

        Implementation does not require autograd
        :param packed_variables:
        :param kwargs:
        :return:
        '''

        with torch.no_grad():
            packed_gradients = self._packed_gradients_only(packed_variables, **kwargs)
            packed_loss = self._packed_eval_smooth_loss(packed_variables, **kwargs)
            return packed_loss, packed_gradients
