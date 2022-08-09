import torch
import numpy as np

from typing import Tuple, Union

from lib.optim.optim_base import BatchedMultiProxProblem


class ProxSolverParams:
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


def _batch_multiproblem_projected_backtracking_search(
        batch_multi_prox_problem: BatchedMultiProxProblem,
        s_loss: torch.Tensor,
        s_vars_vectorized: torch.Tensor,
        gradients_vectorized: torch.Tensor,
        step_size: torch.Tensor,
        backtracking_beta: float,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''

    :param batch_multi_prox_problem: BatchedMultiProxProblem
    :param s_loss: torch.Tensor, shape (batch, n_problems)
    :param s_vars_vectorized: torch.Tensor, shape (batch, n_problems, ?)
    :param gradients_vectorized: torch.Tensor, shape (batch, n_problems, )
    :param step_size: torch.Tensor, shape (batch, n_problems)
    :param backtracking_beta: float
    :param kwargs:
    :return:
    '''

    def eval_quadratic_approximation(next_vars: torch.Tensor,
                                     approx_center: torch.Tensor,
                                     step: torch.Tensor) -> torch.Tensor:
        '''

        :param next_vars: shape (batch, n_problems, ?)
        :param approx_center: shape (batch, n_problems, ?)
        :param step: shape (batch, n_problems)
        :return: quadratic approximation to the loss function,
            shape (n_problems, ?)
        '''

        with torch.no_grad():
            # shape (batch, n_problems)
            div_mul = 1.0 / (2.0 * step)

            # shape (batch, n_problems, ?)
            diff = next_vars - approx_center

            # shape (batch, n_problems, )
            gradient_term = torch.sum(gradients_vectorized * diff, dim=2)

            # shape (batch, n_problems, ?) -> (batch, n_problems)
            diff_term = torch.sum(diff * diff, dim=2)

            return s_loss + gradient_term + div_mul * diff_term

    with torch.no_grad():
        # shape (batch, n_problems, ?)
        stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, :, None]

        # shape (batch, n_problems, ?)
        projected_stepped_vars = batch_multi_prox_problem._packed_prox_proj(stepped_vars_vectorized, **kwargs)

        # shape (batch, n_problems)
        quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

        # shape (batch, n_problems)
        stepped_loss = batch_multi_prox_problem._packed_eval_smooth_loss(projected_stepped_vars, **kwargs)

        # shape (batch, n_problems), boolean valued
        approx_smaller_loss = quad_approx_val < stepped_loss

        # we only care about the above for the valid problems
        cannot_terminate = (approx_smaller_loss & batch_multi_prox_problem.valid_problems) \
            if batch_multi_prox_problem.has_invalid_problems else approx_smaller_loss

        while torch.any(cannot_terminate):
            # shape (batch, n_problems)
            update_multiplier = approx_smaller_loss.float() * backtracking_beta + (
                ~approx_smaller_loss).float() * 1.0
            step_size = step_size * update_multiplier

            # shape (batch, n_problems, ?)
            stepped_vars_vectorized = s_vars_vectorized - gradients_vectorized * step_size[:, :, None]

            # shape (batch, n_problems, ?)
            projected_stepped_vars = batch_multi_prox_problem._packed_prox_proj(stepped_vars_vectorized)

            # shape (batch, n_problems, )
            quad_approx_val = eval_quadratic_approximation(projected_stepped_vars, s_vars_vectorized, step_size)

            # shape (batch, n_problems, )
            stepped_loss = batch_multi_prox_problem._packed_eval_smooth_loss(projected_stepped_vars, **kwargs)

            approx_smaller_loss = quad_approx_val < stepped_loss

            cannot_terminate = (approx_smaller_loss & batch_multi_prox_problem.valid_problems) \
                if batch_multi_prox_problem.has_invalid_problems else approx_smaller_loss

        return projected_stepped_vars, step_size, stepped_loss


def batch_multiproblem_parallel_prox_grad_solve(batched_multi_prox_problem: BatchedMultiProxProblem,
                                                init_learning_rate: float,
                                                max_iter: int,
                                                converge_epsilon: float,
                                                backtracking_beta: float,
                                                verbose: bool = False,
                                                **kwargs) -> torch.Tensor:
    # shape (batch, n_problems, ?)
    vars_iter = batched_multi_prox_problem._batched_multiproblem_flatten_variables(
        batched_multi_prox_problem.parameters(recurse=False)
    ).detach().clone()

    # shape (batch, n_problems)
    step_size = torch.ones((batched_multi_prox_problem.batch_size, batched_multi_prox_problem.n_problems),
                           dtype=vars_iter.dtype,
                           device=vars_iter.device) * init_learning_rate

    for iter in range(max_iter):

        # shape (batch, n_problems) and (batch, n_problems, ?)
        current_loss, gradient_flattened = batched_multi_prox_problem._packed_loss_and_gradients(vars_iter, **kwargs)

        # Backtracking search, with projection inside the backtracking search
        vars_iter, step_size, candidate_loss = _batch_multiproblem_projected_backtracking_search(
            batched_multi_prox_problem,
            current_loss,
            vars_iter,
            gradient_flattened,
            step_size,
            backtracking_beta,
            **kwargs
        )

        if verbose:
            # computing the loss isn't trivial
            with torch.no_grad():
                if batched_multi_prox_problem.has_invalid_problems:
                    to_include_in_mean = torch.sum(current_loss * batched_multi_prox_problem.valid_problems)
                    current_mean = to_include_in_mean / torch.sum(batched_multi_prox_problem.valid_problems)
                else:
                    current_mean = torch.mean(current_loss)
                print(f"iter={iter}, mean loss={current_mean.item()}")

        # early termination criterion
        has_converged = torch.norm(gradient_flattened, dim=-1) < converge_epsilon
        if batched_multi_prox_problem.has_invalid_problems:
            has_converged = has_converged | (~batched_multi_prox_problem.valid_problems)

        if torch.all(has_converged):
            break

    with torch.no_grad():
        stepped_variables = batched_multi_prox_problem._batched_multiproblem_unflatten_variables(vars_iter)
        batched_multi_prox_problem.assign_optimization_vars(*stepped_variables)
        return batched_multi_prox_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def batch_multiproblem_parallel_fista_prox_solve(batch_multi_prox_problem: BatchedMultiProxProblem,
                                                 initial_learning_rate: float,
                                                 max_iter: int,
                                                 converge_epsilon: float,
                                                 backtracking_beta: float,
                                                 verbose: bool = False,
                                                 **kwargs) -> torch.Tensor:
    '''

    @param batch_multi_prox_problem:
    @param initial_learning_rate:
    @param max_iter:
    @param converge_epsilon:
    @param backtracking_beta:
    @param verbose:
    @param kwargs:
    @return:
    '''

    # shape (batch, n_problems, ?)
    vars_iter = batch_multi_prox_problem._batched_multiproblem_flatten_variables(
        batch_multi_prox_problem.parameters(recurse=False))
    vars_iter_min1 = vars_iter.detach().clone()

    # shape (batch, n_problems)
    step_size = torch.ones((batch_multi_prox_problem.batch_size, batch_multi_prox_problem.n_problems),
                           dtype=vars_iter.dtype, device=vars_iter.device) * initial_learning_rate

    t_iter_min1, t_iter = 0.0, 1.0

    # this variable is to keep track of the value of the variables
    # that produces the minimum value of the objective seen so far
    # We always return this variable, so that this implementation of FISTA
    # is a descent method, where the value of the objective function is nonincreasing
    descent_vars_iter = vars_iter.detach().clone()
    descent_loss = torch.empty((batch_multi_prox_problem.batch_size, batch_multi_prox_problem.n_problems),
                               dtype=vars_iter.dtype, device=vars_iter.device)
    descent_loss.data[:] = torch.inf

    for iter in range(max_iter):
        alpha_iter = t_iter_min1 / t_iter
        t_iter_min1 = t_iter
        t_iter = (1 + np.sqrt(1 + 4 * t_iter ** 2)) / 2.0

        with torch.no_grad():
            s_iter = vars_iter + alpha_iter * (vars_iter - vars_iter_min1)
        s_iter.requires_grad = True

        vars_iter_min1 = vars_iter

        # compute the forward loss and gradients for s_iter

        # shape (batch, n_problems, ) and (batch, n_problems, ?)
        current_loss, gradient_flattened = batch_multi_prox_problem._packed_loss_and_gradients(s_iter, **kwargs)

        vars_iter, step_size, candidate_loss = _batch_multiproblem_projected_backtracking_search(
            batch_multi_prox_problem,
            current_loss,
            s_iter,
            gradient_flattened,
            step_size,
            backtracking_beta,
            **kwargs
        )

        # We are doing the descent version of FISTA (implementation from L. Vandenberghe UCLA EE236C notes)
        # We reject vars_iter as a solution to the problem
        # if the loss is too high, but use vars_iter to compute the next guess
        # Basically if the next guess sucks, we keep track of the last good guess to return
        # but use the next guess to compute future guesses.
        with torch.no_grad():
            is_improvement = (candidate_loss < descent_loss)
            descent_loss.data[is_improvement] = candidate_loss.data[is_improvement]

            # FIXME may have to figure out how to do this properly if it crashes
            descent_vars_iter.data[is_improvement, :] = vars_iter.data[is_improvement, :]

        if verbose:
            # computing the loss isn't trivial
            with torch.no_grad():
                if batch_multi_prox_problem.has_invalid_problems:
                    to_include_in_mean = torch.sum(current_loss * batch_multi_prox_problem.valid_problems)
                    current_mean = to_include_in_mean / torch.sum(batch_multi_prox_problem.valid_problems)
                else:
                    current_mean = torch.mean(current_loss)
                print(f"iter={iter}, mean loss={current_mean.item()}")

        # early termination criterion
        has_converged = torch.norm(gradient_flattened, dim=-1) < converge_epsilon
        if batch_multi_prox_problem.has_invalid_problems:
            has_converged = has_converged | (~batch_multi_prox_problem.valid_problems)

        if torch.all(has_converged):
            break

    with torch.no_grad():
        stepped_variables = batch_multi_prox_problem._batched_multiproblem_unflatten_variables(vars_iter)
        batch_multi_prox_problem.assign_optimization_vars(*stepped_variables)
        return batch_multi_prox_problem._eval_smooth_loss(*stepped_variables, **kwargs)


def batch_multiproblem_parallel_prox_solve(
        batch_prox_problem: BatchedMultiProxProblem,
        solver_params: ProxSolverParams,
        verbose: bool = False,
        **kwargs) -> torch.Tensor:
    '''

    @param batch_prox_problem:
    @param solver_params:
    @param verbose:
    @param kwargs:
    @return:
    '''
    if isinstance(solver_params, ProxGradSolverParams):
        return batch_multiproblem_parallel_prox_grad_solve(
            batch_prox_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    elif isinstance(solver_params, ProxFISTASolverParams):
        return batch_multiproblem_parallel_fista_prox_solve(
            batch_prox_problem,
            solver_params.initial_learning_rate,
            solver_params.max_iter,
            solver_params.converge_epsilon,
            solver_params.backtracking_beta,
            verbose=verbose,
            **kwargs
        )

    raise TypeError("solver_params must be instance of ProxSolverParams")
