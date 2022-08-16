import torch

import pickle
import argparse

import lib.batch_ei_decomposition_v2 as batch_ei_decomp2
from lib.optim.prox_optim import ProxFISTASolverParams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Fit EI decomposition with previously initialized basis waveforms. Separate basis waveforms for each cell')

    parser.add_argument('ei_pickle', type=str, help='path to EI pickle')
    parser.add_argument('basis_prior_pickle', type=str, help='path to input pickle file')
    parser.add_argument('output_pickle', type=str, help='path to output pickle file')
    parser.add_argument('--maxiter', '-m', type=int, default=10, help='maximum number of iterations to run')
    parser.add_argument('--weight_reg', '-w', type=float, default=7.5e-2,
                        help='L1 regularization lambda for amplitudes')
    parser.add_argument('--grid_step', type=int, default=5, help='step size for grid search')
    parser.add_argument('--grid_top_n', type=int, default=4, help='top n for grid search')
    parser.add_argument('--fine_search_width', type=int, default=2, help='width for fine search')
    parser.add_argument('--grid_batch_size', type=int, default=8192, help='grid search batch size')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--renormalize_loss', '-r', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--renormalize_penalty', '-p', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--group', '-g', action='store_true', default=False,
                        help='whether or not to use group L1L2 regularization')
    parser.add_argument('--prior_weight', '-pw', default=1.0,
                        help='Lambda for Gaussian prior regularization term')
    parser.add_argument('--prior_width', '-psigma', default=5.0,
                        help='Distance parameter for prior kernel')
    parser.add_argument('--eps_cutoff', '-e', type=float, default=1e-3,
                        help='converge epsilon. Default 1e-3')
    parser.add_argument('--opt_iter', '-o', type=int, default=10,
                        help='Maximum iters for inner FISTA solver')

    args = parser.parse_args()

    compute_device = torch.device('cuda')

    print("Loading pre-prepared EIs")
    with open(args.ei_pickle, 'rb') as pfile:
        x = pickle.load(pfile)
        eis_by_cell_id = x['eis_by_cell_id']

    print("Loading initial bases")
    with open(args.basis_prior_pickle, 'rb') as pfile:
        basis_dict = pickle.load(pfile)
    initial_basis = basis_dict['basis']

    shift_tuple = (-basis_dict['before'], basis_dict['after'])
    upsample_factor = basis_dict['upsample']

    group_assignments = None
    if args.group:
        group_assignments = basis_dict['group_assignments']

    waveform_prior_params = None
    if args.prior_weight != 0.0:
        waveform_prior_params = batch_ei_decomp2.WaveformPriorSummary(args.prior_width,
                                                                      args.prior_weight,
                                                                      True)

    decomposition_dict = batch_ei_decomp2.batch_two_step_decompose_cells_by_fitted_compartments2(
        eis_by_cell_id,
        initial_basis,
        batch_ei_decomp2.RegularizationType.L12_GROUP_SPARSE_REG_CONSTRAINED,
        ProxFISTASolverParams(initial_learning_rate=0.0,
                              max_iter=args.opt_iter,
                              converge_epsilon=args.eps_cutoff),
        compute_device,
        l1_regularize_lambda=args.weight_reg,
        snr_abs_threshold=args.thresh,
        shifts=shift_tuple,
        grid_search_step=args.grid_step,
        grid_search_top_n=args.grid_top_n,
        fine_search_width=args.fine_search_width,
        grid_search_batch_size=args.grid_batch_size,
        maxiter_decomp=args.maxiter,
        waveform_prior_summary=waveform_prior_params,
        grouped_l1l2_groups=group_assignments,
        use_scaled_mse_penalty=args.renormalize_loss,
        use_scaled_regularization_terms=args.renormalize_penalty,
    )

    with open(args.output_pickle, 'wb') as joint_fit_file:
        metadata_dict = {
            'l1_reg': args.weight_reg,
            'maxiter': args.maxiter,
            'padding': shift_tuple,
            'upsample': upsample_factor,
            'thresh': args.thresh,
            'scale_mse_for_waveforms': args.renormalize_loss,
            'scale_regularization_terms': args.renormalize_penalty,
            'use_grouped_l1l2_norm': args.group,
            'group_assignments': group_assignments,
            'use_basis_weighted_l1': args.l1_comp_weights,
            'initial_basis_mean': initial_basis,
            'GP_prior_weight': args.prior_weight,
            'GP_length': args.prior_width,
        }

        pickle_dict = {
            'decomposition': decomposition_dict,
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)
