import torch

import pickle

import lib.ei_decomposition as ei_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='jointly decompose noramlized EIs for a given cell type into constituent waveforms by jointly optimizing waveforms, shifts, and amplitudes')

    parser.add_argument('data_pickle', type=str, help='path to raw data pickle')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--nbasis', '-n', type=int, default=3, help='number of basis waveforms')
    parser.add_argument('--maxiter', '-m', type=int, default=10, help='maximum number of iterations to run')
    parser.add_argument('--weight_reg', '-w', type=float, default=7.5e-2,
                        help='L1 regularization lambda for amplitudes')
    parser.add_argument('--sobolev_reg', '-s', type=float, default=0.0,
                        help='L2 regularization for waveform second derivatives')
    parser.add_argument('--upsample', '-u', type=int, default=5, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--grid_step', type=int, default=5, help='step size for grid search')
    parser.add_argument('--grid_top_n', type=int, default=4, help='top n for grid search')
    parser.add_argument('--fine_search_width', type=int, default=2, help='width for fine search')
    parser.add_argument('--grid_batch_size', type=int, default=8192, help='grid search batch size')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--renormalize_loss', '-r', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--renormalize_penalty', '-p', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--initialize_basis', '-i', type=str, default=None, help='path to initialized basis')
    parser.add_argument('--group', '-g', action='store_true', default=False,
                        help='whether or not to use group L1L2 regularization')
    parser.add_argument('--l1_comp_weights', '-l', action='store_true', default=False,
                        help='whether or not to use componentwise weighted L1 regularization')
    parser.add_argument('--eps_cutoff', '-e', type=float, default=1e-3,
                        help='converge epsilon. Default uses comparison to t^2 |G_t|^2, which is robust but does not guarantee bounds on convergence')
    parser.add_argument('--eps_eigen', '--f', action='store_true', default=False,
                        help='use original eigenvalue based convergence, which provides a provable suboptimality for least squares but breaks with strong regularization')

    args = parser.parse_args()

    compute_device = torch.device('cuda')

    print("Loading data")
    with open(args.data_pickle, 'rb') as pfile:

        preprocessed_dict = pickle.load(pfile)
        eis_by_cell_id = preprocessed_dict['eis_by_cell_id']  # type: Dict[int, np.ndarray]
        dataset_el_map = preprocessed_dict['electrode_map']  # type: np.ndarray

    # use the initialized basis if specified, otherwise specify the number of basis vectors
    initial_basis = None
    group_assignments = None,
    componentwise_weights = None
    if args.initialize_basis is not None:
        with open(args.initialize_basis, 'rb') as pfile:
            basis_dict = pickle.load(pfile)
        initial_basis = basis_dict['basis']

        shift_tuple = (-basis_dict['before'], basis_dict['after'])
        upsample_factor = basis_dict['upsample']
        snr_thresh = basis_dict['thresh']
        print(snr_thresh)

        group_assignments = None
        if args.group:
            group_assignments = basis_dict['group_assignments']

        componentwise_weights = None
        if args.l1_comp_weights:
            componentwise_weights = basis_dict['componentwise_weights']

    else:
        upsample_factor = args.upsample
        shift_tuple = (-args.before, args.after)
        snr_thresh = args.thresh

    if initial_basis is None:
        decomposition_dict, basis_waveforms, mse = ei_decomp.two_step_decompose_cells_by_fitted_compartments(
            eis_by_cell_id,
            compute_device,
            n_basis_vectors=args.nbasis,
            maxiter_decomp=args.maxiter,
            l1_regularize_lambda=args.weight_reg,
            sobolev_regularize_lambda=args.sobolev_reg,
            output_debug_dict=False,
            shifts=shift_tuple,
            supersample_factor=upsample_factor,
            snr_abs_threshold=snr_thresh,
            grid_search_step=args.grid_step,
            grid_search_top_n=args.grid_top_n,
            fine_search_width=args.fine_search_width,
            grid_search_batch_size=args.grid_batch_size,
            use_scaled_mse_penalty=args.renormalize_loss,
            use_scaled_regularization_terms=args.renormalize_penalty,
            converge_epsilon=args.eps_cutoff,
            converge_step_cutoff=args.eps_cutoff if not args.eps_eigen else None
        )
    else:
        decomposition_dict, basis_waveforms, mse = ei_decomp.two_step_decompose_cells_by_fitted_compartments(
            eis_by_cell_id,
            compute_device,
            initialized_basis_vectors=initial_basis,
            maxiter_decomp=args.maxiter,
            l1_regularize_lambda=args.weight_reg,
            sobolev_regularize_lambda=args.sobolev_reg,
            output_debug_dict=False,
            shifts=shift_tuple,
            supersample_factor=upsample_factor,
            snr_abs_threshold=snr_thresh,
            grid_search_step=args.grid_step,
            grid_search_top_n=args.grid_top_n,
            fine_search_width=args.fine_search_width,
            grid_search_batch_size=args.grid_batch_size,
            use_scaled_mse_penalty=args.renormalize_loss,
            use_scaled_regularization_terms=args.renormalize_penalty,
            use_grouped_l1l2_norm=args.group,
            grouped_l1l2_groups=group_assignments,
            use_basis_weighted_l1_norm=args.l1_comp_weights,
            basis_weights_for_l1=componentwise_weights,
            converge_epsilon=args.eps_cutoff,
            converge_step_cutoff=args.eps_cutoff if not args.eps_eigen else None
        )

    with open(args.output, 'wb') as joint_fit_file:
        metadata_dict = {
            'l1_reg': args.weight_reg,
            'sobolev_reg': args.sobolev_reg,
            'maxiter': args.maxiter,
            'padding': shift_tuple,
            'upsample': args.upsample,
            'thresh': snr_thresh,
            'scale_mse_for_waveforms': args.renormalize_loss,
            'scale_regularization_terms': args.renormalize_penalty,
            'use_grouped_l1l2_norm': args.group,
            'group_assignments': group_assignments,
            'use_basis_weighted_l1': args.l1_comp_weights,
            'basis_weights_for_l1': componentwise_weights
        }

        pickle_dict = {
            'decomposition': decomposition_dict,
            'waveforms': basis_waveforms,
            'mse': mse
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)
