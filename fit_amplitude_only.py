import visionloader as vl
import electrode_map as el_map

import torch

import pickle

import lib.ei_decomposition as ei_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decompose EIs into amplitudes with a known fixed waveform basis (amplitude only optimization')

    parser.add_argument('data_pickle', type=str, help='path to raw data pickle')
    parser.add_argument('basis_pickle', type=str, help='path to input pickle file')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--weight_reg', '-w', type=float, default=7.5e-2,
                        help='L1 regularization lambda for amplitudes')
    parser.add_argument('--grid_batch_size', type=int, default=8192, help='grid search batch size')
    parser.add_argument('--grid_step', type=int, default=5, help='step size for grid search')
    parser.add_argument('--grid_top_n', type=int, default=4, help='top n for grid search')
    parser.add_argument('--fine_search_width', type=int, default=2, help='width for fine search')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--renormalize_loss', '-r', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--renormalize_penalty', '-p', action='store_true', default=False,
                        help='renormalize data waveforms')
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
    cell_id_list = list(eis_by_cell_id.keys())

    with open(args.basis_pickle, 'rb') as pfile:
        basis_dict = pickle.load(pfile)
    initial_basis = basis_dict['basis']

    shift_tuple = (-basis_dict['before'], basis_dict['after'])
    upsample_factor = basis_dict['upsample']
    snr_thresh = basis_dict['thresh']

    group_assignments = None
    if args.group:
        group_assignments = basis_dict['group_assignments']

    componentwise_weights = None
    if args.l1_comp_weights:
        componentwise_weights = basis_dict['componentwise_weights']

    group_assignments = None
    if args.group:
        group_assignments = basis_dict['group_assignments']

    componentwise_weights = None
    if args.l1_comp_weights:
        componentwise_weights = basis_dict['componentwise_weights']

    decomposition_dict, basis_waveforms, mse = ei_decomp.decompose_cells_amplitudes_only(
        eis_by_cell_id,
        compute_device,
        initial_basis,
        l1_regularize_lambda=args.weight_reg,
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

    print(mse)

    with open(args.output, 'wb') as joint_fit_file:
        metadata_dict = {
            'l1_reg': args.weight_reg,
            'padding': shift_tuple,
            'upsample': upsample_factor,
            'thresh': snr_thresh
        }

        pickle_dict = {
            'decomposition': decomposition_dict,
            'waveforms': basis_waveforms,
            'mse': mse
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)
