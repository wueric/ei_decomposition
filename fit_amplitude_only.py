import visionloader as vl
import electrode_map as el_map

import torch

import pickle

import lib.ei_decomposition as ei_decomp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decompose EIs into amplitudes with a known fixed waveform basis (amplitude only optimization')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('cell_type', type=str, help='cell type of interest')
    parser.add_argument('basis_pickle', type=str, help='path to input pickle file')
    parser.add_argument('output', type=str, help='path to output pickle file')
    parser.add_argument('--weight_reg', '-w', type=float, default=7.5e-2,
                        help='L1 regularization lambda for amplitudes')
    parser.add_argument('--upsample', '-u', type=int, default=5, help='upsample factor')
    parser.add_argument('--before', '-b', type=int, default=100, help='left shift samples')
    parser.add_argument('--after', '-a', type=int, default=100, help='right shift samples')
    parser.add_argument('--grid_batch_size', type=int, default=8192, help='grid search batch size')
    parser.add_argument('--grid_step', type=int, default=5, help='step size for grid search')
    parser.add_argument('--grid_top_n', type=int, default=4, help='top n for grid search')
    parser.add_argument('--fine_search_width', type=int, default=2, help='width for fine search')
    parser.add_argument('--thresh', '-t', type=float, default=5.0, help='EI amplitude cutoff')
    parser.add_argument('--cell_list', '-c', type=str, default=None,
                        help='Override cell_type argument, instead use cell ids in specified file')
    parser.add_argument('--renormalize_loss', '-r', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--renormalize_penalty', '-p', action='store_true', default=False,
                        help='renormalize data waveforms')
    parser.add_argument('--group', '-g', action='store_true', default=False,
                        help='whether or not to use group L1L2 regularization')
    parser.add_argument('--l1_comp_weights', '-l', action='store_true', default=False,
                        help='whether or not to use componentwise weighted L1 regularization')


    args = parser.parse_args()

    compute_device = torch.device('cuda')

    print("Loading data")
    dataset = vl.load_vision_data(args.ds_path,
                                  args.ds_name,
                                  include_params=True,
                                  include_ei=True)
    dataset_el_map = dataset.get_electrode_map()

    if args.cell_list is not None:

        with open(args.cell_list, 'r') as cell_id_file:
            cell_id_list = list(
                map(lambda x: int(x), cell_id_file.readline().strip('\n').split(',')))

        eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_id_list}
    else:
        cell_id_list = dataset.get_all_cells_of_type(args.cell_type)
        eis_by_cell_id = {cell_id: dataset.get_ei_for_cell(cell_id).ei for cell_id in cell_id_list}

    with open(args.basis_pickle, 'rb') as pfile:
        basis_dict = pickle.load(pfile)
        
    initial_basis = basis_dict['basis']

    group_assignments = None
    if args.group:
        group_assignments = basis_dict['group_assignments']

    componentwise_weights = None
    if args.l1_comp_weights:
        componentwise_weights = basis_dict['componentwise_weights']

    shift_tuple = (-args.before, args.after)

    decomposition_dict, basis_waveforms, mse = ei_decomp.decompose_cells_amplitudes_only(
        eis_by_cell_id,
        compute_device,
        initial_basis,
        l1_regularize_lambda=args.weight_reg,
        output_debug_dict=False,
        shifts=shift_tuple,
        supersample_factor=args.upsample,
        snr_abs_threshold=args.thresh,
        grid_search_step=args.grid_step,
        grid_search_top_n=args.grid_top_n,
        fine_search_width=args.fine_search_width,
        grid_search_batch_size=args.grid_batch_size,
        use_scaled_mse_penalty=args.renormalize_loss,
        use_scaled_regularization_terms=args.renormalize_penalty,
        use_grouped_l1l2_norm=args.group,
        grouped_l1l2_groups=group_assignments,
        use_basis_weighted_l1_norm=args.l1_comp_weights,
        basis_weights_for_l1=componentwise_weights
    )

    print(mse)

    with open(args.output, 'wb') as joint_fit_file:
        metadata_dict = {
            'l1_reg': args.weight_reg,
            'padding': shift_tuple,
            'upsample': args.upsample,
            'thresh': args.thresh
        }

        pickle_dict = {
            'decomposition': decomposition_dict,
            'waveforms': basis_waveforms,
            'mse': mse
        }

        pickle.dump(metadata_dict, joint_fit_file)
        pickle.dump(pickle_dict, joint_fit_file)

